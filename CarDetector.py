
import ppdet.utils.checkpoint as checkpoint
from ppdet.utils.cli import ArgsParser
from ppdet.utils.eval_utils import parse_fetches
from ppdet.core.workspace import load_config, merge_config, create
from paddle import fluid
import os
import cv2
import glob
import time
import hyperlpr

from ppdet.utils.coco_eval import bbox2out, mask2out, get_category_info

import numpy as np
from PIL import Image
from PIL import ImageFont, ImageDraw

from sort import Sort

font_path = r'./simsun.ttc'
font = ImageFont.truetype(font_path, 32)


def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


parser = ArgsParser()
parser.add_argument(
    "--size",
    type=int,
    default=608)
FLAGS = parser.parse_args()


def draw_bbox(image, catid2name, bboxes, threshold):

    for dt in np.array(bboxes):

        catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
        if score < threshold or catid == 6:
            continue

        xmin, ymin, w, h = bbox
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmin + w)
        ymax = int(ymin + h)

        # draw label
        # text = "{} {:.2f}".format(catid2name[catid], score)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
        # image = putText(image, text, xmin, ymin)

    return image


def process_box(image, catid2name, bboxes, threshold):

    boxes = []

    for dt in np.array(bboxes):

        catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
        if score < threshold or catid == 6:
            continue

        xmin, ymin, w, h = bbox
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmin + w)
        ymax = int(ymin + h)

        boxes.append((xmin, ymin, xmax, ymax))

    return boxes


class VehicleDetector(object):

    def __init__(self):

        self.draw_threshold = 0.06

        self.cfg = load_config('./configs/vehicle_yolov3_darknet.yml')

        merge_config(FLAGS.opt)

        self.place = fluid.CUDAPlace(
            0) if self.cfg.use_gpu else fluid.CPUPlace()
        self.exe = fluid.Executor(self.place)

        self.model = create(self.cfg.architecture)

        self.init_params()

    def init_params(self):

        startup_prog = fluid.Program()
        infer_prog = fluid.Program()
        with fluid.program_guard(infer_prog, startup_prog):
            with fluid.unique_name.guard():
                inputs_def = self.cfg['TestReader']['inputs_def']
                inputs_def['iterable'] = True
                feed_vars, loader = self.model.build_inputs(**inputs_def)
                test_fetches = self.model.test(feed_vars)
        infer_prog = infer_prog.clone(True)

        self.exe.run(startup_prog)
        if self.cfg.weights:
            checkpoint.load_params(self.exe, infer_prog, self.cfg.weights)

        extra_keys = ['im_info', 'im_id', 'im_shape']
        self.keys, self.values, _ = parse_fetches(
            test_fetches, infer_prog, extra_keys)
        dataset = self.cfg.TestReader['dataset']
        anno_file = dataset.get_anno()
        with_background = dataset.with_background
        use_default_label = dataset.use_default_label

        self.clsid2catid, self.catid2name = get_category_info(anno_file, with_background,
                                                              use_default_label)

        is_bbox_normalized = False
        if hasattr(self.model, 'is_bbox_normalized') and \
                callable(self.model.is_bbox_normalized):
            is_bbox_normalized = self.model.is_bbox_normalized()

        self.is_bbox_normalized = is_bbox_normalized

        self.infer_prog = infer_prog

    def process_img(self, img):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        shape = img.shape[:2]

        img = cv2.resize(img, (FLAGS.size, FLAGS.size))

        # RBG img [224,224,3]->[3,224,224]
        img = img[:, :, ::-1].astype('float32').transpose((2, 0, 1)) / 255
        img_mean = np.array(mean).reshape((3, 1, 1))
        img_std = np.array(std).reshape((3, 1, 1))
        img -= img_mean
        img /= img_std

        img = img.astype('float32')
        img = np.expand_dims(img, axis=0)

        shape = np.expand_dims(np.array(shape), axis=0)
        im_id = np.zeros((1, 1), dtype=np.int64)

        return img, im_id, shape

    def predict(self, img):

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raw = img.copy()
        img, im_id, shape = self.process_img(img=img)
        outs = self.exe.run(self.infer_prog,
                            feed={'image': img, 'im_size': shape, 'im_id': im_id},
                            fetch_list=self.values,
                            return_numpy=False)
        res = {
            k: (np.array(v), v.recursive_sequence_lengths())
            for k, v in zip(self.keys, outs)
        }

        bbox_results = bbox2out(
            [res], self.clsid2catid, self.is_bbox_normalized)

        result = process_box(raw, self.catid2name,
                             bbox_results, self.draw_threshold)

        return result


class Car_DC():
    '''
    车辆检测和类型识别
    '''

    def __init__(self, line=[(263, 408), (691, 436)]):

        self.detector = VehicleDetector()

        self.tracker = Sort()
        self.memory = {}
        self.line = line
        self.counter = 0

        self.count_info = ''

    def drawTest(self, image, addText, x1, y1):

        img = Image.fromarray(image)
        draw = ImageDraw.Draw(img)
        draw.text((x1, y1),
                  addText.encode("utf-8").decode("utf-8"),
                  (0, 255, 255), font=font)
        imagex = np.array(img)

        return imagex

    def get_time(self, num, name):

        out = '\n{}年 {}月 {}日\n时间：{}\n计数：{}\n车牌：{}'
        t = time.localtime()
        out = out.format(
            t.tm_year, t.tm_mon, t.tm_mday,
            str(t.tm_hour)+':'+str(t.tm_min)+':'+str(t.tm_sec),
            num, name
        )

        return out

    def detect_count(self, img):

        try:

            raw = img.copy()

            bboxes = self.detector.predict(img)

            info = (None, None)

            boxes = []

            for box in bboxes:
                x1, y1, x2, y2 = box
                boxes.append([x1, y1, x2, y2, 1])

            np.set_printoptions(
                formatter={'float': lambda x: "{0:0.3f}".format(x)})

            dets = np.asarray(boxes).astype('int')

            if dets.shape[0] != 0:
                tracks = self.tracker.update(dets)

                boxes = []
                indexIDs = []
                c = []
                previous = self.memory.copy()
                self.memory = {}

                for track in tracks:
                    boxes.append([track[0], track[1], track[2], track[3]])
                    indexIDs.append(int(track[4]))
                    self.memory[indexIDs[-1]] = boxes[-1]

                if len(boxes) > 0:
                    i = int(0)
                    for box in boxes:

                        try:
                            (x, y) = (int(box[0]), int(box[1]))
                            (w, h) = (int(box[2]), int(box[3]))
                        except Exception as e:
                            print(e)
                            continue

                        cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)

                        if indexIDs[i] in previous:
                            previous_box = previous[indexIDs[i]]
                            (x2, y2) = (
                                int(previous_box[0]), int(previous_box[1]))
                            (w2, h2) = (
                                int(previous_box[2]), int(previous_box[3]))
                            p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                            p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                            cv2.line(img, p0, p1, (0, 255, 0), 3)

                            if intersect(p0, p1, self.line[0], self.line[1]):
                                self.counter += 1
                                roi = raw[y:h, x:w, :].astype('uint8')
                                plates = hyperlpr.HyperLPR_plate_recognition(
                                    roi)
                                if len(plates) > 0:
                                    info = (str(self.counter), str(plates[0][0]))
                                else:
                                    info = (str(self.counter), '车牌模糊')
                        text = "{}".format(indexIDs[i])
                        cv2.putText(img, text, (x, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        i += 1

                if not info[0] is None:
                    count, plate = info
                    self.count_info = self.get_time(count, plate)

        except Exception as e:
            print(e)

        cv2.putText(img, str(self.counter), (0, 100),
                            cv2.FONT_HERSHEY_DUPLEX, 4.0, (0, 255, 255), 10)

        cv2.line(img, self.line[0], self.line[1], (0, 255, 255), 5)
        img = self.drawTest(img, self.count_info, 0, 100)

        return img


if __name__ == '__main__':

    import cv2

    det = VehicleDetector()

    im = cv2.imread('./car.jpg')
    result = det.detect(im)

    cv2.imshow('a', result)
    cv2.waitKey(0)

    cv2.destroyAllWindows(0)
