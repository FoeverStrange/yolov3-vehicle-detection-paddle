
from ppdet.data.reader import create_reader
import ppdet.utils.checkpoint as checkpoint
from ppdet.utils.visualizer import visualize_results
from ppdet.utils.check import check_gpu, check_version
from ppdet.utils.cli import ArgsParser
from ppdet.utils.eval_utils import parse_fetches
from ppdet.core.workspace import load_config, merge_config, create
from paddle import fluid
import os
import glob

from ppdet.utils.coco_eval import bbox2out, mask2out, get_category_info

import numpy as np
from PIL import Image

parser = ArgsParser()
parser.add_argument(
    "--infer_img",
    type=str,
    default='car.jpg',
    help="Image path, has higher priority over --infer_dir")
parser.add_argument(
    "--draw_threshold",
    type=float,
    default=0.5,
    help="Threshold to reserve the result for visualization.")
FLAGS = parser.parse_args()

def main():

    draw_threshold = 0.2

    cfg = load_config('./vehicle_yolov3_darknet.yml')

    main_arch = cfg.architecture

    merge_config(FLAGS.opt)

    # check if set use_gpu=True in paddlepaddle cpu version
    check_gpu(cfg.use_gpu)
    # check if paddlepaddle version is satisfied
    check_version()

    dataset = cfg.TestReader['dataset']

    test_images = [FLAGS.infer_img]
    dataset.set_images(test_images)

    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    model = create(main_arch)

    startup_prog = fluid.Program()
    infer_prog = fluid.Program()
    with fluid.program_guard(infer_prog, startup_prog):
        with fluid.unique_name.guard():
            inputs_def = cfg['TestReader']['inputs_def']
            inputs_def['iterable'] = True
            feed_vars, loader = model.build_inputs(**inputs_def)
            test_fetches = model.test(feed_vars)
    infer_prog = infer_prog.clone(True)

    reader = create_reader(cfg.TestReader, devices_num=1)
    loader.set_sample_list_generator(reader, place)

    exe.run(startup_prog)
    if cfg.weights:
        checkpoint.load_params(exe, infer_prog, cfg.weights)

    extra_keys = ['im_info', 'im_id', 'im_shape']
    keys, values, _ = parse_fetches(test_fetches, infer_prog, extra_keys)

    anno_file = dataset.get_anno()
    with_background = dataset.with_background
    use_default_label = dataset.use_default_label

    clsid2catid, catid2name = get_category_info(anno_file, with_background,
                                                use_default_label)

    is_bbox_normalized = False
    if hasattr(model, 'is_bbox_normalized') and \
            callable(model.is_bbox_normalized):
        is_bbox_normalized = model.is_bbox_normalized()

    imid2path = dataset.get_imid2path()
    for iter_id, data in enumerate(loader()):
        im = np.array(data[0]['image'])
        size = np.array(data[0]['im_size'])
        id = np.array(data[0]['im_id'])
        outs = exe.run(infer_prog,
                       feed=data,
                       fetch_list=values,
                       return_numpy=False)
        res = {
            k: (np.array(v), v.recursive_sequence_lengths())
            for k, v in zip(keys, outs)
        }

        bbox_results = None
        mask_results = None
        if 'bbox' in res:
            bbox_results = bbox2out([res], clsid2catid, is_bbox_normalized)
        if 'mask' in res:
            mask_results = mask2out([res], clsid2catid,
                                    model.mask_head.resolution)

        # visualize result
        im_ids = res['im_id'][0]
        for im_id in im_ids:
            image_path = imid2path[int(im_id)]
            image = Image.open(image_path).convert('RGB')

            image = visualize_results(image,
                                      catid2name,
                                      draw_threshold, bbox_results,
                                      mask_results)
            image.show()


if __name__ == '__main__':

    main()
