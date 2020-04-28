import cv2

import numpy as np

if __name__ == '__main__':

    cap = cv2.VideoCapture('E:\\highway.mp4')

    video_width = int(cap.get(3)/2)
    video_height = int(cap.get(4)/3)
    fps = int(cap.get(5))
    # fps = 15
    print(fps)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') #opencv3.0
    videoWriter = cv2.VideoWriter(
        'cut.mp4', fourcc, fps, (video_width, video_height*2))

    while True:

        _, im = cap.read()
        if im is None:
            break
        result = im[video_height:, :video_width, :]
        cv2.imshow('a', result)
        videoWriter.write(result)
        cv2.waitKey(1)

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()
