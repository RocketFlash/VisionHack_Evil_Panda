import cv2
import numpy as np


def detect(frame, focus=None):
    img = cv2.resize(frame, (0, 0), fx=0.333, fy=0.333)
    window = 70
    if focus is None:
        focus = (int(img.shape[0] / 2) + 100, int(img.shape[1] / 2))
    else:
        focus = (int(float(focus[0]) * 0.33), int(float(focus[1]) * 0.33))

    h, w = img.shape[0], img.shape[1]
    shiftBottom = 10
    shiftUp = 10
    src = np.array([
        [focus[0] - window, focus[1]+shiftUp],
        [focus[0] + window, focus[1]+shiftUp],
        [w-shiftBottom, h],
        [0+shiftBottom, h]
    ], np.float32)

    dst = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], np.float32)


    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(img.copy(), M, (w, h))

    warp = cv2.equalizeHist(warp)
    warp = cv2.medianBlur(warp, 5)

    cv2.imshow('warp',warp)
    cv2.waitKey(1)
    sobelx64f = cv2.Sobel(warp, cv2.CV_64F, 2, 0, ksize=1)
    abs_sobel64f = np.absolute(sobelx64f)
    edges = np.uint8(abs_sobel64f)
    (thresh, im_bw) = cv2.threshold(edges, 15, 255, cv2.THRESH_BINARY)
    # im_bw = cv2.adaptiveThreshold(warp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C|cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,0)
    im_bw = cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, (5, 5))


    (thresh, im_bw) = cv2.threshold(edges, 15, 255, cv2.THRESH_BINARY)

    index = int(0.76 * warp.shape[0])
    cnct = im_bw[index] & im_bw[index + 8]
    cnt = np.count_nonzero(cnct)

    answer = cnt > 20
    return answer


textFile = open('/Users/YagfarovRauf/Desktop/trainset/train.txt', "r")
lines = textFile.readlines()


for f in lines:
    FOPFile = open('/Users/YagfarovRauf/Desktop/trainset/' + f.split(" ")[0][0:-9] + '.txt', "r")
    FOP = list(map(int, filter(None, FOPFile.readline().split(" "))))

    print(f)
    if int(f.split(" ")[1][5]) != 1:
        continue

    cap = cv2.VideoCapture("/Users/YagfarovRauf/Desktop/trainset/" + f.split(" ")[0])

    while (1):
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(detect(frame,tuple(FOP)))
