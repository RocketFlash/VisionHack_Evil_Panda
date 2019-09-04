import cv2
import math
import numpy as np

def detect(frame, focus=None):
    img = cv2.resize(frame, (0,0), fx=0.333, fy=0.333) 
    window = 70
    if focus is None:
        focus = (int(img.shape[0] / 2)  + 100, int(img.shape[1] / 2))
    else:
        focus = (int(float(focus[0]) * 0.33), int(float(focus[1])* 0.33) )  

    h, w = img.shape[0], img.shape[1]
    
    src = np.array([
            [focus[0] - window, focus[1] + 20],
            [focus[0] + window, focus[1] + 20],
            [0, h],
            [w, h]
        ],np.float32)

    dst = np.array([
            [0, 0],
            [w, 0],
            [0, h],
            [w, h]
        ], np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(img.copy(), M, (img.shape[1], img.shape[0]))

    warp = cv2.equalizeHist(warp)

    sobelx64f = cv2.Sobel(warp, cv2.CV_64F, 2, 0, ksize=1)
    abs_sobel64f = np.absolute(sobelx64f)
    edges = np.uint8(abs_sobel64f)        
    (thresh, im_bw) = cv2.threshold(edges, 15, 255, cv2.THRESH_BINARY)

    index = int(0.76 * warp.shape[0])
    cnct = im_bw[index] & im_bw[index + 8]
    cnt = np.count_nonzero(cnct)

    answer = cnt > 20
    
    return answer