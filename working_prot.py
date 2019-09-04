import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import imutils
from scipy.spatial.distance import euclidean
import glob

from libs.crosswalk.barcode import detect as cw_detect

def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]

def ang(lineA, lineB):
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    dot_prod = dot(vA, vB)
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    cos_ = dot_prod/magA/magB
    angle = math.acos(dot_prod/magB/magA)
    ang_deg = math.degrees(angle)%360

    if ang_deg-180>=0:
        return 360 - ang_deg
    else: 
        return ang_deg


focus_file = ""
zebra_files = [
    "akn.292.154.left.avi",
    "akn.294.156.left.avi",
    "akn.295.027.left.avi",
    "akn.289.069.left.avi",
    "akn.289.008.left.avi",
    "akn.282.083.left.avi",
    "akn.283.065.left.avi",
    "akn.283.175.left.avi",
    "akn.273.014.left.avi",
    "akn.273.056.left.avi",
    "akn.273.074.left.avi",
    "akn.275.015.left.avi",
    "akn.275.115.left.avi",
    "akn.279.026.left.avi",
    "akn.280.006.left.avi"
]

# zebra_files = [
#     "akn.233.113.left.avi",
#     "akn.217.019.left.avi",
#     "akn.233.148.left.avi",
#     "akn.217.104.left.avi",
#     "akn.250.022.left.avi" 
# ]

for p in glob.glob("data_converted/**/**.jpg"):
    _, f, _ = p.split("/")

    if f not in zebra_files: continue

    

    if focus_file != f:
        f = f.replace(".left.avi", ".txt")

        with open("data/trainset/" + f) as of:
            focus = of.readline().split(" ")
            #print(focus)  
            focus = (int(float(focus[0]) * 0.33), int(float(focus[1])* 0.33) )  

    img = cv2.imread(p, 0)
    img = cv2.resize(img, (0,0), fx=0.333, fy=0.333) 
 
    hist,bins = np.histogram(img.flatten(),256,[0,256])

    window = 70
    h, w = img.shape[0], img.shape[1]

    cv2.circle(img, focus, 3, (255, 255, 255))
    cv2.circle(img, (focus[0] + window, focus[1] + 20), 3, (255, 255, 255))
    cv2.circle(img, (focus[0] - window, focus[1] + 20), 3, (255, 255, 255))
    cv2.circle(img, focus, 3, (255, 255, 255))
    cv2.circle(img, focus, 3, (255, 255, 255))
    
    src = np.array([
            [focus[0] - window, focus[1] + 20],
            [focus[0] + window, focus[1] + 20],
            [0, h],
            [w, h]
        ],np.float32)

    dst = np.array([
            [0,0],
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

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,200)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    index = int(0.76 * warp.shape[0])
    cnct = im_bw[index] & im_bw[index + 16]
    cnt = np.count_nonzero(cnct)

    cv2.line(warp, (0, index), (warp.shape[1], index), (255,255,255), 2)
    cv2.line(warp, (0, index + 16), (warp.shape[1], index + 16), (255,255,255), 2)

    answer = "crosswalk" if cnt > 12 else "-"
    cv2.putText(warp, answer, (10,295), font, fontScale, fontColor, lineType)
    cv2.putText(warp, str(cnt), (10,275), font, fontScale, fontColor, lineType)

    cv2.imshow('frame', warp)
    cv2.imshow('frame2', im_bw)


    cv2.waitKey(1)