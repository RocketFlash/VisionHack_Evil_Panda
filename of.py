import numpy as np
import cv2
import glob

fgbg = cv2.createBackgroundSubtractorMOG2(7, 8000, False)
textFile = open('/Users/YagfarovRauf/Desktop/trainset/train.txt', "r")
lines = textFile.readlines()
minValToDetect = 1400
success = 0
total = 0
i = -1

with open('result.txt', 'a') as the_file:

# for f in lines:
    for f in glob.glob("/Users/YagfarovRauf/Desktop/validationset/**.avi"):
        # if int(f.split(" ")[1][0]) != 1:
        #     if int(f.split(" ")[1][4]) != 1:
        #         continue
        print("File: " + f.split('/')[-1])
        # cap = cv2.VideoCapture("/Users/YagfarovRauf/Desktop/trainset/" + f.split(" ")[0])
        cap = cv2.VideoCapture(f)
        ourRes = 0
        fr = 0
        maxCount = 0
        cntr = 0
        while (1):
            fr += 1
            ret, frame = cap.read()

            if ret == False:
                break
            frame = cv2.resize(frame, (int(frame.shape[1] / 6), int(frame.shape[0] / 6)))
            frame = frame[0:int(frame.shape[1] / 2.5), :]
            fgmask = fgbg.apply(frame)
            if fr < 3:
                continue
            # fgmask[fgmask>120] = 255
            # fgmask[fgmask<=120] = 0
            # fgmask = cv2.erode(fgmask, (5, 5))
            # fgmask = cv2.dilate(fgmask,(5,5))
            # fgmask = cv2.erode(fgmask, (5, 5))

            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, (7, 7))
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, (7, 7))

            count = fgmask[fgmask != 0].shape[0]

            # cv2.imshow('frame',fgmask)
            # cv2.waitKey(1)
            # cv2.imshow('frameTr', frame)
            # cv2.waitKey(1)

            if maxCount < count:
                maxCount = count

            # print(count)
            if count > minValToDetect:
                cntr += 1

            if cntr >= 2:
                ourRes = 1
                break
        print("Max count:" + str(maxCount))
        cap.release()
        # trueRes = int(f.split(" ")[1][4])
        # total += 1
        # print('Our res: ' + str(ourRes) + ' True res: ' + str(trueRes) + '\n')
        # if ourRes == trueRes:
        #     success += 1
        the_file.write(f.split('/')[-1]+' '+'0000'+str(ourRes)+'0\n')
# print("Our precision: "+ str(100*float(success)/total) + "%")
    # print("Our result: " + str(ourRes))
textFile.close()


def detectWipers(fgbg, frameNo, frame):
    frame = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))
    frame = frame[0:int(frame.shape[1] / 2.5), :]
    if frameNo < 3:
        return 0
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, (7, 7))
    count = fgmask[fgmask != 0].shape[0]
    return count

# cap = cv2.VideoCapture("/Users/YagfarovRauf/Desktop/trainset/akn.031.029.left.avi")
#
# while (1):
#     ret, frame = cap.read()
#     if ret == False:
#         break
#     fgmask = fgbg.apply(frame)
#     fgmask[fgmask > 120] = 255
#     fgmask[fgmask <= 120] = 0
#     fgmask = cv2.resize(fgmask, (int(fgmask.shape[1] / 2), int(fgmask.shape[0] / 2)))
#     cv2.imshow('frame',fgmask)
#     cv2.waitKey(0)
