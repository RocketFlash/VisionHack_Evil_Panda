import cv2
import numpy as np
import glob

textFile = open('/Users/YagfarovRauf/Desktop/trainset/train.txt', "r")
lines = textFile.readlines()

surf = cv2.xfeatures2d.SIFT_create()

signImg = cv2.imread('lezh.jpg')
signImg = cv2.resize(signImg,(int(signImg.shape[1] / 2), int(signImg.shape[0] / 2)))
# signImg = cv2.medianBlur(signImg,11)
# find the keypoints and descriptors with SIFT
kp1, des1 = surf.detectAndCompute(signImg,None)
MIN_MATCH_COUNT = 7

success = 0
total = 0

bf = cv2.FlannBasedMatcher_create()
t = 0
with open('resultSign.txt', 'a') as the_file:
    for f in glob.glob("/Users/YagfarovRauf/Desktop/validationset/**.avi"):
        t+=1
        if t <170:
            continue
        print(f)
        # if int(f.split(" ")[1][3]) != 1:
        #     continue
        cap = cv2.VideoCapture(f)
        found = False
        frameNo = 0
        ourRes = 0
        while (1):
            ret, frame = cap.read()
            frameNo+=1
            if ret == False:
                break
            frame = cv2.resize(frame, (int(frame.shape[1] / 3), int(frame.shape[0] / 3)))
            frame = frame[0:int(frame.shape[0]/1.5),int(frame.shape[1]/2):]
            # signIm = cv2.medianBlur(signImg,11)
            kp2, des2 = surf.detectAndCompute(frame, None)

            if (len(kp1) >= 2 and len(kp2) >= 2):
                matches = bf.knnMatch(des1, des2, k=2)
            else:
                continue

            # Apply ratio test
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append([m])

            if len(good) > MIN_MATCH_COUNT:
                    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    matchesMask = mask.ravel().tolist()
                    if M is not None:
                        h, w,c = signImg.shape
                        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)
                        frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
                        found = True
                        print("!!!!!FOUND!!!!!")
                    else:
                        print("No homography :(")


            else:
                    # print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
                    matchesMask = None

            if found and frameNo<250:
                ourRes = 1
                break
            # draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
            #                    singlePointColor=None,
            #                    matchesMask=matchesMask,  # draw only inliers
            #                    flags=2)
            # # cv2.drawMatchesKnn expects list of lists as matches.
            # img3 = cv2.drawMatchesKnn(signImg, kp1, frame, kp2, good,None,flags=2)


            # cv2.imshow('frame', img3)
            # cv2.waitKey(1)

        # total+=1
        # if ourRes==1:
        #     success+=1
        # trueRes = int(f.split(" ")[1][4])
        # print('Our res: ' + str(ourRes) + ' True res: ' + str(trueRes) + '\n')
        # if ourRes == trueRes:
        #     success += 1
        the_file.write(f.split('/')[-1] + ' ' + '000' + str(ourRes) + '00\n')
print("Our precision: "+ str(100*float(success)/total) + "%")
textFile.close()