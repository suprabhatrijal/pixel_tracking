import numpy as np
import cv2 as cv
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                              # The example file can be downloaded from: \
                                              # https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
# parser.add_argument('image', type=str, help='path to image file')
# args = parser.parse_args()

cap = cv.VideoCapture("./cam1.wmv")

font = cv.FONT_HERSHEY_SIMPLEX
fps = cap.get(cv.CAP_PROP_FPS)
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 10,
                       qualityLevel = 0.2,
                       minDistance = 10,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (500, 700),
                  maxLevel = 0,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

df = pd.DataFrame()

x= None
y= None

while(1):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]

    # draw the tracks
    x = [[] for _ in good_new] if x is None else x
    y = [[] for _ in good_new] if y is None else y
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        x[i].append(a)
        y[i].append(b)
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        cv.putText(frame, str(i), (int(a-25), int(b-25)), font, 1.25, (255, 255, 255), 3)

    img = cv.add(frame, mask)

    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv.destroyAllWindows()
for i in range(len(x)):
    df[f"x{i}"] = x[i]
    df[f"y{i}"] = y[i]

df.to_csv("./results/cam1/all_points.csv")

for i in range(len(x)):
    frame = [i/fps for i in range(len(x[i]))]
    plt.figure()
    plt.xlabel("time/sec")
    plt.ylabel("postion/pixel")
    plt.plot(frame,x[i])
    plt.savefig(f"./results/cam1/x{i}.png")

for i in range(len(y)):
    frame = [i/fps for i in range(len(y[i]))]
    plt.figure()
    plt.xlabel("time/sec")
    plt.ylabel("postion/pixel")
    plt.plot(frame,y[i])
    plt.savefig(f"./results/cam1/y{i}.png")
