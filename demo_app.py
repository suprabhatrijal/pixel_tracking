import streamlit as st
import os
import cv2 as cv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_cap(uploaded_file):
    return cv.VideoCapture(uploaded_file)

st.title("OpenCV Demo App")
st.subheader("This app allows you to play with Image filters!")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mpeg", "wmv"])


@st.cache_data
def process_video(uploaded_file):
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    save_path = Path(f"./{uploaded_file.name}")
    with open(save_path, mode="wb") as w:
        w.write(uploaded_file.getvalue())

    cap = cv.VideoCapture(f"./{uploaded_file.name}")

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

    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4)) 
   

    # temp_file_name = 'filename.avi'
    # result = cv.VideoWriter(temp_file_name,  
    #                      cv.VideoWriter_fourcc(*'MJPG'), 
    #                      10, size) 


    ref_frame = None
    with st.spinner('Wait for it...'):
        while(1):
            ret, frame = cap.read()
            if not ret:
                print('No frames grabbed!')
                break

            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # calculate optical flow
            p1, stt, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select good points
            if p1 is not None:
                good_new = p1[stt==1]
                good_old = p0[stt==1]

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

            # result.write(img)
            if  ref_frame is None:
                ref_frame = img


            k = cv.waitKey(30) & 0xff
            if k == 27:
                break

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)


        # result.release()
        cv.destroyAllWindows()




        for i in range(len(x)):
            df[f"x{i}"] = x[i]
            df[f"y{i}"] = y[i]

        os.remove(save_path)
        return (df, ref_frame,  x, y,fps)


if uploaded_file is not None:
        df, ref_frame, x, y, fps = process_video(uploaded_file)

        csv = df.to_csv().encode('utf-8')

        st.download_button(
            "Press to Download CSV file",
            csv,
            "file.csv",
            "text/csv",
            key='download-csv'
        )
        ref_frame_data = cv.imencode('.png', ref_frame)[1].tobytes()
        st.download_button(label='Download Image',
                        data= ref_frame_data,
                        file_name='ref_frame.png',
                        mime='image/png')

        st.text("reference image")
        st.image(ref_frame_data)

        fig_arr = []
        for i in range(len(x)):
            frameX = [i/fps for i in range(len(x[i]))]
            frameY = [i/fps for i in range(len(x[i]))]

            figureX = plt.figure()
            plt.title(f"Graph for point {i} in the X axis")
            plt.xlabel("time/sec")
            plt.ylabel("postion/pixel")
            plt.plot(frameX,x[i])


            figureY = plt.figure()
            plt.title(f"Graph for point {i} in the Y axis")
            plt.xlabel("time/sec")
            plt.ylabel("postion/pixel")
            plt.plot(frameY,y[i])

            fig_arr.append([figureX, figureY])
        
        st.selectbox(
                'Choose point',
                (i for i in range(len(x))),
                key="points_opt"
                )

        st.pyplot(fig_arr[st.session_state.points_opt][0])
        st.pyplot(fig_arr[st.session_state.points_opt][1])

        
