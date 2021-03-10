# Importing required libraries, obviously
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
import dlib
from imutils import face_utils


def detect_landmark(image):

    # --------------------------------
    # 1.顔ランドマーク検出の前準備
    # --------------------------------
    # 顔検出ツールの呼び出し
    face_detector = dlib.get_frontal_face_detector()

    # 顔のランドマーク検出ツールの呼び出し
    predictor_path = 'shape_predictor_5_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)

    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_gry = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2GRAY)

    # --------------------------------
    # 2.顔のランドマーク検出
    # --------------------------------
    # 顔検出
    # ※2番めの引数はupsampleの回数。基本的に1回で十分。
    faces = face_detector(img_gry, 1)

    # 検出した全顔に対して処理
    for face in faces:
        landmark = face_predictor(img_gry, face)
        # 処理高速化のためランドマーク群をNumPy配列に変換(必須)
        landmark = face_utils.shape_to_np(landmark)

        # ランドマーク描画
        for (i, (x, y)) in enumerate(landmark):
            cv2.circle(img_bgr, (x, y), 1, (255, 0, 0), -1)

            cv2.putText(img_bgr, str(i + 1), (x - 10, y - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), faces



def about():
    st.write(
        '''
        **Algorithm description and references**. 

        ランドマーク検出手法:

            1. Ensemble of regression treesを用いた手法
                回帰ツリー分析を用いてリアルタイムで高精度なランドマーク検出を実現
            2. Active appearance modelを用いた手法
                物体の形状と外観から学習された統計モデルに基づき物体検出を行う
            3. Local Binary Featuresを用いた手法
                回帰学習により非常に高速なランドマーク検出が可能




Read more :point_down:
    1.[pythonとdlibでお手軽に顔のランドマークを検出してみた](https://qiita.com/mimitaro/items/bbc58051104eafc1eb38#1ensemble-of-regression-trees%E3%82%92%E7%94%A8%E3%81%84%E3%81%9F%E6%89%8B%E6%B3%95)
    2.[顔のランドマーク（68 ランドマーク）の取得（Python，Dlib を使用）](https://www.kkaneko.jp/dblab/dlib/landmark.html)
    3.[ibug](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
        ''')


def main():
    st.title(":fire: :waxing_gibbous_moon: Face Detection App :waning_gibbous_moon: :fire:")
    st.write("**Dlib : Ensemble of regression trees**")

    activities = ["Home", "About"]
    choice = st.sidebar.selectbox("Sidebar", activities)

    if choice == "Home":

        st.write("Go to the About section from the sidebar to learn more about it.")
        
        # You can specify more file types below if you want
        image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])

        if image_file is not None:

            image = Image.open(image_file)
            st.image(image)

            if st.button("Process"):
                result_img, faces = detect_landmark(image=image)
                st.image(result_img)
                st.success("Found {} faces\n".format(len(faces)))

    elif choice == "About":
        about()




if __name__ == "__main__":
    main()
