import math
import pandas as pd
import glob
import os
import numpy as np
from feat import Detector
import cv2


def smile_extractor(images_path):
    face_cascade = cv2.CascadeClassifier('./Video/haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier('./Video/haarcascade_smile.xml')

    smile_lst = [0, 0]
    for image_file in glob.glob(os.path.join(images_path, '*.jpg')):
        image = cv2.imread(image_file)
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame)

        smile_or_not = 0
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (100, 200, 50), 4)
            face_roi = gray_frame[y:y + h, x:x + w]
            smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=20)
            if len(smiles) > 0:
                smile_or_not = 1

        if smile_or_not:
            smile_lst[1] += 1
        else:
            smile_lst[0] += 1

    smile = smile_lst[1]
    not_smile = smile_lst[0]

    if smile > not_smile:
        return 1
    else:
        return 0


def pose_extractor(images_path):
    detector = Detector(
        face_model="retinaface",
        landmark_model="mobilefacenet",
        au_model='xgb',
        emotion_model="resmasknet",
        facepose_model="img2pose",
        device="cpu"
    )

    pose_lst = ['Pitch', 'Roll', 'Yaw']
    pose_df = pd.DataFrame(columns=pose_lst)

    for image_file in glob.glob(os.path.join(images_path, '*.jpg')):
        single_face_prediction = detector.detect_image(image_file)
        pose_prediction = single_face_prediction.facepose

        news_df = pd.DataFrame({'Pitch': pose_prediction['Pitch'], 'Roll': pose_prediction['Roll'], 'Yaw': pose_prediction['Yaw']}, index=[0])
        pose_df = pd.concat([pose_df, news_df], ignore_index=True)

    final_pose_dict = {}

    for pose in pose_lst:
        final_pose_dict[f"mean_{pose}"] = pose_df[pose].mean()
        final_pose_dict[f"std_{pose}"] = pose_df[pose].std()

    return final_pose_dict


def gender_extractor(xlsx_path, video_name):
    information = pd.read_excel(xlsx_path)
    information.set_index("VideoID", inplace=True)
    return information.loc[int(video_name), "CEOGender"]


def video_features_extractor(videos_path, large_images_path, video_name, skip_frames=24):
    detector = Detector(
        face_model="retinaface",
        landmark_model="mobilefacenet",
        au_model='xgb',
        emotion_model="resmasknet",
        facepose_model="img2pose",
        device="cpu"
    )
    

    retrieve_list = ["FaceScore", "anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

    # gender
    xlsx_path = '../Transcript/Original_Videos_List.xlsx'
    gener = gender_extractor(xlsx_path, video_name)

    video_df = pd.DataFrame()
    videos_lst = os.listdir(videos_path)
    for video_file in videos_lst:
        try:
            video_file = os.path.join(videos_path, video_file)
            file_name = os.path.splitext((video_file.split("/")[-1]).split('\\')[-1])[0]
            
            old_video_df = pd.read_csv(f'../Features/{video_name}/{video_name}_video_local.csv', index_col=0)

            if file_name not in old_video_df.index:
                video_prediction = detector.detect_video(video_file, skip_frames)
                features = video_prediction[retrieve_list]

                # smile
                images_path = f"{large_images_path}/{file_name.split('_')[0]}/{file_name}"
                new_video = {"video_Score": features["anger"].count() / len(features), "smile_or_not": smile_extractor(images_path), "gender": gener}
                new_video.update(pose_extractor(images_path))

                for index in range(len(retrieve_list)):
                    feature = retrieve_list[index]
                    new_video[f"mean_{feature}"] = features[feature].mean()
                    new_video[f"std_{feature}"] = features[feature].std()

                if int(new_video["video_Score"]) == 0:
                    continue
                new_df = pd.DataFrame(new_video, index=[file_name])
                print(new_df)
                video_df = pd.concat([video_df, new_df])
        except:
            pass
    return video_df






