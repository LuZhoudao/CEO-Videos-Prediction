import os
import pandas as pd
import numpy as np

def padding(feature, MAX_LEN):
    """
    mode:
        zero: padding with 0
        normal: padding with normal distribution
    location: front / back
    """
#     assert self.padding_mode in ['zeros', 'normal']
#     assert self.padding_location in ['front', 'back']

    length = feature.shape[0]
#     if length >= MAX_LEN:
#         return feature[:MAX_LEN, :]

    pad = np.zeros([1, MAX_LEN-feature.shape[1]])


    feature = np.concatenate((feature, pad), axis=0)
    return feature

feature_path = "../Features"
features_dir_lst = os.listdir(feature_path)
for features_dir in features_dir_lst:
    features_dir_path = os.path.join(feature_path, features_dir)
    global_features_path = os.path.join(features_dir_path, "Global_features")
    global_features_files = os.listdir(global_features_path)

    local_video_path = os.path.join(features_dir_path, f"{features_dir}_video_local.csv")
    local_audio_path = os.path.join(features_dir_path, f"{features_dir}_audio_local.csv")
    local_text_path = os.path.join(features_dir_path, f"{features_dir}_text_local.csv")

    local_video_df = pd.read_csv(local_video_path, index_col=["Unnamed: 0"])
    local_audio_df = pd.read_csv(local_audio_path, index_col=["Unnamed: 0"])
    local_text_df = pd.read_csv(local_text_path, index_col=["Unnamed: 0"])

    for global_features in global_features_files:
        small_global_features_path = os.path.join(global_features_path, global_features)

        global_video_features = np.load(os.path.join(small_global_features_path, f"{global_features}_video_global.npy"))
        local_video_features = np.array(local_video_df.loc[global_features])

        global_audio_features = np.load(os.path.join(small_global_features_path, f"{global_features}_audio_global.npy"))
        local_audio_features = np.array(local_audio_df.loc[global_features])

        global_text_features = np.load(os.path.join(small_global_features_path, f"{global_features}_text_global.npy"))
        local_text_features = np.array(local_text_df.loc[global_features])
        padding_local_video_features = padding(local_video_features.reshape((1, local_video_features.shape[0])), global_video_features.shape[0])
        features = np.concatenate((padding_local_video_features, global_video_features.reshape((1, global_video_features.shape[0]))))
        print(features)
        #print(local_features)



