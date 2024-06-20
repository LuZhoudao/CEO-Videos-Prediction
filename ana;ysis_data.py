import os

features_path = 'Features'

features_lst = os.listdir(features_path)
print("Total number of data", len(features_lst))

audio_local_count = 0
video_local_count = 0
text_local_count = 0
audio_global_count = 0
video_global_count = 0
text_global_count = 0
total_small_files_count = 0
check_lst = []
AUDIO_CHECK_LST = []
for feature in features_lst:
    file_path = os.path.join(features_path, feature)
    files_lst = os.listdir(file_path)

    if f'{feature}_audio_local.csv' in files_lst:
        audio_local_count += 1
    if f'{feature}_video_local.csv' in files_lst:
        video_local_count += 1
    if f'{feature}_text_local.csv' in files_lst:
        text_local_count += 1

    global_path = os.path.join(file_path, "Global_features")

    global_file_lst = os.listdir(global_path)
    total_small_files_count += len(global_file_lst)
    for small_file in global_file_lst:
        small_file_path = os.path.join(global_path, small_file)
      
        small_file_lst = os.listdir(small_file_path)
        if f'{small_file}_audio_global.npy' in small_file_lst:
            audio_global_count += 1
        else:
            AUDIO_CHECK_LST.append(small_file)
        if f'{small_file}_video_global.npy' in small_file_lst:
            video_global_count += 1
        else:
            check_lst.append(small_file)
        if f'{small_file}_text_global.npy' in small_file_lst:
            text_global_count += 1
       
print("total number of sentence", total_small_files_count)
print("Total number of local text:", text_local_count)
print("Total number of local audio:", audio_local_count)
print("Total number of local video:", video_local_count)
print("Total number of global text:", text_global_count)
print("Total number of global audio:", audio_global_count)
print("Total number of global video:", video_global_count)
#print(check_lst)
#print(AUDIO_CHECK_LST)