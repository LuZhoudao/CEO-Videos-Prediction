import os
from Audio.audio_features import audio_features_extractor
from Text.text_features import text_features_extractor



def makedir(new_path):
    if not os.path.exists(new_path):
        os.makedirs(new_path)


local_features_path = '../Features'
makedir(local_features_path)
video_lst = os.listdir('../Original_videos')
for video in video_lst:
    makedir(f'{local_features_path}/{os.path.splitext(video)[0]}')


# Audio
all_audios_path = '../Transcript/audio'
audio_lst = os.listdir(all_audios_path)
for audio in audio_lst:
    audio_path = os.path.join(all_audios_path, audio)
    audio_features = audio_features_extractor(audio_path)
    audio_output_path = f'{local_features_path}/{audio}'
    audio_features.to_csv(f'{audio_output_path}/{audio}_audio_local.csv')

# Text
all_texts_path = '../Transcript/text'
text_lst = os.listdir(all_texts_path)
for text in text_lst:
    text_path = os.path.join(all_texts_path, text)
    text_features = text_features_extractor(text_path)
    text_output_path = f'{local_features_path}/{text}'
    text_features.to_csv(f'{text_output_path}/{text}_text_local.csv')



