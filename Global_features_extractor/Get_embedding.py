import os
#from Audio.audio_embedding import
from Text.text_embedding import text_embedder
from Video.video_embedding import video_embedder


def makedir(new_path):
    if not os.path.exists(new_path):
        os.makedirs(new_path)


all_input_path = '../Transcript'
all_output_path = '../Features'

all_text_input_path = os.path.join(all_input_path, 'text')
all_audio_input_path = os.path.join(all_input_path, 'audio')
all_video_input_path = os.path.join(all_input_path, 'video/small_video')
video_lst = os.listdir(all_text_input_path)

for video in video_lst:

    output_path = os.path.join(os.path.join(all_output_path, video), 'Global_features')
    makedir(output_path)
    text_input_path = os.path.join(all_text_input_path, video)
    audio_input_path = os.path.join(all_audio_input_path, video)
    video_input_path = os.path.join(all_video_input_path, video)
    # final_output_path = os.path.join(output_path, video_name)
    # makedir(final_output_path)

    text_embedder(text_input_path, output_path)
    # video_embedder(video_input_path, output_path)
