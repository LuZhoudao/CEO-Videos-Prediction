import os
import numpy as np
from towhee import pipe, ops, DataCollection
# video environment

def makedir(new_path):
    if not os.path.exists(new_path):
        os.makedirs(new_path)


def video_embedder(input_path, output_path):
    p = (
        pipe.input('path')
            .map('path', 'frames', ops.video_decode.ffmpeg())
            .map('frames', ('labels', 'scores', 'features'),
                 ops.action_classification.video_swin_transformer(model_name='swin_t_k400_1k'))
            .output('path', 'labels', 'scores', 'features')
    )

    video_lst = os.listdir(input_path)
    for video_file in video_lst:
        name = os.path.splitext(video_file)[0]
        final_output_path = os.path.join(output_path, name)
        makedir(final_output_path)
        if f"{name}_video_global.npy" not in os.listdir(final_output_path):
     
        #try:
            print(video_file)
            video_file_path = os.path.join(input_path, video_file)
            data_collection = DataCollection(p(video_file_path))
            features = data_collection[0]['features']
            print(features)
            np.save(f'{final_output_path}/{name}_video_global.npy', features)
        # except:
        #     pass