import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import torch
import math
from torch.nn.utils.rnn import pad_sequence
import csv

class data_creater:

    def __init__(self):
        # DATES_PATH = os.path.join(str(config.dataset_dir), "Text")
        DATES_PATH = '../../../Features'
        Y_PATH = "../data/Original_Videos_List_adjusted_with_return.csv"
        er = pd.read_csv(Y_PATH, index_col='videoid')
        

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        date_lst = os.listdir(DATES_PATH)
        train_length = 2702
        test_length = validation_length = 338

        o_train_id_lst = date_lst[:train_length]
        o_test_id_lst = date_lst[train_length:train_length + test_length]
        o_validation_id_lst = date_lst[train_length + test_length:]

        self.train = train = {}
        self.dev = dev = {}
        self.test = test = {}

        train_id_lst = []
        train_video_lst = []
        train_audio_lst = []
        train_text_lst = []
        train_label_lst = []
        train_video_mask_lst = []
        train_audio_mask_lst = []
        train_text_mask_lst = []

        test_id_lst = []
        test_video_lst = []
        test_audio_lst = []
        test_text_lst = []
        test_label_lst = []
        test_video_mask_lst = []
        test_audio_mask_lst = []
        test_text_mask_lst = []


        valid_id_lst = []
        valid_video_lst = []
        valid_audio_lst = []
        valid_text_lst = []
        valid_label_lst = []
        valid_video_mask_lst = []
        valid_audio_mask_lst = []
        valid_text_mask_lst = []

        for id in date_lst:
            try:
            #if 1:    
                id_path = os.path.join(DATES_PATH, id)
                files_lst = os.listdir(os.path.join(id_path, "Global_features"))
                files_lst = self.bubble_sort(files_lst)

                with open(f'{id_path}/{id}_video_local.csv', 'r') as file:
                    reader = csv.reader(file)
                    # 提取第一列数据并转换为列表
                    try:
                        if reader !="":
                            video_name_lst = [row[0] for row in reader if row[1]]
                    except:
                        video_name_lst = []
                # with open(f'{id_path}/{id}_audio_local.csv', 'r') as file:
                #     reader = csv.reader(file)
                #     # 提取第一列数据并转换为列表
                #     audio_name_lst = [row[0] for row in reader if row[1]]

                # with open(f'{id_path}/{id}_text_local.csv', 'r') as file:
                #     reader = csv.reader(file)
                #     # 提取第一列数据并转换为列表
                #     text_name_lst = [row[0] for row in reader if row[1]]

                audio_lst = []
                video_lst = []
                text_lst = []
                video_mask_lst = []
                audio_mask_lst = []
                text_mask_lst = []

                for file in files_lst:
                    
                    channel_path = os.path.join(os.path.join(id_path, "Global_features"), file)
                    channel_files_lst = os.listdir(channel_path)

                    not_miss_lst = []
                    for channel in channel_files_lst:
                        judge = channel.split('_')[-2] 
                        not_miss_lst.append(judge)
                    
                        if judge == 'audio':
                            audio_embedding = np.load(os.path.join(channel_path, channel))[np.newaxis, :]
                            shape1, shape2 = audio_embedding.shape[1], audio_embedding.shape[-1]
                            audio_lst.append(audio_embedding.reshape(shape1, shape2))
                            audio_mask_lst.append(np.ones((1), dtype=np.float32))
                        #     #print(np.load(os.path.join(channel_path, channel))[np.newaxis, :].shape)
                        elif judge == 'video':
                            video_lst.append(np.load(os.path.join(channel_path, channel))[np.newaxis, :])
                            if file in video_name_lst:
                                video_mask_lst.append(np.ones((1), dtype=np.float32))
                            else:
                                video_mask_lst.append(np.zeros((1), dtype=np.float32))
                        elif judge == 'text':
                            print(np.load(os.path.join(channel_path, channel))[np.newaxis, :].shape)
                            text_lst.append(np.load(os.path.join(channel_path, channel))[np.newaxis, :])
                            text_mask_lst.append(np.ones((1), dtype=np.float32))

                    if 'video' not in not_miss_lst:
                        
                        #video_mask_lst.append(np.zeros((1, 768), dtype=np.float32))
                        video_lst.append(np.zeros((1, 768), dtype=np.float32))
                        if file in video_name_lst:
                            video_mask_lst.append(np.ones((1), dtype=np.float32))
                        else:
                            video_mask_lst.append(np.zeros((1), dtype=np.float32))
                    if 'text' not in not_miss_lst:
                        text_mask_lst.append(np.zeros((1), dtype=np.float32))
                        text_lst.append(np.zeros((1, 768), dtype=np.float32))
                    if 'audio' not in not_miss_lst:
                        audio_mask_lst.append(np.zeros((1), dtype=np.float32))             
                        audio_lst.append(np.zeros((1, 128), dtype=np.float32))             
                
                #print(len(video_lst), len(audio_lst), len(text_lst))
                label = round(er.loc[int(id), 'car01']*100, 4)

                if math.isnan(label):
                    continue
                
                video_array = np.array(video_lst, dtype=np.float32).reshape(np.array(video_lst).shape[0], np.array(video_lst).shape[-1])
                audio_array = self.connect2(audio_lst) #S.reshape(np.array(audio_lst).shape[0], np.array(audio_lst).shape[-1])
                text_array = np.array(text_lst, dtype=np.float32).reshape(np.array(text_lst).shape[0], np.array(text_lst).shape[-1])
                label = np.array([label])
                video_mask_array = np.array(video_mask_lst, dtype=np.float32)
                audio_mask_array = np.array(audio_mask_lst, dtype=np.float32)
                text_mask_array = np.array(text_mask_lst, dtype=np.float32)


                if id in o_train_id_lst:
                    #id = np.array([id])
                    #print(id)
                    train_id_lst.append(np.array([id]))
                    train_video_lst.append(video_array)
                    train_audio_lst.append(audio_array)
                    train_text_lst.append(text_array)
                    train_label_lst.append(label)
                    train_video_mask_lst.append(video_mask_array)
                    train_audio_mask_lst.append(audio_mask_array)
                    train_text_mask_lst.append(text_mask_array)
                    #self.add(self.train, id, country1, country2, pairs, label)
                elif id in o_test_id_lst:
                    #id = np.array([int(id)])
                    test_id_lst.append(np.array([id]))
                    test_video_lst.append(video_array)
                    test_audio_lst.append(audio_array)
                    test_text_lst.append(text_array)
                    test_label_lst.append(label)
                    test_video_mask_lst.append(video_mask_array)
                    test_audio_mask_lst.append(audio_mask_array)
                    test_text_mask_lst.append(text_mask_array)
                    #self.add(self.test, id, country1, country2, pairs, label)
                elif id in o_validation_id_lst:
                    #id = np.array([int(id)])
                    valid_id_lst.append(np.array([id]))
                    valid_video_lst.append(video_array)
                    valid_audio_lst.append(audio_array)
                    valid_text_lst.append(text_array)
                    valid_label_lst.append(label)
                    valid_video_mask_lst.append(video_mask_array)
                    valid_audio_mask_lst.append(audio_mask_array)
                    valid_text_mask_lst.append(text_mask_array)
                #self.add(self.dev, id, country1, country2, pairs, label)
            except:
                pass
                


        self.train["id"] = np.vstack(train_id_lst)
        self.train["video"] = self.connect(train_video_lst)
        self.train["audio"] = self.connect3(train_audio_lst)
        #print(self.train['audio'].shape)
        self.train["text"] = self.connect(train_text_lst)
        self.train["label"] = np.vstack(train_label_lst)
        self.train["video_mask"] = self.connect(train_video_mask_lst)
        self.train["video_mask"] = self.train["video_mask"].reshape(self.train["video_mask"].shape[0], self.train["video_mask"].shape[1])
        self.train["audio_mask"] = self.connect(train_audio_mask_lst)
        self.train["audio_mask"] = self.train["audio_mask"].reshape(self.train["audio_mask"].shape[0], self.train["audio_mask"].shape[1])
        self.train["text_mask"] = self.connect(train_text_mask_lst)
        self.train["text_mask"] = self.train["text_mask"].reshape(self.train["text_mask"].shape[0], self.train["text_mask"].shape[1])
        
        self.test["id"] = np.vstack(test_id_lst)
        self.test["video"] = self.connect(test_video_lst)
        self.test["audio"] = self.connect3(test_audio_lst)
        #print(self.test['audio'].shape)
        self.test["text"] = self.connect(test_text_lst)
        self.test["label"] = np.vstack(test_label_lst)
        self.test["video_mask"] = self.connect(test_video_mask_lst)
        self.test["video_mask"] = self.test["video_mask"].reshape(self.test["video_mask"].shape[0], self.test["video_mask"].shape[1])
        self.test["audio_mask"] = self.connect(test_audio_mask_lst)
        self.test["audio_mask"] = self.test["audio_mask"].reshape(self.test["audio_mask"].shape[0], self.test["audio_mask"].shape[1])
        self.test["text_mask"] = self.connect(test_text_mask_lst)
        self.test["text_mask"] = self.test["text_mask"].reshape(self.test["text_mask"].shape[0], self.test["text_mask"].shape[1])

        self.dev["id"] = np.vstack(valid_id_lst)
        self.dev["video"] = self.connect(valid_video_lst)
        self.dev["audio"] = self.connect3(valid_audio_lst)
        self.dev["text"] = self.connect(valid_text_lst)
        self.dev["label"] = np.vstack(valid_label_lst)
        self.dev["video_mask"] = self.connect(valid_video_mask_lst)
        self.test["video_mask"] = self.dev["video_mask"].reshape(self.dev["video_mask"].shape[0], self.dev["video_mask"].shape[1])
        self.dev["audio_mask"] = self.connect(valid_audio_mask_lst)
        self.dev["audio_mask"] = self.dev["audio_mask"].reshape(self.dev["audio_mask"].shape[0], self.dev["audio_mask"].shape[1])
        self.dev["text_mask"] = self.connect(valid_text_mask_lst)
        self.test["text_mask"] = self.dev["text_mask"].reshape(self.dev["text_mask"].shape[0], self.dev["text_mask"].shape[1])


        self.final_data = {}
        self.final_data["train"] = self.train
        self.final_data["test"] = self.test
        self.final_data["valid"] = self.dev
        with open(f"../data/car01_mask.pkl", "wb") as file:
            pickle.dump(self.final_data, file, protocol = 4)
        # with open(f"{path}/test.pkl", "wb") as file:
        #     pickle.dump(self.test, file)
        # with open(f"{path}/valid.pkl", "wb") as file:
        #     pickle.dump(self.dev, file)
       
            

    def add(self, lst, id, country1, country2, pairs, label):

        if "id" not in lst.keys():
            lst["id"] = np.array([id])
            lst["country1"] = country1[np.newaxis, ...]
            lst["country2"] = country2[np.newaxis, ...]
            lst["pairs"] = pairs[np.newaxis, ...]
            lst["label"] = np.array([label])

        else:
            lst["id"] = np.vstack((lst["id"], np.array([id])))
            lst["country1"] = np.concatenate((lst["country1"], self.check(lst["country1"], country1)[np.newaxis, ...]), axis=0)
            lst["country2"] = np.concatenate((lst["country2"], self.check(lst["country2"], country1)[np.newaxis, ...]), axis=0)
            lst["pairs"] = np.concatenate((lst["pairs"], self.check(lst["pairs"], country1)[np.newaxis, ...]), axis=0)
            lst["label"] = np.vstack((lst["label"], np.array([label])))

    def connect(self, array_lst):
        max_rows = max(arr.shape[0] for arr in array_lst)
        max_cols = max(arr.shape[-1] for arr in array_lst)
        num_arrays = len(array_lst)

        # 创建空的四维数组
        result = np.zeros((num_arrays, max_rows, max_cols), dtype=float)

        # 将三维数组复制到四维数组
        for i, arr in enumerate(array_lst):
            result[i, :arr.shape[0], :arr.shape[-1]] = arr

        return result
    
    def connect2(self, array_lst):
        for arr in array_lst:
            print(arr.shape)
        max_rows = max(arr.shape[0] for arr in array_lst)
        max_cols = max(arr.shape[-1] for arr in array_lst)
        #max_cols = 128
        num_arrays = len(array_lst)

        # 创建空的四维数组
        result = np.zeros((num_arrays, max_rows, max_cols), dtype=float)

        # 将三维数组复制到四维数组
        for i, arr in enumerate(array_lst):
            result[i, :arr.shape[0], :arr.shape[1]] = arr

        return result
        
    def connect3(self, array_lst):
        max_rows = max(arr.shape[-2] for arr in array_lst)
        #max_cols = max(arr.shape[1] for arr in array_lst)
        max_cols = 128
        max1 = max(arr.shape[0] for arr in array_lst)
        num_arrays = len(array_lst)

        # 创建空的四维数组
        result = np.zeros((num_arrays ,max1,  max_rows, max_cols), dtype=float)

        # 将三维数组复制到四维数组
        for i, arr in enumerate(array_lst):
            result[i, :arr.shape[0], :arr.shape[1], :arr.shape[-1], ] = arr
 
        return result
    
    def connect4(self, array_lst):
        max_rows = max(arr.shape[0] for arr in array_lst)
        #max_cols = max(arr.shape[1] for arr in array_lst)
        #max_cols = 128
        num_arrays = len(array_lst)

        # 创建空的四维数组
        result = np.zeros((num_arrays, max_rows), dtype=float)

        # 将三维数组复制到四维数组
        for i, arr in enumerate(array_lst):
            result[i, :arr.shape[0],] = arr

        return result
        

    def bubble_sort(self, arr):
        """
        实现冒泡排序算法
        :param arr: 需要排序的数组
        :return: 排序后的数组
        """
        n = len(arr)
        # 遍历数组的所有元素
        for i in range(n):
        # 优化:如果在某一轮排序中没有发生交换,说明数组已经有序,可以提前退出
            swapped = False

        # 将数组中相邻的元素进行比较和交换
            for j in range(0, n - i - 1):
                index_j = int(arr[j].split('_')[-2])
                index_j1 = int(arr[j + 1].split('_')[-2])
                if index_j > index_j1:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True

            if not swapped:
                break

        return arr
        
data_creater()