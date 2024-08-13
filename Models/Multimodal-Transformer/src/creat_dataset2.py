import os
import pandas as pd
import numpy as np
import pandas as pd
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
        # train_video_mask_lst = []
        # train_audio_mask_lst = []
        # train_text_mask_lst = []

        test_id_lst = []
        test_video_lst = []
        test_audio_lst = []
        test_text_lst = []
        test_label_lst = []
        # test_video_mask_lst = []
        # test_audio_mask_lst = []
        # test_text_mask_lst = []


        valid_id_lst = []
        valid_video_lst = []
        valid_audio_lst = []
        valid_text_lst = []
        valid_label_lst = []
        # valid_video_mask_lst = []
        # valid_audio_mask_lst = []
        # valid_text_mask_lst = []
        text_lst = ['funct', 'pronoun', 'ppron', 'i', 'we', 'you', 'shehe', 'they', 'ipron', 'article', 'verb', 'auxverb', 'past', 'present', 'future', 'adverb', 'preps', 'conj', 'negate', 'quant', 'number', 'swear', 'social', 'family', 'friend', 'humans', 'affect', 'posemo', 'negemo', 'anx', 'anger', 'sad', 'cogmech', 'insight', 'cause', 'discrep', 'tentat', 'certain', 'inhib', 'incl', 'excl', 'percept', 'see', 'hear', 'feel', 'bio', 'body', 'health', 'sexual', 'ingest', 'relativ', 'motion', 'space', 'time', 'work', 'achieve', 'leisure', 'home', 'money', 'relig', 'death', 'assent', 'nonfl', 'filler']


        for id in date_lst:
            try:
            #if 1:    
                id_path = os.path.join(DATES_PATH, id)
                local_video_path = f'{id_path}/{id}_video_local.csv'
                local_audio_path = f'{id_path}/{id}_audio_local.csv'
                local_text_path = f'{id_path}/{id}_text_local.csv'
                
                        

                local_video_df = pd.read_csv(local_video_path, index_col=0)
                local_audio_df = pd.read_csv(local_audio_path, index_col=0)
                local_text_df = pd.read_csv(local_text_path, index_col=0)
                                
                local_audio_df = local_audio_df.replace(' --undefined--',0)
                local_audio_df = local_audio_df.replace('--undefined--',0)
                
                local_audio_df["degree_of_breaks"] = local_audio_df["degree_of_breaks"].apply(lambda x: float(x.strip()[0:-1])/100 if isinstance(x, str) and len(x.strip()) > 1 and x.strip()[0] != '-' else 0.0)
                local_audio_df["fraction_of_breaks"] = local_audio_df["fraction_of_breaks"].apply(lambda x: float(x.strip()[0:-1])/100 if isinstance(x, str) and len(x.strip()) > 1 and x.strip()[0] != '-' else 0.0)
                # local_audio_df = local_audio_df.replace('nan',0)
                # local_audio_df = local_audio_df.fillna(0)
                #local_audio_df = local_audio_df.replace('--undefined--',0)
                str_columns = local_audio_df.select_dtypes(include='object').columns
                print(id)
                print(str_columns)
                #print(local_audio_df["degree_of_breaks"], local_audio_df["fraction_of_breaks"])
                # for col in local_audio_df.columns:
                #     local_audio_df[col] = local_audio_df[col].apply(lambda x: 0.0 if isinstance(x, str) else x)

                # audio_lst = []
                # video_lst = []
                # text_lst = []
               
                index_lst = list(local_text_df.index)
                index_lst = self.bubble_sort(index_lst)
                
                
                # video_mask_lst = []
                # audio_mask_lst = []
                # text_mask_lst = []
                new_text_lst = []
                new_video_lst = []
                new_audio_lst = []
            
                for index in index_lst:
                #     if file in video_name_lst:
                    #     video_mask_lst.append(np.ones((768), dtype=np.float32))
                    # else:
                    #     video_mask_lst.append(np.zeros((768), dtype=np.float32))
                    
                    # audio_mask_lst.append(np.zeros((128), dtype=np.float32))
                    # text_mask_lst.append(np.zeros((768), dtype=np.float32))
                    
                    
                    if index in local_video_df.index:
                        video_row = local_video_df.loc[index]
                        new_video_lst.append(video_row.to_numpy(na_value=0))
                    else:
                        new_video_lst.append(np.zeros(shape=(1, 25)))
                    
                    if index in local_audio_df.index:
                        audio_row = local_audio_df.loc[index]
                        new_audio_lst.append(audio_row.to_numpy())
                    else:
                        new_audio_lst.append(np.zeros(shape=(1, 28)))
            
                    if index in local_text_df.index:
                        text_output_list = []
                        for name in text_lst:
                            try:
                                judge_answer = local_text_df.loc[index, name]
                                text_output_list.append(judge_answer)
                            except:
                                text_output_list.append(0.0)


                        # text_row = local_text_df.loc[index]
                        # if len(text_row) == 65:
                        #     text_row = text_row[:-1]
                        #print(text_output_list)
                        new_text_lst.append(np.array(text_output_list))
                    else:
                        new_text_lst.append(np.zeros(shape=(1, 64)))
                    
            
                    
                #print(len(video_lst), len(audio_lst), len(text_lst))
                label = round(er.loc[int(id), 'car01']*100, 4)

                if math.isnan(label):
                    continue
                
                video_array = np.vstack(new_video_lst)
                audio_array = np.vstack(new_audio_lst)
                text_array = np.vstack(new_text_lst)
                #print(audio_array)

                # if torch.cuda.is_available(): 
                #     video_array = torch.from_numpy(video_array).cuda() 
                #     audio_array = torch.from_numpy(audio_array).cuda() 
                #     text_array = torch.from_numpy(text_array).cuda() # audio_mask_array = torch.from_numpy(audio_mask_array).cuda() # text_mask_array = torch.from_numpy(text_mask_array).cuda() else: video_array = torch.from_numpy(video_array) audio_array = torch.from_numpy(audio_array) text_array = torch.from_numpy(text_array)
            
                # audio_mask_array = np.array(audio_mask_lst, dtype=np.float32)
                # text_mask_array = np.array(text_mask_lst, dtype=np.float32)


                if id in o_train_id_lst:
                    #id = np.array([id])
                    #print(id)
                    train_id_lst.append(np.array([id]))
                    train_video_lst.append(video_array)
                    train_audio_lst.append(audio_array)
                    train_text_lst.append(text_array)
                    train_label_lst.append(label)
                    # train_video_mask_lst.append(video_mask_array)
                    # train_audio_mask_lst.append(audio_mask_array)
                    # train_text_mask_lst.append(text_mask_array)
                    #self.add(self.train, id, country1, country2, pairs, label)
                elif id in o_test_id_lst:
                    #id = np.array([int(id)])
                    test_id_lst.append(np.array([id]))
                    test_video_lst.append(video_array)
                    test_audio_lst.append(audio_array)
                    test_text_lst.append(text_array)
                    test_label_lst.append(label)
                    # test_video_mask_lst.append(video_mask_array)
                    # test_audio_mask_lst.append(audio_mask_array)
                    # test_text_mask_lst.append(text_mask_array)
                    #self.add(self.test, id, country1, country2, pairs, label)
                elif id in o_validation_id_lst:
                    #id = np.array([int(id)])
                    valid_id_lst.append(np.array([id]))
                    valid_video_lst.append(video_array)
                    valid_audio_lst.append(audio_array)
                    valid_text_lst.append(text_array)
                    valid_label_lst.append(label)
                # valid_video_mask_lst.append(video_mask_array)
                # valid_audio_mask_lst.append(audio_mask_array)
                # valid_text_mask_lst.append(text_mask_array)
            #self.add(self.dev, id, country1, country2, pairs, label)
            except:
                pass
            

        
        self.train["id"] = np.vstack(train_id_lst)
        self.train["video"] = self.connect(train_video_lst)
        self.train["audio"] = self.connect(train_audio_lst)
        self.train["text"] = self.connect(train_text_lst)
        self.train["label"] = np.vstack(train_label_lst)
        #self.train["video_mask"] = self.connect(train_video_mask_lst)
        #self.train["audio_mask"] = self.connect(train_audio_mask_lst)
        #self.train["text_mask"] = self.connect(train_text_mask_lst)
        print(self.train["video"].shape, self.train["audio"].shape, self.train["text"].shape)

        self.test["id"] = np.vstack(test_id_lst)
        self.test["video"] = self.connect(test_video_lst)
        self.test["audio"] = self.connect(test_audio_lst)
        #print(self.test['audio'].shape)
        self.test["text"] = self.connect(test_text_lst)
        self.test["label"] = np.vstack(test_label_lst)
        #self.test["video_mask"] = self.connect(test_video_mask_lst)
        #self.test["audio_mask"] = self.connect(test_audio_mask_lst)
        #self.test["text_mask"] = self.connect(test_text_mask_lst)
        print(self.test["video"].shape, self.test["audio"].shape, self.test["text"].shape)


        self.dev["id"] = np.vstack(valid_id_lst)
        self.dev["video"] = self.connect(valid_video_lst)
        self.dev["audio"] = self.connect(valid_audio_lst)
        self.dev["text"] = self.connect(valid_text_lst)
        self.dev["label"] = np.vstack(valid_label_lst)
        #self.dev["video_mask"] = self.connect(valid_video_mask_lst)
        #self.dev["audio_mask"] = self.connect(valid_audio_mask_lst)
        #self.dev["text_mask"] = self.connect(valid_text_mask_lst)
        print(self.dev["video"].shape, self.dev["audio"].shape, self.dev["text"].shape)



        self.final_data = {}
        self.final_data["train"] = self.train
        self.final_data["test"] = self.test
        self.final_data["valid"] = self.dev
        #print(self.final_data)
        with open(f"../data/car01_local.pkl", "wb") as file:
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
        max_cols = max(arr.shape[1] for arr in array_lst)
        num_arrays = len(array_lst)

        # 创建空的四维数组
        result = np.zeros((num_arrays, max_rows, max_cols), dtype=float)

        # 将三维数组复制到四维数组
        for i, arr in enumerate(array_lst):
            result[i, :arr.shape[0], :arr.shape[1]] = arr

        return result
    
    def connect2(self, array_lst):
        max_rows = max(arr.shape[0] for arr in array_lst)
        max_cols = max(arr.shape[1] for arr in array_lst)
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
            result[i, :arr.shape[0], :arr.shape[1], :arr.shape[-1]] = arr
 
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