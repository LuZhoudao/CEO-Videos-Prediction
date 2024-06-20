# # # time_str = "00:00:000.45"
# # #
# # # # 分割时间字符串
# # # time_parts = time_str.split(":")
# # #
# # # # 提取时、分、秒部分
# # # hours = int(time_parts[0])
# # # minutes = int(time_parts[1])
# # # seconds = int(time_parts[2].split(".")[0])
# # #
# # # # 构建只有时分秒的时间字符串
# # # time_only_hms = f"{hours:02}:{minutes:02}:{seconds:02}"
# # #
# # # # 打印结果
# # # print(time_only_hms)

# # from transformers import BertTokenizer, BertModel
# # import string
# # import os
# # import numpy as np


# # def makedir(new_path):
# #     if not os.path.exists(new_path):
# #         os.makedirs(new_path)


# # def split_string_into_sentences(input_string, punctuation_list):
# #     for char in punctuation_list:
# #         input_string = input_string.replace(char, char + ' ')
# #     sentence_list = input_string.split('. ')
# #     sentence_list = [sentence.strip() for sentence in sentence_list]
# #     return sentence_list


# # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# # model = BertModel.from_pretrained('bert-base-uncased')
# # text = """"Matthew Rocco January 11 2019Jump to comments sectionPrint this pageGet ahead with daily markets updates.Join the FT's WhatsApp channelMoodys Investors Service on Thursday became the second rating agency to push PG&Es credit rating to junk status amid concerns over wildfire-related costs, sending shares of the California power utility sharply lower.
# # Moodys lowered its rating for PG&E to B2 from the investment grade level of Baa3, saying potential liabilities have grown and liquidity reserves have declined. PG&E subsidiary Pacific Gas & Electric also received a rating cut to Ba3 from Baa2.
# # The company is increasingly reliant on extraordinary intervention by legislators and regulators, which may not occur soon enough or be of sufficient magnitude to address these adverse developments, Moodys vice-president and senior credit officer Jeff Cassella said.
# # The rating agency said PG&E debt remains under review, which could result in another multi-notch downgrade.
# # Earlier this week, S&P Global Ratings also reduced its credit rating for PG&E to B from BBB- and threatened to cut it again by one notch or more.
# # PG&E faces billions of dollars in potential liabilities and lawsuits filed by victims of the Camp Fire. California officials said 86 people died and thousands of homes were destroyed as a result of last years blaze, the most destructive in the states history. State authorities are still investigating the cause of the fire.
# # The company has warned that it could be subject to significant liability exceeding insurance coverage. Reuters reported last week that PG&E is exploring bankruptcy for some or all of its businesses.
# # Shares in PG&E were down 6.4 per cent at $16.62 in after-hours trading.
# # Event details and informationEnergy Transition SummitLondon, UK & Online22 October - 24 October 2024Delivering a stable and efficient clean energy systemRegister nowPresented byFT LiveExplore all events"

# # """


# # punctuation_list = [char for char in string.punctuation if char not in ['$', '&', '-']]
# # text_list = split_string_into_sentences(text, punctuation_list)
# # date_path = "E:/year3_sem2/URIS/data"
# # date_list = os.listdir(date_path)
# # # for date in date_list:
# # #     data_path = os.path.join(date_path, date)
# # #     data_list = os.listdir(data_path)
# # #     output_path = os.path.join("E:/year3_sem2/URIS/Text", date)
# # #     for data in data_list:
# # #         one_data_path = os.path.join(data_path, data)
# # #         text_embedder()



# # tokens = tokenizer.encode_plus(text, add_special_tokens=False, return_tensors='pt')
# # # 獲取BERT模型的輸
# #     new = model(**tokens)
# #     new_embedding = new.last_hidden_state[:, 0, :].detach().numpy()


# import os
# path = 'Features'
# local_lst = os.listdir(path)
# three_lst = []
# for lst in local_lst:
#     lst_path = os.path.join(path, lst)
#     features_lst = os.listdir(lst_path)
#     if(len(features_lst)) == 3:
#         three_lst.append(features_lst)
# print(len(three_lst))

# path = 'Transcript/video/small_video'
# local_lst = os.listdir(path)
# three_lst = []
# for lst in local_lst:
#     lst_path = os.path.join(path, lst)
#     features_lst = os.listdir(lst_path)
#     if(len(features_lst)) == 0:
#         three_lst.append(lst)
# print(len(three_lst))
# print(three_lst)

# input_path = "/home/boris/DataDisk/CNBC_Original_Videos/Videos"
# video_files = os.listdir(input_path)
# ts_lst = []
# for video in video_files:
#     if os.path.splitext(video)[-1] == '.ts':
#         ts_lst.append(os.path.splitext(video)[0])
# print(ts_lst)

import numpy as np
import pickle

with open('Models/Multimodal-Transformer/data/mosei_senti_data_noalign.pkl', 'rb') as file:
    dk = pickle.load(file)
print(dk)
print(dk.shape)