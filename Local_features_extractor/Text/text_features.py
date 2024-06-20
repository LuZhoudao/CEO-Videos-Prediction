import liwc
import re
import os
import glob
import pandas as pd
from collections import Counter


def count_words(text):
    # 去除标点符号和非单词字符
    text = re.sub(r'[^\w\s]', '', text)
    # 以空格为分隔符拆分成单词
    words = text.split()
    # 返回单词数量
    return len(words)


def tokenize(text):
    # you may want to use a smarter tokenizer
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)


liwcPath = './Text/LIWC2007_English100131.dic'
parse, category_names = liwc.load_token_parser(liwcPath)


def text_features_extractor(path):
    text_df = pd.DataFrame()
    for text_file in glob.glob(os.path.join(path, '*.txt')):
        with open(text_file, 'r')as f:
            text = f.read().lower()

        tokens = tokenize(text)

        length = count_words(text)
        counts = dict(Counter(category for token in tokens for category in parse(token)))
        counts = {key: round(value/length, 3) for key, value in counts.items()}
        #file_lst.append(os.path.splitext((text_file.split("/")[-1]).split('\\')[-1])[0])
        new_df = pd.DataFrame(counts, index=[os.path.splitext((text_file.split("/")[-1]).split('\\')[-1])[0]])
        text_df = pd.concat([text_df, new_df])
    text_df = text_df.fillna(0)
    return text_df

#print(text_features_extractor("E:/year3_sem2/SA/video_prediction/Transcript/text/3000144366"))