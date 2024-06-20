import liwc
import re
import os
import glob
import pandas as pd
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
import string
nltk.download('punkt')
nltk.download('stopwords')



def calculate_similarity(text1, text2): 
    text_1_lst = text1.split()
    text_2_lst = text2.split()

    count = 0
    for i in text_1_lst:
        if i in text_2_lst:
            count += 1
    if count > 1:
        return True
    else:
        return False




def count_words(text):
    # 以空格为分隔符拆分成单词
    words = text.split()
    # 返回单词数量
    return len(words)


def tokenize(text):
    # you may want to use a smarter tokenizer
    # tokens = word_tokenize(text)
    # token_lst = [token.strip() for token in tokens]
    # return " ".join(token_lst)
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)
    

liwcPath = 'LIWC2007_English100131.dic'
parse, category_names = liwc.load_token_parser(liwcPath)


def text_handler(text):
    text_df = pd.DataFrame()
    

    tokens = tokenize(text)

    length = count_words(text)
    counts = dict(Counter(category for token in tokens for category in parse(token)))
    
    counts = {key: round(value/length, 3) for key, value in counts.items() if key in ['i', 'we', 'they']}
    print(counts)
    for index in ['i', 'we', 'they']:
        if index not in counts:
            counts[index] = 0

    if counts['i']+counts["we"] != 0:
        counts["narcissism"] = counts['i'] / (counts['i']+counts["we"])
    else:
        counts["narcissism"] = None
    return counts

conference_path = "/home/boris/DataDisk/conference_call/clean_transcript_qa"
conference_lst = os.listdir(conference_path)

people_df = pd.read_csv('exeid.csv',encoding ='latin-1')

all_df = pd.DataFrame()
name_dict = {}
for conference in conference_lst:
    try:
        para_dict = {}
        single_conference_path = os.path.join(conference_path, conference)
        conference_df = pd.read_csv(single_conference_path)

        text = "" 
        CEO_df = conference_df[conference_df["position"].str.contains('ceo', case=False)]
        
        people = CEO_df['people'].iloc[0]
        # flag = False
        # for index, row in people_df.iterrows():
        #     if calculate_similarity(people, row['ceoname']):
        #         para_dict['gvkey'] = row['gvkey']
        #         para_dict['directorid'] = row['directorid']
        #         flag = True
        # if flag:
        for index, row in CEO_df.iterrows():
            # name_dict[row['people']] = row['position']
            if row['people'] == people:
                text += row["content"]

        
        para_dict.update(text_handler(text))
        para_dict['people'] = CEO_df['people'].iloc[0]
        para_dict["position"] = CEO_df["position"].iloc[0]


        new_df = pd.DataFrame(para_dict, index=[os.path.splitext(conference)[0]])
        all_df = pd.concat([all_df, new_df], axis=0)
        #print(new_df)
    except:
        pass


# name_df = pd.DataFrame({"name":list(name_dict.keys()), "position": list(name_dict.values())})
# name_df.to_csv("name.csv")
print(all_df)
all_df.to_csv("conference_score.csv")
    