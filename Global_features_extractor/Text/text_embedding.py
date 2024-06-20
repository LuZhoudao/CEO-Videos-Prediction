from transformers import BertTokenizer, BertModel
import numpy as np
import os


def makedir(new_path):
    if not os.path.exists(new_path):
        os.makedirs(new_path)


def text_embedder(input_path, output_path):
    # 載入BERT模型和分詞器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    text_lst = os.listdir(input_path)
    for text_file in text_lst:
        try:
            text_file_path = os.path.join(input_path, text_file)
            name = os.path.splitext(text_file)[0]
            final_output_path = os.path.join(output_path, name)
            makedir(final_output_path)
            with open(text_file_path, 'r') as f:
                text = f.read()
                f.close()

            # 將文字分詞並加上特殊標記
            tokens = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')

            # 獲取BERT模型的輸出
            outputs = model(**tokens)
            
            # 獲取最後一層的隱藏狀態（CLS token的向量表示）
            embedding = outputs.last_hidden_state[:, 0, :]
            # 將嵌入向量轉換為NumPy陣列
            embedding_array = embedding.detach().numpy()[0]
            print(embedding_array)
            # 將嵌入向量保存為.npy檔案
            np.save(f'{final_output_path}/{name}_text_global.npy', embedding_array)
        except:
            pass

