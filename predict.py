import pandas as pd
import torch
from transformers import BertTokenizer
max_len = 128
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
test = pd.read_csv('test3.csv')
test['all'] = test['摘要'] + test['标题']
model = torch.load('chatgpt-bert',map_location='cpu')
df = pd.read_csv('data.csv')
labels = df['label'].unique()
def classify_text(text):
    # 对文本进行tokenize和padding
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    # if input_ids.shape[1]>512:
    #     input_ids[0] = input_ids[0][:512]
    # 使用BERT模型进行文本分类
    input_ids = encoded_dict['input_ids']
    outputs = model(input_ids)[0]
    # 获取预测结果
    _, predicted = torch.max(outputs, 1)
    return labels[predicted.item()]

test['predict_lable'] = 0

for i in range(len(test)):

    test['predict_lable'][i] = classify_text(test['all'][i])


test[['标题','摘要','类型','predict_lable']].to_excel('pre_test3.xlsx')


