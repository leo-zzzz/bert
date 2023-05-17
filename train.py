

# 接下来，我们需要加载BERT模型和tokenizer：


from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=855)



# 在这个例子中，我们假设我们有一个包含855个标签的数据集。接下来，我们需要准备数据集。我们可以使用Pandas库来读取CSV文件，并将标签转换为数字：


import pandas as pd

df = pd.read_csv('data.csv')

labels = df['label'].unique()
label_map = {label: i for i, label in enumerate(labels)}

df['label'] = df['label'].map(label_map)


# 接下来，我们需要将数据集转换为BERT模型可以处理的格式。我们可以使用tokenizer将文本转换为tokens，并使用pad_sequences函数将tokens填充到相同的长度：


import torch

max_len = 128

input_ids = []
attention_masks = []

for text in df['text']:
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(df['label'].values)

# 现在，我们可以将数据集分成训练集和测试集，并使用DataLoader将它们转换为可以输入到模型中的格式：

from torch.utils.data import TensorDataset, random_split, DataLoader

dataset = TensorDataset(input_ids, attention_masks, labels)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 32

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 最后，我们可以定义训练和评估函数，并开始训练模型：

from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 10

total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def train(model, train_dataloader,validation_dataloader):
    for epoch in range(epochs):
        model.train()
        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            with torch.no_grad():
                outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += (logits.argmax(1) == inputs['labels']).sum().item()
            nb_eval_examples += inputs['input_ids'].size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        print('Epoch: {}/{}'.format(epoch + 1, epochs))
        print('Training Loss: {:.4f}'.format(loss.item()))
        print('Validation Loss: {:.4f}'.format(eval_loss))
        print('Validation Accuracy: {:.4f}'.format(eval_accuracy))




# 现在，我们可以调用train函数来训练模型：


train(model, train_dataloader, validation_dataloader)
torch.save(model,'chatgpt-bert')

# 以下为测试代码
# def classify_text(text):
#     # 对文本进行tokenize和padding
#     input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
#     # 使用BERT模型进行文本分类
#     outputs = model(input_ids)[0]
#     # 获取预测结果
#     _, predicted = torch.max(outputs, 1)
#     return labels[predicted.item()]