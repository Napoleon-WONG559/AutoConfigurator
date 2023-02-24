import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from sklearn.utils.class_weight import compute_class_weight

# specify GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
print(device)

df = pd.read_csv("/kaggle/input/reviewgraphiclabel/review_graphic_label_map.csv")

train_text, val_text, train_labels, val_labels = train_test_split(df['review'], df['graphicard_label'], 
                                                                    random_state=2018, 
                                                                    test_size=0.9, 
                                                                    stratify=df['graphicard_label'])

# import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased')

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

max_seq_len = 512

# tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# for train set
train_seq = torch.tensor(tokens_train['input_ids'])
#train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

# for validation set
val_seq = torch.tensor(tokens_val['input_ids'])
#val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

batch_size = 8

# wrap tensors
train_data = TensorDataset(train_seq, train_y)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_y)

# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

# freeze all the parameters
for param in bert.parameters():
    param.requires_grad = True

class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert 
        # dropout layer
        #hjh dropout test
        self.dropout = nn.Dropout(0.1)
        #self.dropout = nn.Dropout(0.2)
        #self.dropout = nn.Dropout(0.05)
        # relu activation function
        self.relu =  nn.ReLU()
        # dense layer 1
        self.fc1 = nn.Linear(768,512)
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512,6) # (6 LABELS)
        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id):
        #pass the inputs to the model  
        outputs = self.bert(sent_id)
#         print(cls_hs)
#         x = self.fc1(outputs.last_hidden_state)
        x = self.fc1(outputs.pooler_output)
        #x = self.fc1(outputs)
        #x dim 512
        x = self.relu(x)
        x = self.dropout(x)
        # output layer
        x = self.fc2(x)
        # apply softmax activation
        x = self.softmax(x)
        return x

# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)

# push the model to GPU
model = model.to(device)

# define the optimizer
optimizer = AdamW(model.parameters(), lr = 1e-5)

#compute the class weights
class_wts = compute_class_weight(class_weight='balanced', classes =np.unique(train_labels), y=train_labels)
# convert class weights to tensor
weights= torch.tensor(class_wts,dtype=torch.float)
weights = weights.to(device)

# loss function
cross_entropy  = nn.NLLLoss(weight=weights) 

# train
epoch=10
model.train()
for i in range(epoch):
    count=0
    loss_rec=0
    for batch in train_dataloader:
        #print(batch)
        #batch=torch.stack(batch,dim=1)
        batch = [r.to(device) for r in batch]
        inputs, labels=batch
        #batch.to(device)
        #inputs, labels=batch
        #inputs.to(device)
        #labels.to(device)
        logits=model(inputs)
        loss=cross_entropy(logits,labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        count+=1
        loss_rec+=loss
    print('NO.',i,' epoch avg loss: ',loss_rec/count)

model.eval()
preds=[]
labels=[]
for batch in val_dataloader:
    #batch=torch.stack(batch,dim=1)
    #batch.to(device)
    batch = [r.to(device) for r in batch]
    inputs, label=batch
    #inputs.to(device)
    #labels.to(device)
    logits=model(inputs)
    labels.extend(label.cpu().tolist())
    preds.extend(torch.argmax(logits,dim=-1).cpu().tolist())
acc=sum([int(i==j) for i,j in zip(preds, labels)])/len(preds)

print("accuracy is : ",acc)