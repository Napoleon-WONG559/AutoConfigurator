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
import gc

# specify GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
print(device)

df = pd.read_csv("/kaggle/input/newreviewallmap/review_all_map.csv")

train_text, val_text, \
train_cpu_labels, val_cpu_labels,\
train_gra_labels,val_gra_labels,\
train_hard_labels, val_hard_labels,\
train_ram_labels, val_ram_labels,\
train_scre_labels, val_scre_labels = train_test_split(df['review'], df['cpu'], df['graphicard'],df['hardisk'],df['ram'], df['screen'],
                                                                    #random_state=2018, 
                                                      #shuffle=False,
                                                      test_size=0.3)
train_cpu_labels_ori=train_cpu_labels
train_gra_labels_ori=train_gra_labels
train_hard_labels_ori=train_hard_labels
train_ram_labels_ori=train_ram_labels
train_scre_labels_ori=train_scre_labels

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
#train_y = torch.tensor(train_labels.tolist())
train_cpu_labels=torch.tensor(train_cpu_labels.tolist())
train_gra_labels=torch.tensor(train_gra_labels.tolist())
train_hard_labels=torch.tensor(train_hard_labels.tolist())
train_ram_labels=torch.tensor(train_ram_labels.tolist())
train_scre_labels=torch.tensor(train_scre_labels.tolist())

# for validation set
val_seq = torch.tensor(tokens_val['input_ids'])
#val_mask = torch.tensor(tokens_val['attention_mask'])
#val_y = torch.tensor(val_labels.tolist())
val_cpu_labels=torch.tensor(val_cpu_labels.tolist())
val_gra_labels=torch.tensor(val_gra_labels.tolist())
val_hard_labels=torch.tensor(val_hard_labels.tolist())
val_ram_labels=torch.tensor(val_ram_labels.tolist())
val_scre_labels=torch.tensor(val_scre_labels.tolist())

batch_size = 4

# wrap tensors
train_data = TensorDataset(train_seq, train_cpu_labels, train_gra_labels, train_hard_labels, \
                           train_ram_labels, train_scre_labels)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_cpu_labels, val_gra_labels, val_hard_labels, \
                        val_ram_labels, val_scre_labels)

# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

# freeze all the parameters
for param in bert.parameters():
    param.requires_grad = True

class BERT_Arch(nn.Module):
    def __init__(self, bert, m_type):
        super(BERT_Arch, self).__init__()
        self.bert = bert 
        # dropout layer
        self.dropout = nn.Dropout(0.1)
        # relu activation function
        self.relu =  nn.ReLU()
        # dense layer 1
        self.fc1 = nn.Linear(768,512)
        # dense layer 2 (Output layer)
        if(m_type=='cpu'):
            self.fc2 = nn.Linear(512,7)
        elif(m_type=='graphic'):
            self.fc2 = nn.Linear(512,6) # (6 LABELS)
        elif(m_type=='hard'):
            self.fc2 = nn.Linear(512,6)
        elif(m_type=='ram'):
            self.fc2 = nn.Linear(512,7)
        elif(m_type=='screen'):
            self.fc2 = nn.Linear(512,8)
        else:
            raise NotImplementedError
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
#model = BERT_Arch(bert)
cpu_model = BERT_Arch(bert,'cpu')
gra_model = BERT_Arch(bert,'graphic')
hard_model = BERT_Arch(bert,'hard')
ram_model = BERT_Arch(bert,'ram')
scre_model = BERT_Arch(bert,'screen')

# push the model to GPU
cpu_model = cpu_model.to(device)
gra_model = gra_model.to(device)
hard_model = hard_model.to(device)
ram_model = ram_model.to(device)
scre_model = scre_model.to(device)

# define the optimizer
#ind_para_name=["bert.pooler.dense.weight","bert.pooler.dense.bias","fc1.weight","fc1.bias",\
#               "fc2.weight","fc2.bias"]
#shared_para=[{'params': [p for n, p in model.named_parameters() 
#                         if(n not in ind_para_name)]}]
shared_optimizer = AdamW(bert.parameters(), lr = 1e-5)

cpu_para=[{'params':[p for n,p in cpu_model.named_parameters() if ('bert' not in n)]}]
cpu_optimizer=AdamW(cpu_para, lr=1e-5)

gra_para=[{'params':[p for n,p in gra_model.named_parameters() if ('bert' not in n)]}]
gra_optimizer=AdamW(gra_para, lr=1e-5)

hard_para=[{'params':[p for n,p in hard_model.named_parameters() if ('bert' not in n)]}]
hard_optimizer=AdamW(hard_para, lr=1e-5)

ram_para=[{'params':[p for n,p in ram_model.named_parameters() if ('bert' not in n)]}]
ram_optimizer=AdamW(ram_para, lr=1e-5)

scre_para=[{'params':[p for n,p in scre_model.named_parameters() if ('bert' not in n)]}]
scre_optimizer=AdamW(scre_para, lr=1e-5)

#compute the class weights
cpu_class_wts = compute_class_weight(class_weight='balanced', classes =np.unique(train_cpu_labels_ori), y=train_cpu_labels_ori)
# convert class weights to tensor
cpu_weights= torch.tensor(cpu_class_wts,dtype=torch.float)
cpu_weights = cpu_weights.to(device)
# loss function
cpu_cross_entropy  = nn.NLLLoss(weight=cpu_weights)

#compute the class weights
gra_class_wts = compute_class_weight(class_weight='balanced', classes =np.unique(train_gra_labels_ori), y=train_gra_labels_ori)
# convert class weights to tensor
gra_weights= torch.tensor(gra_class_wts,dtype=torch.float)
gra_weights = gra_weights.to(device)
# loss function
gra_cross_entropy  = nn.NLLLoss(weight=gra_weights) 

#compute the class weights
hard_class_wts = compute_class_weight(class_weight='balanced', classes =np.unique(train_hard_labels_ori), y=train_hard_labels_ori)
# convert class weights to tensor
hard_weights= torch.tensor(hard_class_wts,dtype=torch.float)
hard_weights = hard_weights.to(device)
# loss function
hard_cross_entropy  = nn.NLLLoss(weight=hard_weights) 

#compute the class weights
ram_class_wts = compute_class_weight(class_weight='balanced', classes =np.unique(train_ram_labels_ori), y=train_ram_labels_ori)
# convert class weights to tensor
ram_weights= torch.tensor(ram_class_wts,dtype=torch.float)
ram_weights = ram_weights.to(device)
# loss function
ram_cross_entropy  = nn.NLLLoss(weight=ram_weights) 

#compute the class weights
scre_class_wts = compute_class_weight(class_weight='balanced', classes =np.unique(train_scre_labels_ori), y=train_scre_labels_ori)
# convert class weights to tensor
scre_weights= torch.tensor(scre_class_wts,dtype=torch.float)
scre_weights = scre_weights.to(device)
# loss function
scre_cross_entropy  = nn.NLLLoss(weight=scre_weights) 

# train
epoch=4
#model.train()
cpu_model.train()
gra_model.train()
hard_model.train()
ram_model.train()
scre_model.train()
for i in range(epoch):
    count=0
    loss_rec=0
    for batch in train_dataloader:
        batch = [r.to(device) for r in batch]
        inputs, cpu_labels, gra_labels, hard_labels, ram_labels, scre_labels=batch
        #logits=model(inputs)
        #loss=cross_entropy(logits,labels)
        
        cpu_logits=cpu_model(inputs)
        cpu_loss=cpu_cross_entropy(cpu_logits,cpu_labels)
        
        gra_logits=gra_model(inputs)
        gra_loss=gra_cross_entropy(gra_logits,gra_labels)
        
        hard_logits=hard_model(inputs)
        hard_loss=hard_cross_entropy(hard_logits,hard_labels)
        
        ram_logits=ram_model(inputs)
        ram_loss=ram_cross_entropy(ram_logits,ram_labels)
        
        scre_logits=scre_model(inputs)
        scre_loss=scre_cross_entropy(scre_logits,scre_labels)
        
        loss=cpu_loss+gra_loss+hard_loss+ram_loss+scre_loss
        loss.backward()
        
        shared_optimizer.step()
        shared_optimizer.zero_grad()
        
        cpu_optimizer.step()
        cpu_optimizer.zero_grad()
        
        gra_optimizer.step()
        gra_optimizer.zero_grad()
        
        hard_optimizer.step()
        hard_optimizer.zero_grad()
        
        ram_optimizer.step()
        ram_optimizer.zero_grad()
        
        scre_optimizer.step()
        scre_optimizer.zero_grad()
        
        count+=1
        loss_rec+=loss
    
    if(i==3):
        PATH = "/kaggle/working/cpu_model_epoch_"+str(i)+".pt"
        torch.save({
                'epoch': i,
                'model_state_dict': cpu_model.state_dict(),
                'optimizer_state_dict': cpu_optimizer.state_dict(),
                }, PATH)

        PATH = "/kaggle/working/gra_model_epoch_"+str(i)+".pt"
        torch.save({
                'epoch': i,
                'model_state_dict': gra_model.state_dict(),
                'optimizer_state_dict': gra_optimizer.state_dict(),
                }, PATH)

        PATH = "/kaggle/working/hard_model_epoch_"+str(i)+".pt"
        torch.save({
                'epoch': i,
                'model_state_dict': hard_model.state_dict(),
                'optimizer_state_dict': hard_optimizer.state_dict(),
                }, PATH)

        PATH = "/kaggle/working/ram_model_epoch_"+str(i)+".pt"
        torch.save({
                'epoch': i,
                'model_state_dict': ram_model.state_dict(),
                'optimizer_state_dict': ram_optimizer.state_dict(),
                }, PATH)

        PATH = "/kaggle/working/scre_model_epoch_"+str(i)+".pt"
        torch.save({
                'epoch': i,
                'model_state_dict': scre_model.state_dict(),
                'optimizer_state_dict': scre_optimizer.state_dict(),
                }, PATH)
    gc.collect()
    torch.cuda.empty_cache()
    print('NO.',i,' epoch avg loss: ',loss_rec/count)

#model.eval()
cpu_model.eval()
gra_model.eval()
hard_model.eval()
ram_model.eval()
scre_model.eval()

cpu_preds=[]
cpu_labels=[]
gra_preds=[]
gra_labels=[]
hard_preds=[]
hard_labels=[]
ram_preds=[]
ram_labels=[]
scre_preds=[]
scre_labels=[]
with torch.no_grad():
    for batch in val_dataloader:
        batch = [r.to(device) for r in batch]
        #inputs, label=batch
        inputs, cpu_label, gra_label, hard_label, ram_label, scre_label=batch

        #logits=model(inputs)
        #labels.extend(label.cpu().tolist())
        #preds.extend(torch.argmax(logits,dim=-1).cpu().tolist())

        cpu_logits=cpu_model(inputs)
        cpu_labels.extend(cpu_label.cpu().tolist())
        cpu_preds.extend(torch.argmax(cpu_logits,dim=-1).cpu().tolist())

        gra_logits=gra_model(inputs)
        gra_labels.extend(gra_label.cpu().tolist())
        gra_preds.extend(torch.argmax(gra_logits,dim=-1).cpu().tolist())

        hard_logits=hard_model(inputs)
        hard_labels.extend(hard_label.cpu().tolist())
        hard_preds.extend(torch.argmax(hard_logits,dim=-1).cpu().tolist())

        ram_logits=ram_model(inputs)
        ram_labels.extend(ram_label.cpu().tolist())
        ram_preds.extend(torch.argmax(ram_logits,dim=-1).cpu().tolist())

        scre_logits=scre_model(inputs)
        scre_labels.extend(scre_label.cpu().tolist())
        scre_preds.extend(torch.argmax(scre_logits,dim=-1).cpu().tolist())
    
#acc=sum([int(i==j) for i,j in zip(preds, labels)])/len(preds)

cpu_acc=sum([int(i==j) for i,j in zip(cpu_preds, cpu_labels)])/len(cpu_preds)
gra_acc=sum([int(i==j) for i,j in zip(gra_preds, gra_labels)])/len(gra_preds)
hard_acc=sum([int(i==j) for i,j in zip(hard_preds, hard_labels)])/len(hard_preds)
ram_acc=sum([int(i==j) for i,j in zip(ram_preds, ram_labels)])/len(ram_preds)
scre_acc=sum([int(i==j) for i,j in zip(scre_preds, scre_labels)])/len(scre_preds)

print("cpu accuracy is : ",cpu_acc)
print("graphic card accuracy is : ",gra_acc)
print("hard disk accuracy is : ",hard_acc)
print("ram accuracy is : ",ram_acc)
print("scre accuracy is : ",scre_acc)