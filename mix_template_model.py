import torch
from openprompt.data_utils import InputExample
import csv
from openprompt.plms import load_plm
from openprompt.prompts import MixedTemplate
from transformers.utils.dummy_pt_objects import PreTrainedModel
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
from transformers.tokenization_utils import PreTrainedTokenizer
import torch.nn as nn
from transformers import AdamW
from torch.utils.data import random_split

#graphic card classes and label words
graphic_classes = [
    0,
    1,
    2,
    3,
    4,
    5
]
graphic_label_words = {
    0: ['NVIDIA GeForce GTX 1050', 'NVIDIA GeForce GTX 1050 Ti', 'GTX 1050 Ti', 'NVIDIA GeForce GTX 1060', 'NVIDIA GeForce GTX 1070', '4GB GDDR5 NVIDIA GeForce GTX 1050', 'GTX 1050'],
    1: ['AMD Radeon R4', 'radeon r5', 'AMD Radeon R5 Graphics', 'AMD Radeon R7'],
    2: ['Intel UHD Graphics 620', 'Intel Iris Plus Graphics 640', 'NVIDIA GeForce 940MX'],
    3: ['Intel HD Graphics 3000', 'Intel', 'Intel HD 620 graphics', 'Intel HD Graphics 500', 'Intel HD Graphics 520', 'Intel HD Graphics 620', 'Intel HD Graphics 400', 'Intel Celeron', 'Intel HD Graphics 505', 'AMD Radeon R2', 'Intel HD Graphics 5500', 'Intel HD Graphics', 'Intel?? HD Graphics 620 (up to 2.07 GB)', 'intel 620'],
    4: ['Integrated', 'integrated intel hd graphics', 'integrated AMD Radeon R5 Graphics', 'Integrated Graphics', 'Integrated intel hd graphics'],
    5: ['515', '4', 'FirePro W4190M', 'NONE', 'PC', 'na', 'AMD'],
}



def read_data_csv(file):
    record=[]
    with open(file,newline='') as csvfile:
        read=csv.reader(csvfile)
        for item in read:
            record.append(item[1:])
    record=record[1:]
    for ind,sample in enumerate(record):
        sample.insert(0,ind)
        sample[2]=int(sample[2])
    train_set, valid_set=random_split(record,
                 [0.7,0.3],
                 generator=torch.Generator().manual_seed(42))
    dataset={}
    train_dataset=[]
    valid_dataset=[]
    for item in train_set:
        train_dataset.append(InputExample(guid=item[0],text_a=item[1],label=item[2]))
    for item in valid_set:
        valid_dataset.append(InputExample(guid=item[0],text_a=item[1],label=item[2]))
    dataset['train']=train_dataset
    dataset['valid']=valid_dataset
    return dataset




class MixTemplateModel(nn.Module):
    def __init__(self,
                plm:PreTrainedModel,
                tokenizer: PreTrainedTokenizer,
                WrapperClass,
                dataset,
                classes,
                epoch,
                template_text,
                label_words,
                device,
                ):
        
        super().__init__()
        self.promptTemplate = MixedTemplate(
            model=plm,
            text = template_text,
            tokenizer = tokenizer,
        )

        self.promptVerbalizer = ManualVerbalizer(
            classes = classes,
            label_words = label_words,
            tokenizer = tokenizer,
            multi_token_handler="mean",
        )

        self.promptModel = PromptForClassification(
            template = self.promptTemplate,
            plm = plm,
            verbalizer = self.promptVerbalizer,
        )
        self.promptModel.to(device)

        #train_set, valid_set=random_split(dataset,
        #                                  [0.7,0.3],
        #                                  generator=torch.Generator().manual_seed(42))
        train_set=dataset['train']
        valid_set=dataset['valid']

        self.train_data_loader = PromptDataLoader(
            dataset = train_set,
            tokenizer = tokenizer,
            template = self.promptTemplate,
            tokenizer_wrapper_class=WrapperClass,
            batch_size=5,
            shuffle=True,
        )
        self.valid_data_loader = PromptDataLoader(
            dataset = valid_set,
            tokenizer = tokenizer,
            template = self.promptTemplate,
            tokenizer_wrapper_class=WrapperClass,
            batch_size=5,
        )

        self.cross_entropy  = nn.NLLLoss()
        no_decay = ['bias', 'LayerNorm.weight']
        # it's always good practice to set no decay to biase and LayerNorm parameters
        optimizer_grouped_parameters1 = [
            {'params': [p for n, p in self.promptModel.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.promptModel.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # Using different optimizer for prompt parameters and model parameters
        optimizer_grouped_parameters2 = [
            {'params': [p for n,p in self.promptModel.template.named_parameters() if "raw_embedding" not in n]}
        ]
        self.optimizer1 = AdamW(optimizer_grouped_parameters1, lr=1e-4)
        self.optimizer2 = AdamW(optimizer_grouped_parameters2, lr=1e-3)

        self.epoch=epoch

    def forward(self,batch):
        outputs=self.promptModel(batch)

        return outputs

    def train(self):
        self.promptModel.train()

    def eval(self):
        self.promptModel.eval()
    
    def set_epoch(self,epoch):
        self.epoch=epoch






if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    #load pre-trained model
    graphic_plm, graphic_tokenizer, graphic_model_config, graphic_WrapperClass = load_plm("bert", "bert-base-cased")

    #graphic model
    graphic_dataset=read_data_csv("data/review_graphic_label_map.csv")
    #graphic_train_set, graphic_valid_set=random_split(graphic_dataset,
    #                                                  [0.7,0.3],
    #                                                  generator=torch.Generator().manual_seed(42))
    graphic_epoch=5
    graphic_template='{"soft": "Someone said : "} {"placeholder":"text_a"} {"soft": "Then he need"} a computer with a {"mask"} graphic card'
    graphic_model=MixTemplateModel(graphic_plm,
                                   graphic_tokenizer,
                                   graphic_WrapperClass,
                                   graphic_dataset,
                                   graphic_classes,
                                   graphic_epoch,
                                   graphic_template,
                                   graphic_label_words,
                                   device)

    #-----------------------Train-------------------------
    graphic_model.train()
    for i in range(graphic_model.epoch):
        count=0
        loss_rec=0
        for batch in graphic_model.train_data_loader:
            batch.to(device)
            graphic_labels=batch['label']
            graphic_logits=graphic_model(batch)
            graphic_loss=graphic_model.cross_entropy(graphic_logits,graphic_labels)
            graphic_loss.backward()
            graphic_model.optimizer1.step()
            graphic_model.optimizer1.zero_grad()
            graphic_model.optimizer2.step()
            graphic_model.optimizer2.zero_grad()
            count+=1
            loss_rec+=graphic_loss
        print('NO.',i,' epoch avg loss: ',loss_rec/count)
    
    #-----------------------Validate-------------------------
    graphic_model.eval()
    preds=[]
    labels=[]
    for step, inputs in enumerate(graphic_model.valid_data_loader):
        inputs.to(device)
        graphic_logits=graphic_model(inputs)
        graphic_label=inputs['label']
        labels.extend(graphic_label.cpu().tolist())
        preds.extend(torch.argmax(graphic_logits,dim=-1).cpu().tolist())
    acc=sum([int(i==j) for i,j in zip(preds, labels)])/len(preds)

    print("accuracy is : ",acc)