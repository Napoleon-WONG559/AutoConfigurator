import torch
from mix_template_model import MixTemplateModel,read_data_csv

class Configurator(object):
    def __init__(self,
                 review_dataset,
                 need_dataset,
                 epoch,
                 template,
                 classes,
                 label_words,
                 ld_plm,
                 ld_tokenizer,
                 ld_WrapperClass,
                 device):
        self.device = device#torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)

        #load pre-trained model
        #graphic_plm, graphic_tokenizer, graphic_model_config, graphic_WrapperClass = load_plm("bert", "bert-base-cased")

        #graphic model
        #graphic_dataset=read_data_csv("data/review_graphic_label_map.csv",[0.9,0.1])
        self.review_dataset=read_data_csv(review_dataset,[0.9,0.1])
        #graphic_train_set, graphic_valid_set=random_split(graphic_dataset,
        #                                                  [0.7,0.3],
        #                                                  generator=torch.Generator().manual_seed(42))
        self.epoch=epoch
        #graphic_template='{"soft": "Someone said : "} {"placeholder":"text_a"} {"soft": "Then he need"} a computer with a {"mask"} graphic card'
        self.template=template
        self.model=MixTemplateModel(ld_plm,#graphic_plm,
                                    ld_tokenizer,#graphic_tokenizer,
                                    ld_WrapperClass,#graphic_WrapperClass,
                                    self.review_dataset,
                                    classes,#graphic_classes,
                                    self.epoch,
                                    self.template,
                                    label_words,#graphic_label_words,
                                    self.device)
        
    
    def train(self):
        self.model.train()
        for i in range(self.epoch):
            count=0
            loss_rec=0
            for batch in self.model.train_data_loader:
                batch.to(self.device)
                labels=batch['label']
                logits=self.model(batch)
                loss=self.model.cross_entropy(logits,labels)
                loss.backward()
                self.model.optimizer1.step()
                self.model.optimizer1.zero_grad()
                self.model.optimizer2.step()
                self.model.optimizer2.zero_grad()
                count+=1
                loss_rec+=loss
            print('NO.',i,' epoch avg loss: ',loss_rec/count)
    
    def validate(self):
        self.model.eval()
        preds=[]
        labels=[]
        for step, inputs in enumerate(self.model.valid_data_loader):
            inputs.to(self.device)
            logits=self.model(inputs)
            label=inputs['label']
            labels.extend(label.cpu().tolist())
            preds.extend(torch.argmax(logits,dim=-1).cpu().tolist())
        acc=sum([int(i==j) for i,j in zip(preds, labels)])/len(preds)

        print("accuracy is : ",acc)

    def finetune(self):
        pass

    def test(self):
        pass