# This script finetunes a pretrained clip model
# Usage fine_tune_clip.py config_file

import json
import os
import sys
from PIL import Image

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

import clip
import time
from transformers import CLIPProcessor, CLIPModel
from csv import DictWriter
import re

stop_words_filename = '/media/avsr2.0/code/Backend/stop_words.txt'

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


class MultimodalDataset(Dataset):
    def __init__(self, dataset_files, preprocess, load_keyword_files = False):
        self.preprocess = preprocess
        self.images = []
        self.texts = []
        if type(dataset_files) != list:
            dataset_files = list(dataset_files)
        num_files = 0

                
        with open(stop_words_filename) as  stop_words_file:
            self.stop_words = stop_words_file.readlines()

        for i in range(0,len(self.stop_words)):
            self.stop_words[i] = self.stop_words[i].strip()

        for data_file in dataset_files:
            if data_file[-5:] != '.json':
               json_file = os.path.join(data_file,'image_description.train_sample.json')
               if os.path.isfile(json_file):
                    real_data_file = json_file
               else:
                   print('WARNING. Skipping dir: ', data_file, '. No data files found there')
                   continue
            else:
                real_data_file = data_file
            print('Loading data from ', data_file)
            num_files += 1
            with open(real_data_file) as json_file:
                json_data = json.load(json_file)
                # This is format 1: Car manuals labeled with GPT-4
                if isinstance(json_data,dict):
                    img_path = json_data['images_path']
                    data = json_data['images']
                    json_data = []
                    for row in data:
                        json_data.append({'image_path':os.path.join(img_path,row['image']),'text':row['description']})



                for item in json_data:
                    if not 'image_path' in item:
                        print('ERROR. Image path missing in file: ', dataset_file)
                        quit()
                    else:
                        # If running out of memory, just store the image paths
                        # and load the image when building the minibatch
#                        self.images.append(self.preprocess(Image.open(item['image_path'])))
                        self.images.append(item['image_path'])


                    if not 'text' in item:
                        print('ERROR. Text missing in file: ', dataset_file)
                        quit()
                    else:                
                        self.texts.append(clip.tokenize(self.filter_stop_words(item['text']),context_length=77,truncate=True))


            # If there is a data file that contains key words it is also loaded

            
            if load_keyword_files and os.path.isfile(os.path.join(data_file,'image_description.json.keywordsadded.json')):
                print('Loading additional data (keywords) from ', data_file,'/image_description.json.keywordsadded.json')
                new_data_file = os.path.join(data_file,'image_description.json.keywordsadded.json')
                with open(new_data_file) as json_file:
                    json_data = json.load(json_file)
                    # This is format 1: Car manuals labeled with GPT-4
                    if isinstance(json_data,dict):
#                        img_path = json_data['images_path']
                        data = json_data['images']
                        json_data = []
                        for row in data:
                            json_data.append({'image_path':os.path.join(img_path,row['image']),'text':row['keywords']})



                    for item in json_data:
                        if not 'image_path' in item:
                            print('ERROR. Image path missing in file: ', dataset_file)
                            quit()
                        else:
                            # If running out of memory, just store the image paths
                            # and load the image when building the minibatch
    #                        self.images.append(self.preprocess(Image.open(item['image_path'])))
                            self.images.append(item['image_path'])


                        if not 'text' in item:
                            print('ERROR. Text missing in file: ', dataset_file)
                            quit()
                        else:                
                            self.texts.append(clip.tokenize(item['text'],context_length=77,truncate=True))


                        
            
        print('>>> Number of input files in dataset: ', num_files)



    def filter_stop_words(self,input_text):
        for word in self.stop_words:
            input_text = input_text.replace(' '+word+' ',' ')
            input_text = input_text.replace(' '+word+'.',' ')
        input_text = input_text.replace('.',' ')
        input_text = input_text.replace(',',' ')
        input_text = input_text.replace('\n',' ')
        input_text = input_text.replace('(',' ')
        input_text = input_text.replace(')',' ')
        input_text = input_text.replace(';',' ')
        input_text = input_text.replace(':',' ')
        input_text = re.sub('[ ]+', ' ', input_text)
        input_text
        return input_text
    
                        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Preprocess image using CLIP's preprocessing function
        image = self.preprocess(Image.open(self.images[idx]))
        return image,self.texts[idx]
 #       return self.images[idx],self.texts[idx]



class CLIP:
    def __init__(self,config_file, device='cpu', model_to_resume = None):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
         
        
        #self.device = device
#        self.device = 'cpu'
#        print('self.device = ', self.device)


        with open(config_file) as json_file:
            config_data = json.load(json_file)
            train_data_file = config_data.get('train_data',None)
            self.learning_rate = config_data.get('learning_rate', 1e-07)
            self.batch_size = config_data.get('batch_size',64)
            self.base_model = config_data.get('base_model',"ViT-B/32")
            self.eval_batch_size = config_data.get('eval_batch_size',32)
            self.weight_decay = config_data.get('weight_decay', 0.001)
            self.num_epochs = config_data.get('num_epochs', 30)
            self.out_dir = config_data.get('out_dir','/tmp/')
            if self.out_dir == '/tmp/':
                print('WARNING. Out dir not specified. Using /tmp/')

            if not train_data_file:
                print('ERROR. Train data not defined in config file')
                quit()
            dev_data_file = config_data.get('dev_data',None)
            if not dev_data_file:
                print('WARNING. Dev data not defined in config file')


            self.model, self.preprocess = clip.load(self.base_model, device=self.device, jit=False)
            print('>>> Base model = ', self.base_model)
            if model_to_resume:
                print('Loading pretraining model from file ', model_to_resume)
                self.model.load_state_dict(torch.load(model_to_resume))


            self.train_dataset = MultimodalDataset(train_data_file,self.preprocess)
            self.train_dataloader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle=True) 
            if not dev_data_file:
                print('ERROR. Dev data not defined in config file')
                quit()
            else:
                self.dev_dataset = MultimodalDataset(dev_data_file,self.preprocess)
                self.dev_dataloader =  DataLoader(self.dev_dataset, batch_size = self.eval_batch_size, shuffle=True) 
                

                
            

    def train_epoch(self,epoch_number):
        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()

        train_loss = 0.0
        dev_loss = 0.0

        start_time = time.time()

        print('Evaluating model on the dev set ...', flush=True,end='')
        self.model.eval()
        for batch in self.dev_dataloader:
            
            images,texts = batch 

            images= images.to(self.device)
            texts = texts.to(self.device).squeeze(1)

            # Forward pass
            logits_per_image, logits_per_text = self.model(images, texts)

            # Compute loss
            ground_truth = torch.arange(len(images),dtype=torch.long,device=self.device)
            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            dev_loss += (total_loss.cpu().detach()) / len(batch)


        print(' Done!')
        
        print('Training model on the training set ...',flush = True)
        self.model.train()
        batch_count = 0
        show_batches = len(self.train_dataloader) // 10
        print('batch_size = ', self.batch_size, '  num_batches = ', len(self.train_dataloader))
        for batch in self.train_dataloader:

           if batch_count % show_batches == 0:
               print('Processing batch ',batch_count,'/',len(self.train_dataloader), flush = True)
           batch_count += 1
           self.optimizer.zero_grad()

           images,texts = batch

           images= images.to(self.device)
           texts = texts.to(self.device).squeeze(1)

           # Forward pass
           logits_per_image, logits_per_text = self.model(images, texts)

           # Compute loss
           ground_truth = torch.arange(len(images),dtype=torch.long,device=self.device)
           total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2

           # Backward pass
           total_loss.backward()
           if self.device == "cpu":
               self.optimizer.step()
           else : 
               convert_models_to_fp32(self.model)
               self.optimizer.step()
               clip.model.convert_weights(self.model)

           train_loss += (total_loss.cpu().detach() / len(batch))
           
        print('  Done !')
        end_time = time.time()
        print('+++ Time = ', ((end_time-start_time)/60.0),' minutes')
        return train_loss.item(),dev_loss.item()



    def train_model(self):
        self.progress = []
        rows = []
        if self.device == "cpu":
            self.model.float()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,weight_decay=self.weight_decay) # the lr is smaller, more safe for fine tuning to new dataset
        best_loss = float('inf')
        for epoch in range(self.num_epochs):
            train_loss,dev_loss = self.train_epoch(epoch)
            self.progress.append((train_loss,dev_loss))
            print('>>>>> Epoch ', epoch, ' train_loss = ', train_loss, '  dev_loss = ', dev_loss)
            rows.append({'epoch':epoch,'train_loss':train_loss, 'dev_loss':dev_loss})
            with open(os.path.join(self.out_dir,'progress.csv'),'w') as csv_file:
                writer = DictWriter(csv_file,fieldnames = ['epoch','train_loss','dev_loss'])
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            if epoch % 5 == 0:
                torch.save(self.model.state_dict(), os.path.join(self.out_dir,'model.epoch'+str(epoch)+'.pth'))

            if dev_loss < best_loss:
                torch.save(self.model.state_dict(), os.path.join(self.out_dir,'best_model.pth'))
                best_loss = dev_loss
                
            
            
        torch.save(self.model.state_dict(), os.path.join(self.out_dir,"final_model.pth"))

                
                

def main():

    if not (3 >= len(sys.argv) >= 2):
        print('Usage: python fine_tune_clip.py config.json [model_to_resume]')
        
        quit()


    config_file = sys.argv[1]
    if not os.path.exists(sys.argv[1]):
        print('ERROR. config file ', config_file, ' does not exist')
        quit()

    model_to_resume = None
    if len(sys.argv) > 2:
        model_to_resume = sys.argv[2]
        if not os.path.exists(model_to_resume):
            print('ERROR. File ', model_to_resume, ' not found')
            quit()
                  
    clip_model = CLIP(config_file, model_to_resume = model_to_resume)
    clip_model.train_model()

    
    
    


if __name__ == "__main__":
    main()
   
