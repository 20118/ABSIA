from simpletransformers.seq2seq import Seq2SeqModel
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig

import torch
import numpy as np
import os
import numpy as np
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '4'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def predict_val(model, device):
    candidate_list2 = ["low positive", "moderate positive", "high positive", "neutral", "low negative", "moderate negative", "high negative"]
    candidate_list1 = ["positive", "neutral", "negative"]

    
    model.eval()
    model.config.use_cache = False
    # print("evaluation........................")
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    with open("data/dev.txt", "r") as f:
        file = f.readlines()
    train_data = []
    count1 = 0;count2 = 0;count=0
    total = 0
    for line in file:
        total += 1
        
        score_list1 = []
        score_list2 = []
        score_list3 = []
        score_list4 = []
        score_list5 = []
        line = line.strip()
        
        x, term, golden_polarity, taskk = line.split("\t")[0], line.split("\t")[1], line.split("\t")[2], line.split("\t")[3]
       
        if(taskk=="pol"):
            input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']
            target_list = ["The sentiment polarity of the aspect " + term.lower() + " is " + candi1.lower() +" ." for candi1 in
                       candidate_list1]
            output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        else:
            target_list = ["The aspect " + term.lower() + " has " + candi1.lower() + " intensity "+" ." for candi1 in
                       candidate_list2]
            input_ids = tokenizer([x] * 7, return_tensors='pt')['input_ids']
            output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']

        
        
        with torch.no_grad():
            output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
            logits= output.softmax(dim=-1).to('cpu').numpy()
           
        if(taskk=="pol"):
            for i in range(3):
                score = 1
                for j in range(logits[i].shape[0] - 2):
                    score *= logits[i][j][output_ids[i][j + 1]]
                score_list2.append(score)
            score_list = score_list2
            predict = candidate_list1[np.argmax(score_list)]
           
            if predict == golden_polarity:
                count1 += 1
                count +=1
                
                

    
        if(taskk=="intt"):
            for i in range(7):
                score = 1
                for j in range(logits[i].shape[0] - 2):
                    score *= logits[i][j][output_ids[i][j + 1]]
                score_list2.append(score)
            score_list = score_list2
            predict = candidate_list2[np.argmax(score_list)]
            if predict == golden_polarity:
                count2 += 1;count +=1
               
    total1=total/2
    
    print("validation", total, count1/total1, count2/total1, count/total)
    return count1/total1, count2/total1, count/total

def predict_test(model, device):
    candidate_list2 = ["low positive", "moderate positive",  "high positive", "neutral", "low negative", "moderate negative", "high negative"]
    candidate_list1 = ["positive", "neutral", "negative"]

    
    model.eval()
    model.config.use_cache = False
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    with open("data/test.txt", "r") as f:
        file = f.readlines()
    train_data = []
    count1 = 0;count2 = 0;count=0
    total = 0
    for line in file:
        total += 1
        
        score_list1 = []
        score_list2 = []
        score_list3 = []
        score_list4 = []
        score_list5 = []
        line = line.strip()
        x, term, golden_polarity,taskk = line.split("\t")[0], line.split("\t")[1], line.split("\t")[2], line.split("\t")[3]
        
        if(taskk=="pol"):
            input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']
            target_list = ["The sentiment polarity of the aspect " + term.lower() + " is " + candi1.lower() +" ." for candi1 in
                       candidate_list1]
            output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        else:
            target_list = ["The aspect " + term.lower() + " has " + candi1.lower() + " intensity "+" ." for candi1 in
                       candidate_list2]
            input_ids = tokenizer([x] * 7, return_tensors='pt')['input_ids']
            output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']


        
        with torch.no_grad():
            output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
            logits= output.softmax(dim=-1).to('cpu').numpy()
            
        if(taskk=="pol"):
            for i in range(3):
                score = 1
                for j in range(logits[i].shape[0] - 2):
                    score *= logits[i][j][output_ids[i][j + 1]]
                score_list2.append(score)
            score_list = score_list2
            predict1 = candidate_list1[np.argmax(score_list)]
        
            if predict1 == golden_polarity:
                count1 += 1
                count +=1
       
        else:
            for i in range(7):
                score = 1
                for j in range(logits[i].shape[0] - 2):
                    score *= logits[i][j][output_ids[i][j + 1]]
                score_list2.append(score)
            score_list = score_list2
            predict1 = candidate_list2[np.argmax(score_list)]
        
            if predict1 == golden_polarity:
                count2 += 1 
                count +=1
    total1=total/2
    print("test", total, count1/total1, count2/total1, count/total)
    
    return count1/total1, count2/total1, count/total
