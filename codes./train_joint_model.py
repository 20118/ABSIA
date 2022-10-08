from seq2seq_joint_model import Seq2SeqModel
import pandas as pd

import os
import numpy as np
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '4'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
with open("data/train.txt", "r") as f:
    file = f.readlines()
train_data1 = [];train_data2 = [];train_data=[]
for line in file:
    x, y,task_name = line.split("\t")[0], line.strip().split("\t")[1], line.strip().split("\t")[2]
    train_data1.append([x, y])
  

train_df= pd.DataFrame(train_data1, columns=["input_text", "target_text"])

print("train df", len(train_df))

steps = [1]
learing_rates = [3e-5]



best_accuracy = 0;best_accuracy2 = 0
for lr in learing_rates:
    for step in steps:
        model_args = {
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "max_seq_length": 50,
            "train_batch_size": 16,
            "num_train_epochs":25,
            "save_eval_checkpoints": False,
            "save_model_every_epoch": False,
            "evaluate_during_training": False,
            "evaluate_generated_text": False,
            "evaluate_during_training_verbose": False,
            "use_multiprocessing": False,
            "max_length": 30,
            "manual_seed": 42,
            "gradient_accumulation_steps": step,
            "learning_rate":  lr,
            "save_steps": 99999999999999,
        }

        # Initialize model
        model = Seq2SeqModel(
            encoder_decoder_type="bart",
            encoder_decoder_name="facebook/bart-base",
            args=model_args,
        )

        # Train the model
       
        best_accuracy2 = model.train_model(train_df,best_accuracy,best_accuracy2)