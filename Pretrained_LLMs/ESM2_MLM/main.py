import pandas as pd
import torch
import random 
from dataset import ESM2_MLM_Dataset
from model import ESM2_MLM 
from train import train 
import random  

if __name__=="__main__":
    df = pd.read_csv('../data/model_data_full.csv') 
    
    # # Heavy Chain Model 
    # vh_seqs = df['Antibody VH'].unique().tolist() 
    # max_length = 170 
    # mlm_mask_rate = 0.3 
    # n = int(0.9*len(vh_seqs)) 
    # train_sequences = random.sample(vh_seqs, n)
    # valid_sequences = vh_seqs 
    # for x in train_sequences: valid_sequences.remove(x)  
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = ESM2_MLM(device) 
    # model.load_esm2_weights() 
    # train(model, train_sequences, valid_sequences, max_length, mlm_mask_rate, 
    #       num_epochs=30, save_path='saved_model/vh_esm2_mlm', 
    #       batch_size=64, learning_rate=5e-5, max_grad_norm=1.0, device=device)
    
    
    # Light Chain Model 
    vl_seqs = df['Antibody VL'].unique().tolist() 
    max_length = 125 
    mlm_mask_rate = 0.3 
    n = int(0.9*len(vl_seqs)) 
    train_sequences = random.sample(vl_seqs, n)
    valid_sequences = vl_seqs 
    for x in train_sequences: valid_sequences.remove(x)  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ESM2_MLM(device) 
    model.load_esm2_weights() 
    train(model, train_sequences, valid_sequences, max_length, mlm_mask_rate, 
          num_epochs=30, save_path='saved_model/vl_esm2_mlm', 
          batch_size=64, learning_rate=5e-5, max_grad_norm=1.0, device=device)
    
    
     

