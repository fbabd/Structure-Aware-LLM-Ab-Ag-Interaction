import pandas as pd
import torch
import random 
from model import ProteinBERTForMLM
from train import train 


if __name__=="__main__":
    df = pd.read_csv('../data/model_data_full.csv') 
    num_epochs = 30 
    batch_size = 64 
    
    # # Heavy Chain Sequences 
    # vh_seqs = df['Antibody VH'].unique().tolist()
    # n = int(0.9*len(vh_seqs)) 
    # train_sequences = random.sample(vh_seqs, n)
    # valid_sequences = vh_seqs 
    # for x in train_sequences: valid_sequences.remove(x) 
    # vocab_size = 25   
    # hidden_size = 768   
    # num_layers = 12   
    # max_length = 170  
    # mask_rate = 0.3 
    # model = ProteinBERTForMLM(vocab_size, hidden_size=hidden_size, 
    #                           num_layers=num_layers, max_length=max_length)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # trained_model = train(model=model, train_seqs = train_sequences, val_seqs = valid_sequences, 
    #                       max_length=max_length, mlm_mask_rate=mask_rate, 
    #                       num_epochs=num_epochs,save_path='saved_model/vh_bert_mlm',
    #                       batch_size=batch_size, learning_rate=5e-5, max_grad_norm=1.0, device=device )
    
    
    # Ligth chain sequences 
    vl_seqs = df['Antibody VL'].unique().tolist()
    n = int(0.9*len(vl_seqs)) 
    train_sequences = random.sample(vl_seqs, n)
    valid_sequences = vl_seqs 
    for x in train_sequences: valid_sequences.remove(x) 
    vocab_size = 25   
    hidden_size = 768   
    num_layers = 12   
    max_length = 125   
    mask_rate = 0.3 
    model = ProteinBERTForMLM(vocab_size, hidden_size=hidden_size, 
                              num_layers=num_layers, max_length=max_length)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trained_model = train(model=model, train_seqs = train_sequences, val_seqs = valid_sequences, 
                          max_length=max_length, mlm_mask_rate=mask_rate, 
                          num_epochs=num_epochs,save_path='saved_model/vl_bert_mlm',
                          batch_size=batch_size, learning_rate=5e-5, max_grad_norm=1.0, device=device )

    
     

