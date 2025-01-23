import pandas as pd
import torch
import random 
from tokenizer import ProteinTokenizer
from dataset import StructureDataset 
from model import CovAbHeavy 
from train import train 
from save_load_model import save_model, load_model 
from torch.utils.data import DataLoader, random_split


if __name__=="__main__":
    vocab_size = 25   
    hidden_size = 768   
    num_layers = 12   
    max_length = 228  
    
    tok = ProteinTokenizer() 
    dataset = StructureDataset('../data/vh_structures/', tok, max_length=max_length) 
    train_ratio = 0.8
    val_ratio = 1 - train_ratio
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize the model
    model = CovAbHeavy(vocab_size, hidden_size=hidden_size, num_layers=num_layers, 
                       max_length=max_length).to(device) 

    
    model = train(model, train_loader, val_loader, 
                  epochs=30, learning_rate=5e-5,  max_grad_norm=1.0, device=device)
     

