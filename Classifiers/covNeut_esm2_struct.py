import pandas as pd 
import torch   
from torch.utils.data import Dataset, DataLoader, random_split 
from tqdm import tqdm 
from sklearn.metrics import classification_report 
from Pretrained.ESM2_MLM_Struct.model import ESM2_MLM_Struct 
from Pretrained.ESM2_MLM_Struct.tokenizer import ESM2_Tokenizer  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


#==========
class CovNeut_ftESM(torch.nn.Module):
    def __init__(self, freeze=False, load_weights=True):
        super().__init__() 
        self.vh_model = ESM2_MLM_Struct(hidden_size=640, max_length=230, device=device)
        self.vl_model = ESM2_MLM_Struct(hidden_size=640, max_length=219, device=device)
        self.clf = torch.nn.Linear(640*2+320, 1)
        
        if load_weights:
            self.vh_model.load_state_dict(torch.load('Pretrained/ESM2_MLM_Struct/saved_model/esm_struct_heavy.pth', map_location=device)) 
            self.vl_model.load_state_dict(torch.load('Pretrained/ESM2_MLM_Struct/saved_model/esm_struct_light.pth', map_location=device))
            
            for param in self.vh_model.binary_matrix_head.parameters():
                    param.requires_grad = False
            for param in self.vl_model.binary_matrix_head.parameters():
                    param.requires_grad = False
        
        # Freeze the VH and VL models
        if freeze:
            for param in self.vh_model.parameters():
                param.requires_grad = False
            for param in self.vl_model.parameters():
                param.requires_grad = False

    
    def forward(self, vh_id,  vl_id, ag_embs, return_attn=False, return_logits=False, return_combined_emb=False):
        vh_logits, vh_embs, vh_attns = self.vh_model(vh_id, output_attentions=True)
        vl_logits, vl_embs, vl_attns = self.vl_model(vl_id, output_attentions=True)
        
        combined_ab_embs = torch.cat((vh_embs[:,0,:].squeeze(1),vl_embs[:,0,:].squeeze(1)), dim=1)  
        #print(ag_embs.shape, combined_ab_embs.shape)
        combined_embs = torch.cat((combined_ab_embs, ag_embs), dim=1)
        output = self.clf(combined_embs)
        if return_attn:
            return vh_attns, vl_attns
        if return_logits:
            return vh_logits, vl_logits 
        if return_combined_emb:
            return combined_embs
        return output 
 
 
#==========
class SequenceDataset(Dataset):
    def __init__(self, vh_seqs, vl_seqs, targets, labels):
        self.vh_seqs = vh_seqs 
        self.vl_seqs = vl_seqs 
        self.targets = targets
        self.labels = labels 
        self.tg_embs = torch.load('data/target_embeddings.pt') 
        self.tok = ESM2_Tokenizer()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        vh_seq = self.tok.encode(self.vh_seqs[idx], max_length=228) 
        vl_seq = self.tok.encode(self.vl_seqs[idx], max_length=217)  
        target = self.tg_embs[self.targets[idx]][0,:] 
        label = self.labels[idx]
        return vh_seq, vl_seq, target, torch.tensor(label, dtype=torch.float32) 


#==========
def prepare_dataloaders(vh_seqs, vl_seqs, targets, labels, val_ratio=0.2, batch_size=32):
    dataset = SequenceDataset(vh_seqs, vl_seqs, targets, labels)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


#========== 
def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for vh_id,  vl_id, ag_embs, labels in tqdm(data_loader, mininterval=30, desc="Training"):
        # Move data to the specified device
        vh_id = vh_id.to(device)
        vl_id = vl_id.to(device)
        ag_embs = ag_embs.to(device) 
        labels = labels.to(device) 
        # Forward pass
        optimizer.zero_grad()
        logits = model(vh_id, vl_id, ag_embs)
        # Compute loss
        loss = criterion(logits.view(-1), labels)
        total_loss += loss.item()
        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()
        print(loss.item(), end=" ", flush=True) 
    avg_loss = total_loss / len(data_loader)
    torch.cuda.empty_cache()  
    return avg_loss


#==========
def val_epoch(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0 
    with torch.no_grad():
        for vh_id, vl_id, ag_embs, labels in tqdm(data_loader, mininterval=20, desc="Validating"):
            # Move data to the specified device
            vh_id = vh_id.to(device)
            vl_id = vl_id.to(device)
            ag_embs = ag_embs.to(device) 
            labels = labels.to(device) 
            # Forward pass
            logits = model(vh_id, vl_id, ag_embs)
            # Compute loss
            loss = criterion(logits.view(-1), labels)
            total_loss += loss.item()
            # Calculate accuracy
            predictions = torch.sigmoid(logits).round()
            correct_predictions += (predictions.view(-1) == labels).sum().item()
            total_samples += labels.size(0)
            torch.cuda.empty_cache() 
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


#==========  
def test_run(model, data_loader, device):
    model.to(device)
    model.eval()
    all_predictions = [] 
    actual_class = []
    with torch.no_grad():
        for vh_ids,  vl_ids, ag_embs, labels in tqdm(data_loader, desc="Validating"):
            vh_ids = vh_ids.to(device) 
            vl_ids = vl_ids.to(device)
            ag_embs = ag_embs.to(device) 
            labels = labels.to(device)
            logits = model(vh_ids, vl_ids, ag_embs)
            predictions = torch.sigmoid(logits).round() 
            all_predictions.extend(predictions.view(-1).tolist())
            actual_class.extend(labels.detach().tolist())
    report = classification_report(actual_class, all_predictions, target_names=['Neg', 'Pos'], digits=4)
    print(report)


#==========  
def train(model, train_loader, val_loader, num_epochs, learning_rate, device, savepath):
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    max_val_acc = 0.0 
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}", flush=True)
    
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        #print(f"Training Loss: {train_loss:.4f}", flush=True)
        val_loss, val_accuracy = val_epoch(model, val_loader, criterion, device)
        print(f"Training Loss: {train_loss:.4f}", flush=True)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}", flush=True) 
        torch.cuda.empty_cache() 
        if val_accuracy > max_val_acc:
            max_val_acc = val_accuracy 
            torch.save(model.state_dict(), savepath) 
            print(f"Saved model from epoch {epoch+1}")
    print("Training complete.")
    return model



#======================
if __name__=="__main__":
    print("== CovBind-ESM2-MLM-Struct ==")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
     
    BATCH_SIZE = 32
    num_epochs = 30  
    
    df = pd.read_csv('data/model_data.csv') 
    train_loader, val_loader = prepare_dataloaders(df['Antibody VH'].tolist(), 
                                               df['Antibody VL'].tolist(), 
                                               df['Target'].tolist(),
                                               df['Neutralizing'].tolist(),
                                               batch_size=BATCH_SIZE)
    
    df_test = pd.read_csv('data/test_data.csv') 
    dataset_test = SequenceDataset(df_test['Antibody VH'].tolist(), 
                          df_test['Antibody VL'].tolist(),
                          df_test['Target'].tolist(),
                          df_test['Neutralizing'].tolist())
    data_loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False) 
     
    
    savepath = 'saved_models/covneut_ESM2_Struct_pt.pth'
    print(savepath)
    model = CovNeut_ftESM(freeze=True) 
    model = train(model, train_loader, val_loader, num_epochs, 0.00005, device, 
                  savepath=savepath) 
    model.load_state_dict(torch.load(savepath, map_location=device)) 
    test_run(model, data_loader_test, device) 
    
    
    savepath = 'saved_models/covneut_ESM2_Struct_ft.pth'
    print(savepath)
    model = CovNeut_ftESM(freeze=False) 
    model = train(model, train_loader, val_loader, num_epochs, 0.00005, device, 
                  savepath=savepath) 
    model.load_state_dict(torch.load(savepath, map_location=device)) 
    test_run(model, data_loader_test, device) 
    
    print("Execution Completed! ")
    



