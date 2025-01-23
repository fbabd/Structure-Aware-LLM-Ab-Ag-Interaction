import pandas as pd 
import torch   
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader, random_split 
from tqdm import tqdm 
from sklearn.metrics import classification_report 
from Pretrained.ESM2 import ESM2, load_model, convert 



#==========
class CovBind_ESM2(torch.nn.Module):
    def __init__(self, freeze=False):
        super().__init__() 
        self.vh_bert = ESM2() 
        self.vh_bert = load_model(self.vh_bert, 'Pretrained/ESM2/esm2_t30_150M_UR50D.pt','cpu')
        self.vl_bert = ESM2()  
        self.vl_bert = load_model(self.vl_bert, 'Pretrained/ESM2/esm2_t30_150M_UR50D.pt','cpu')
        self.clf = torch.nn.Linear(640*2+320, 1)
        init.xavier_uniform_(self.clf.weight) 
        # Freeze the VH and VL models
        if freeze:
            for param in self.vh_bert.parameters():
                param.requires_grad = False
            for param in self.vl_bert.parameters():
                param.requires_grad = False

    
    def forward(self, vh_id,  vl_id, ag_embs, return_attn=False, return_logits=False):
        vh_res = self.vh_bert(vh_id, repr_layers=[30], return_attention=True) 
        vl_res = self.vl_bert(vl_id, repr_layers=[30], return_attention=True) 
        combined_ab_embs = torch.cat((vh_res['representations'][30][:,0,:].squeeze(1),
                                      vl_res['representations'][30][:,0,:].squeeze(1)), dim=1)  
        #print(ag_embs.shape, combined_ab_embs.shape)
        combined_embs = torch.cat((combined_ab_embs, ag_embs), dim=1)
        output = self.clf(combined_embs)
        if return_attn:
            vh_res['attentions'], vl_res['attentions']
        if return_logits:
            vh_res['logits'], vl_res['logits']
        return output 
 
 
#==========
class SequenceDataset(Dataset):
    def __init__(self, vh_seqs, vl_seqs, targets, labels):
        self.vh_seqs = vh_seqs 
        self.vl_seqs = vl_seqs 
        self.targets = targets
        self.labels = labels 
        self.tg_embs = torch.load('data/target_embeddings.pt') 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        vh_id = torch.tensor(convert(self.vh_seqs[idx], length=170)) 
        vl_id = torch.tensor(convert(self.vl_seqs[idx], length=125))  
        target = self.tg_embs[self.targets[idx]][0,:]
        label = self.labels[idx]
        return vh_id, vl_id, target, torch.tensor(label, dtype=torch.float32) 


#==========
def prepare_dataloaders(vh_seqs, vl_seqs, targets, labels, val_ratio=0.15, batch_size=32):
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
    for vh_id,  vl_id, ag_embs, labels in tqdm(data_loader, desc="Training"):
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
        for vh_id, vl_id, ag_embs, labels in tqdm(data_loader, desc="Validating"):
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
    print("== CovBind-ESM2 ==")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    df = pd.read_csv('data/model_data.csv')  
    
    BATCH_SIZE = 32
    num_epochs = 30  
    train_loader, val_loader = prepare_dataloaders(df['Antibody VH'].tolist(), 
                                               df['Antibody VL'].tolist(), 
                                               df['Target'].tolist(),
                                               df['Binding'].tolist(),
                                               batch_size=BATCH_SIZE)
    
    model_savepath = 'saved_models/covbind_ESM2_pt.pth' 
    print(model_savepath) 
    model = CovBind_ESM2(freeze=True)   
    
    model = train(model, train_loader, val_loader, num_epochs, 0.00001, device, 
                  savepath=model_savepath)  
    
    # Test 
    model.load_state_dict(torch.load(model_savepath, map_location=device), strict=False) 
    df_test = pd.read_csv('data/test_data.csv') 
    dataset_test = SequenceDataset(df_test['Antibody VH'].tolist(), 
                          df_test['Antibody VL'].tolist(),
                          df_test['Target'].tolist(),
                          df_test['Binding'].tolist())
    data_loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False) 
    
    test_run(model, data_loader_test, device) 
    
    print("Execution Completed! ")
    
