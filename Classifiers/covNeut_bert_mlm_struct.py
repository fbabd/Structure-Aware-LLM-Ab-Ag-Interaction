import pandas as pd 
import torch   
from torch.utils.data import Dataset, DataLoader, random_split 
from tqdm import tqdm 
from sklearn.metrics import classification_report  
from Pretrained.VH_BERT_MLM_Struct.save_load_model import load_vh_bert_struct_model 
from Pretrained.VL_BERT_MLM_Struct.save_load_model import load_vl_bert_struct_model 
from Pretrained.VH_BERT_MLM_Struct.tokenizer import ProteinTokenizer 




#====== 
class CovBind(torch.nn.Module):
    def __init__(self, freeze=False) -> None:
        super().__init__()
        self.vh_bert = load_vh_bert_struct_model() 
        self.vl_bert = load_vl_bert_struct_model()  
        self.clf = torch.nn.Linear(768*2+320, 1)
             
        if freeze:
                for param in self.vh_bert.parameters():
                    param.requires_grad = False
                for param in self.vl_bert.parameters():
                    param.requires_grad = False
        
    def forward(self, vh_ids, vh_masks, vl_ids, vl_masks, ag_embs):
        #print(vh_ids.shape, vh_masks.shape)
        vh_logits, vh_embs, vh_attns = self.vh_bert(vh_ids, vh_masks, output_attentions=True) 
        vl_logits, vl_embs, vl_attns = self.vl_bert(vl_ids, vl_masks, output_attentions=True) 
        combined_ab_embs = torch.cat((vh_embs[:,0,:].squeeze(1), vl_embs[:,0,:].squeeze(1)), dim=1) 
        #print(combined_ab_embs.shape, ag_embs.shape) 
        combined_embs = torch.cat((combined_ab_embs, ag_embs), dim=1)
        output = self.clf(combined_embs)
        return output


#====== 
class SequenceDataset(Dataset):
    def __init__(self, vh_seqs, vl_seqs, targets, labels):
        self.vh_seqs = vh_seqs 
        self.vl_seqs = vl_seqs 
        self.targets = targets
        self.labels = labels 
        self.tokenizer = ProteinTokenizer() 
        self.tg_embs = torch.load('data/target_embeddings.pt') 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx): 
        vh_ids, vh_masks =  self.tokenizer.encode(self.vh_seqs[idx], max_length=228) 
        vl_ids, vl_masks = self.tokenizer.encode(self.vl_seqs[idx], max_length=217) 
        target = self.tg_embs[self.targets[idx]][0,:]
        label = self.labels[idx]
        return vh_ids, vh_masks, vl_ids, vl_masks, target, torch.tensor(label, dtype=torch.float32)  


#====== 
def prepare_dataloaders(vh_seqs, vl_seqs, targets, labels, val_ratio=0.2, batch_size=32):
    dataset = SequenceDataset(vh_seqs, vl_seqs, targets, labels,)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


#====== 
def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for vh_ids, vh_masks, vl_ids, vl_masks, ag_embs, labels in tqdm(data_loader, desc="Training"):
        vh_ids, vh_masks = vh_ids.to(device), vh_masks.to(device) 
        vl_ids, vl_masks = vl_ids.to(device), vl_masks.to(device)
        ag_embs = ag_embs.to(device) 
        labels = labels.to(device) 

        optimizer.zero_grad()
        logits = model(vh_ids, vh_masks, vl_ids, vl_masks, ag_embs)

        loss = criterion(logits.view(-1), labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        print(loss.item(), end=" ", flush=True)
    avg_loss = total_loss / len(data_loader)
    return avg_loss


#======
def val_epoch(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for vh_ids, vh_masks, vl_ids, vl_masks, ag_embs, labels in tqdm(data_loader, desc="Validating"):
            vh_ids, vh_masks = vh_ids.to(device), vh_masks.to(device) 
            vl_ids, vl_masks = vl_ids.to(device), vl_masks.to(device)
            ag_embs = ag_embs.to(device) 
            labels = labels.to(device)

            logits = model(vh_ids, vh_masks, vl_ids, vl_masks, ag_embs)
            loss = criterion(logits.view(-1), labels)
            total_loss += loss.item()

            predictions = torch.sigmoid(logits).round()
            correct_predictions += (predictions.view(-1) == labels).sum().item()
            total_samples += labels.size(0)
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


#==========  
def test_run(model, data_loader, device):
    model.to(device)
    model.eval()
    all_predictions = [] 
    all_actual_class = [] 
    with torch.no_grad():
        for vh_ids, vh_masks, vl_ids, vl_masks, ag_embs, labels in tqdm(data_loader, desc="Validating"):
            vh_ids, vh_masks = vh_ids.to(device), vh_masks.to(device) 
            vl_ids, vl_masks = vl_ids.to(device), vl_masks.to(device)
            ag_embs = ag_embs.to(device) 
            labels = labels.to(device)
            logits = model(vh_ids, vh_masks, vl_ids, vl_masks, ag_embs)
            predictions = torch.sigmoid(logits).round() 
            all_predictions.extend(predictions.view(-1).tolist())
            all_actual_class.extend(labels.detach().tolist())
    report = classification_report(all_actual_class, all_predictions, digits=4, target_names=['Neg', 'Pos'])
    print(report)
    

#======  
def train(model, train_loader, val_loader, num_epochs, learning_rate, device, savepath):
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    max_val_acc = 0.0 
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}", flush=True)
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}", flush=True)
        # Validate after each epoch
        val_loss, val_accuracy = val_epoch(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}", flush=True) 
        if val_accuracy>max_val_acc:
            torch.save(model.state_dict(), savepath) 
            print(f"Saved from epoch {epoch+1}")
    print("Training complete.")
    return model



#======
if __name__=="__main__":
    print("== CovNeut-MLM+Struct ==")
    df = pd.read_csv('data/model_data.csv')  
    BATCH_SIZE = 32 
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
    data_loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)
    
    num_epoch = 30 
    learning_rate = 0.00001 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    savepath = "saved_models/covNeut_mlm_struct_pt.pth" 
    print(savepath) 
    model = CovBind(freeze=True) 
    model = train(model, train_loader, val_loader, num_epoch, learning_rate, device, savepath) 
    model.load_state_dict(torch.load(savepath), strict=False) 
    test_run(model, data_loader_test, device) 
    
    savepath = "saved_models/covNeut_mlm_struct_ft.pth" 
    print(savepath) 
    model = CovBind(freeze=False) 
    model = train(model, train_loader, val_loader, num_epoch, learning_rate, device, savepath) 
    model.load_state_dict(torch.load(savepath), strict=False) 
    test_run(model, data_loader_test, device) 
    
