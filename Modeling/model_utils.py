import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
import mlflow
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import psutil
import platform

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = {
            'CENSURE': 0, 'IMMORAL_NONE': 1, 'HATE': 2, 
            'DISCRIMINATION': 3, 'SEXUAL': 4, 'ABUSE': 5, 
            'VIOLENCE': 6, 'CRIME': 7
        }

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label_list = self.labels[idx].split(',')
        label_vector = [0] * len(self.label2id)
        for label in label_list:
            if label in self.label2id:
                label_vector[self.label2id[label]] = 1

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label_vector, dtype=torch.float)
        }

class BERTClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask
        ).pooler_output
        return self.fc(self.drop(pooled_output))

def setup_gpu():
    """Setup GPU if available"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        return True
    print("GPU unavailable. Using CPU.")
    return False

def create_data_loaders(train_df, val_df, test_df, tokenizer, batch_size):
    """Create DataLoader objects for train, validation and test sets"""
    train_loader = DataLoader(
        TextClassificationDataset(train_df['Text'].values, train_df['Category'].values, tokenizer),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TextClassificationDataset(val_df['Text'].values, val_df['Category'].values, tokenizer),
        batch_size=batch_size
    )
    test_loader = DataLoader(
        TextClassificationDataset(test_df['Text'].values, test_df['Category'].values, tokenizer),
        batch_size=batch_size
    )
    return train_loader, val_loader, test_loader

def log_system_metrics(step=None):
    """Log system metrics to MLflow"""
    cpu_percent = psutil.cpu_percent(interval=1)
    virtual_memory = psutil.virtual_memory().percent
    mlflow.log_metric("cpu_usage", cpu_percent, step=step)
    mlflow.log_metric("memory_usage", virtual_memory, step=step)

def evaluate_model(model, val_loader, device):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    all_labels, all_preds = [], []
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            all_preds.extend((torch.sigmoid(outputs) > 0.5).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return {
        'val_loss': total_loss / len(val_loader), 
        'val_accuracy': accuracy, 
        'val_f1': f1
    }

def train_model(model, train_loader, val_loader, epochs, learning_rate, device):
    """Train the model"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    best_val_f1 = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # Log metrics every 100 steps
            if step % 100 == 0:
                mlflow.log_metric("train_loss_step", loss.item(), step=step + epoch * len(train_loader))
                log_system_metrics(step=step + epoch * len(train_loader))

        # Evaluate at the end of each epoch
        val_metrics = evaluate_model(model, val_loader, device)
        mlflow.log_metrics({
            'train_loss_epoch': total_loss / len(train_loader),
            **val_metrics
        }, step=epoch)

        # Save best model
        if val_metrics['val_f1'] > best_val_f1:
            best_val_f1 = val_metrics['val_f1']
            torch.save(model.state_dict(), 'model.pkl')
            mlflow.log_artifact('model.pkl')

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Training Loss: {total_loss / len(train_loader):.4f}")
        print(f"Validation Loss: {val_metrics['val_loss']:.4f}")
        print(f"Validation Accuracy: {val_metrics['val_accuracy']:.4f}")
        print(f"Validation F1: {val_metrics['val_f1']:.4f}")