import pandas as pd
import numpy as np
from transformers import MPNetTokenizer, MPNetForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from tqdm.auto import tqdm



# Set device to 'mps' for running on Azure Machine Learning Hardware
device = torch.device('mps')

# Load CSV data
df = pd.read_csv('./processed_description.csv')

# Ensure the salary_bin is of type 'category' to handle it more efficiently
df['salary_bin'] = df['salary_bin'].astype('category')
df['processed_description'] = df['processed_description'].astype(str)  # Ensure all text data is string type

# Initialize the MPNet tokenizer and model
tokenizer = MPNetTokenizer.from_pretrained('microsoft/mpnet-base')

class JobDescriptionDataset(Dataset):
    def __init__(self, descriptions, labels, tokenizer, max_len=512):
        self.descriptions = descriptions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        description = self.descriptions[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            description,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
dataset = JobDescriptionDataset(df['processed_description'], df['salary_bin'], tokenizer)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize MPNet for sequence classification
model = MPNetForSequenceClassification.from_pretrained(
    'microsoft/mpnet-base',
    num_labels=len(df['salary_bin'].cat.categories)
).to('mps')  


from torch.optim import AdamW
from tqdm import tqdm

optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)

train_loss = train_epoch(model, loader, loss_fn, optimizer, device='mps')
print(f"Training loss: {train_loss}")


