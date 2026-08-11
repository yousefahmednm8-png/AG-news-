#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from collections import Counter 
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn 
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


# In[2]:


train_data=pd.read_csv(r"E:\Datasets\AG news\train.csv")
test_data=pd.read_csv(r"E:\Datasets\AG news\test.csv")


# In[3]:


train_data.head()


# In[4]:


print(train_data.info())


# In[5]:


train_data['text']=train_data["Title"]+" "+train_data["Description"]
test_data['text']=test_data["Title"]+" "+test_data["Description"]
train_data["Description"].head()


# In[6]:


train_data['Class Index'].value_counts()


# In[7]:


train_data["word_count"]=train_data["text"].apply(
    lambda x:len(str(x).split())
)
train_data['word_count'].describe()


# In[8]:


train_data.isnull().sum()


# In[9]:


plt.hist(train_data["word_count"],bins=50)
plt.show()


# In[10]:


train_data["word_count"].quantile([0.5,0.7,0.8,0.95,0.99])


# In[11]:


for i in range(20):
    print(train_data["text"].iloc[i])


# In[12]:


import re

def clean_text(text):
    text = text.lower()
    text = text.replace("\\", " ")
    text = re.sub(r"#\d+;", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    # keep letters/numbers/basic punctuation
    text = re.sub(r"[^a-z0-9.,!?'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# In[13]:


train_data['text']=train_data['text'].apply(clean_text)
test_data['text']=test_data['text'].apply(clean_text)


# In[14]:


train_data,val_data=train_test_split(
    train_data,
    test_size=0.1,
    random_state=42,
    stratify=train_data["Class Index"]
)


# In[15]:


def tokenize(text):
    return text.split()
train_tk=train_data['text'].apply(tokenize)
test_tk=test_data['text'].apply(tokenize)
val_tk=val_data['text'].apply(tokenize)


# In[16]:


counter=Counter()
for tokens in train_tk:
    counter.update(tokens)
print('number of unique words :',len(counter))
print(counter.most_common(10))


# In[17]:


min_freq=2
vocab={
    "<PAD>":0,
    "<UNK>":1
}

for word,freq in counter.items():
    
    if freq>=min_freq:
        vocab[word]=len(vocab)
print("vocab_size",len(vocab))


# In[18]:


def numericallize(tokens,vocab):
    ids=[]
    for token in tokens:
        if token in vocab:
            ids.append(vocab[token])
        else:
            ids.append(vocab["<UNK>"])
    return ids 


# In[19]:


x_train_ids=[
    numericallize(tokens,vocab)
    for tokens in train_tk
]
x_test_ids=[
    numericallize(tokens,vocab)
    for tokens in test_tk
]
x_val_ids=[
    numericallize(tokens,vocab)
    for tokens in val_tk
]


# In[20]:


class AGNewsDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences=sequences
        self.labels=labels
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx],dtype=torch.long),
            torch.tensor(self.labels[idx],dtype=torch.long)
        )

def collate_fn(batch):
    sequences,labels=zip(*batch)
        
    sequences=pad_sequence(
    sequences,
    batch_first=True,
    padding_value=vocab["<PAD>"]
        )
        
    attention_mask=sequences!=vocab["<PAD>"]
    labels=torch.stack(labels)
    return sequences,attention_mask,labels


# In[21]:


train_data["Class Index"] -= 1
val_data["Class Index"] -= 1
test_data["Class Index"] -= 1


# In[22]:


train = AGNewsDataset(
    x_train_ids,
    train_data["Class Index"].tolist()
)

val = AGNewsDataset(
    x_val_ids,
    val_data["Class Index"].tolist()
)

test = AGNewsDataset(
    x_test_ids,
    test_data["Class Index"].tolist()
)


# In[23]:


train_loader=DataLoader(
    train,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_fn
)
val_loader=DataLoader(
    val,
    batch_size=64,
    shuffle=False,
    collate_fn=collate_fn
)
test_loader=DataLoader(
    test,
    batch_size=64,
    shuffle=False,
    collate_fn=collate_fn
)


# In[24]:


embed_dim=128
num_head=4
num_layer=4
ff_dim=128
num_classes=4
dropout=0.2
device="cuda" if torch.cuda.is_available() else "cpu"
pad_idx=vocab["<PAD>"]
device


# In[25]:


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=512, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embed_dim, 2)
            * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )

        pe = torch.zeros(max_len, embed_dim)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# In[26]:


class AGNewsTransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        ff_dim,
        num_classes,
        pad_idx,
        dropout=0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx
        )

        self.positional_encoding = PositionalEncoding(
            embed_dim=embed_dim,
            dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(
            embed_dim,
            num_classes
        )

    def forward(self, x, attention_mask):

        x = self.embedding(x)

        x = self.positional_encoding(x)

        padding_mask = ~attention_mask

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        mask = attention_mask.unsqueeze(-1)

        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        x = self.classifier(x)

        return x
    


# In[27]:


vocab_size = len(vocab)

model = AGNewsTransformer(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_heads=num_head,
    num_layers=num_layer,
    ff_dim=ff_dim,
    num_classes=num_classes,
    pad_idx=vocab["<PAD>"],
    dropout=dropout
).to(device)
print(model)


# In[28]:


criterion=nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer=torch.optim.AdamW(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-4
)


# In[29]:


def train_model(model,train_loader,val_loader,critertion,optimizer,device,epochs=8):
    train_losses=[]
    val_losses=[]
    train_accs=[]
    val_accs=[]
    
    best_val_loss = float("inf")
    for epoch in range(epochs):
        
        model.train()
        train_loss=0
        train_correct=0
        train_total=0
        for x,mask,y in train_loader:
            x,mask,y =x.to(device),mask.to(device),y.to(device)
            
            optimizer.zero_grad()
            outputs=model(x,mask)
            loss=critertion(outputs,y)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            train_correct+=(outputs.argmax(1)==y).sum().item()
            train_total+=y.size(0)
        train_loss/=len(train_loader)
        train_acc=train_correct/train_total
        
        model.eval()
        val_loss=0
        val_correct=0
        val_total=0
        
        with torch.no_grad():
            for x,mask,y in val_loader:
                x,mask,y,=x.to(device),mask.to(device),y.to(device)
                outputs=model(x,mask)
                loss=critertion(outputs,y)
                val_loss+=loss.item()
                val_correct+=(outputs.argmax(1)==y).sum().item()
                val_total+=y.size(0)
            val_loss/=len(val_loader)
            val_acc=val_correct/val_total
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                torch.save(
                    model.state_dict(),
                    "best_transformer_model.pth"
                )

                print("Best model saved")
                        
                print(
                        f"Epoch {epoch+1}/{epochs} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Train Acc: {train_acc:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Acc: {val_acc:.4f}"
                    )

    return train_losses, val_losses, train_accs, val_accs
        


# In[30]:


train_losses, val_losses, train_accs, val_accs = train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=8
)


# In[37]:


def test_model(model, test_loader, criterion, device):
    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, mask, y in test_loader:
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)

            outputs = model(x, mask)
            loss = criterion(outputs, y)

            test_loss += loss.item()

            preds = outputs.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    test_loss /= len(test_loader)
    test_acc = correct / total

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    return test_loss, test_acc, all_labels, all_preds


# In[38]:


test_loss, test_acc, test_labels, test_preds = test_model(
    model,
    test_loader,
    criterion,
    device
)


# In[39]:


import matplotlib.pyplot as plt

def plot_history(train_losses, val_losses, train_accs, val_accs):

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 5))

    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Val Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.show()


# In[40]:


plot_history(
    train_losses,
    val_losses,
    train_accs,
    val_accs
)


# In[41]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(
    test_labels,
    test_preds
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["World", "Sports", "Business", "Sci/Tech"]
)

disp.plot()
plt.title("AG News Confusion Matrix")
plt.show()


# In[ ]:




