from transformers import T5Tokenizer, T5EncoderModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import pandas as pd 
import os 
import matplotlib.pyplot as plt

######################### (HYPER)PARAMETERS #########################

BATCH_SIZE = 1
R = 4
ALPHA = 1
DROPOUT = 0.05

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\n>>> Device: {device}')
data_path = '/home/fs01/mv487/ProteinLM/training data/SubLoc/'

######################### (HYPER)PARAMETERS ##########################

########################## CLASSES & FUNCTIONS ##########################

def get_data(name, data_path=data_path, extension='.pkl'):
    file_name = os.path.join(data_path, name) + extension                       
    data = pd.read_pickle(file_name) #open binary mode (instead of text mode #with open(file_name) as f:)
    return data

def _printt(tensor, name, check_shape_only=False):
    if not check_shape_only:
        print(f'{name}: {tensor} -- Shape = {tensor.shape} -- Dtype = {tensor.dtype}')
    else:
        print(f'{name}: Shape = {tensor.shape} -- Dtype = {tensor.dtype}')

def inspect_data_pd(df, name):
    print(f'\n>>> {name} data:\n')
    print(f'## Shape: {df.shape}')
    print(f'## Columns: {df.columns}')
    print(f'## Dtypes: {df.dtypes}')
    print(df.head())
    loc_num_groupby = None 
    for col in train_data.columns[1:]: #Except for Sequence column
        print(f"\nColumn: {col}")
        #print(train_data[col].unique())
        column = train_data[col]
        change = column != column.shift() # Identify where value changes   
        group_id = change.cumsum() # Assign group IDs to contiguous blocks
        result = train_data.groupby(group_id).agg( # Aggregate start and end indices per block
            value=(col, "first"),
            start_index=(col, lambda x: x.index[0]),
            end_index=(col, lambda x: x.index[-1])
        )
        if col == 'loc num':
            loc_num_groupby = result
        print(result)
        print("-" * 40)
    print(df["Sequence"].apply(len).describe())
    return loc_num_groupby

def save_model(model, path='Trained_LoRA.pth'):
    torch.save({"model_state_dict": model.state_dict()}, path)

def load_model(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

class ProteinDataset(Dataset):
    def __init__(self, input_ids, labels, seq_lens=None):
        self.input_ids = input_ids
        self.labels = labels
        self.seq_lens = seq_lens
        self.attention_mask = (self.input_ids != 0)
        # if seq_lens is not None:
        #     for i in range(len(self.labels)):
        #         self.input_ids[i, self.seq_lens[i]-1] = 0 #don't count </s>

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "attention_mask": self.attention_mask[idx]
        }
        return item
    
class LoRA(nn.Module):
    def __init__(self, base_module, r=R, alpha=ALPHA, dropout=DROPOUT, device=device):
        super().__init__()
        self.base = base_module
        self.r = r
        self.alpha = alpha
        self.scaling = alpha/r
        self.dropout = nn.Dropout(dropout)
        in_dim = base_module.in_features
        out_dim = base_module.out_features

        #LoRA matrices
        #Does A needs random directions, healthy variance, non-degenerate gradients??
        # -> Just go with simple random for now
        #B: init = 0 so that init Delta_W = 0 (ie. W' = W)
        #nn.init.kaiming_uniform_(self.A, a=math.sqrt(5)) #LeakyReLU with slope w/ negative slope = sqrt(5)
        self.B = nn.Parameter(torch.zeros(out_dim, r, device=device))
        self.A = nn.Parameter(torch.randn(r, in_dim, device=device) * 0.01)
        nn.init.zeros_(self.B) 

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (self.dropout(x) @ self.A.T) @ self.B.T
        return base_out + self.scaling * lora_out #W' = W + (alpha/r)*BA
    
def add_lora(model, r=R, alpha=ALPHA):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            child_name = name.split(".")[-1]
            if child_name in ["q", "k", "v", "o"]: #find the matrices
                parent_name = ".".join(name.split(".")[:-1])
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, LoRA(module, r, alpha))

def check_lora(model):
    for name, module in model.named_modules():
        if isinstance(module, LoRA):
            print(name, "->", module)

def check_trainable_params(model):
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name)

class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        
        self.blk = nn.Sequential(
            nn.LayerNorm(input_dim),   
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),                 
            nn.Dropout(p=0.5),         
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.blk(x) 


class T5LoRAClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim=512, num_classes=10):
        super().__init__()
        self.encoder = encoder
        d_model = encoder.config.d_model
        self.head = MLPHead(input_dim=d_model, hidden_dim=hidden_dim, num_classes=num_classes)

    def mean_pooling(self, tensor, attention_mask):
        mask = attention_mask.unsqueeze(-1).float() #2d -> 3d
        tensor = tensor * mask
        return tensor.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9) #clamp to avoid divide 0

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        pooled = self.mean_pooling(hidden, attention_mask)
        logits = self.head(pooled)
        return logits


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(dataloader)
    acc = correct / total

    return avg_loss, acc

def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs):

    train_losses = []
    val_losses = []
    val_accs = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val Acc:    {val_acc:.4f}")
        print("-" * 40)

    return train_losses, val_losses, val_accs

# for batch in train_loader:
#     input_ids = batch["input_ids"].to(device)
#     attention_mask = batch["attention_mask"].to(device)
#     labels = batch["labels"].to(device)
#     optimizer.zero_grad()
#     logits = model(input_ids=input_ids, attention_mask=attention_mask)
#     loss = criterion(logits, labels)
#     loss.backward()
#     optimizer.step()

def plot_curves(train_losses, val_losses, val_accs):
    """Plot the loss curves"""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 4))
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Validation Accuracy")
    plt.show()

########################## CLASSES & FUNCTIONS ##########################


########################## DATA PROCESSING & TESTS ##########################
name = ['train','valid','test']
train_data, valid_data, test_data = [get_data(n) for n in name]
for df in [train_data, valid_data, test_data]:
    df["length"] = df["Sequence"].str.len()
loc_num_groupby = inspect_data_pd(train_data, 'Train')
#exit()

#Okay, now how does ProtT5 work 

AA = "ACDEFGHIKLMNPQRSTVWY"
model_name = "Rostlab/prot_t5_xl_uniref50"

"""NOTE: WE WILL USE legacy=True -- SIMILAR TO WHAT THEY DID IN THEIR GITHUB"""
tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False) #Keep using capitalized Amino Acids
model = T5EncoderModel.from_pretrained(model_name).to(device)

for p in model.parameters(): #Freeze
    p.requires_grad = False

print(f'\n>>> Model config: {model.config}')
print(f'>>> Model parameters: {sum(p.numel() for p in model.parameters())}')
vocab = tokenizer.get_vocab()
vocab_sorted = dict(sorted(vocab.items(), key=lambda item: item[0]))
print('\n>>> All vocab:')
for key,val in vocab_sorted.items():
    print(f'- {key}: {val}')
vocab_backwards = {val: key for key, val in vocab_sorted.items()}

# =========================================== SANITY CHECK =========================================== #
def first_sanity_test(model=model, tokenizer=tokenizer, examples=None):
    
    print('\n# =========================================== SANITY CHECK =========================================== #')
    if not examples:
        examples = [
            "A L M K T",
            "G G V A L S T P",
            "M S P"
        ]

    tok_tensor = tokenizer(examples, return_tensors="pt", max_length=10000, truncation=True, padding=True)['input_ids'] #pad to handle varied len
    _printt(tok_tensor, '\n# Example tokenized inputs')
    attn_mask = (tok_tensor != 0)
    _printt(attn_mask, 'Attention mask')
    last_idx = attn_mask.sum(dim=-1).flatten()
    _printt(last_idx, 'Last idx')

    tok_tensor = tok_tensor.to(device)
    attn_mask = attn_mask.to(device)

    #Forward pass
    outputs = model(tok_tensor, attention_mask=attn_mask)
    embeds = outputs.last_hidden_state
    _printt(embeds, '\n# Embeddings (last hidden state)', check_shape_only=True)

    #How does embeddings change with pads & attention masks ?
    idx = 0
    one_row = tok_tensor[idx][attn_mask[idx]].unsqueeze(0)
    _printt(one_row, '\n- One row')

    out_one_row = model(one_row)
    embed_one_row = out_one_row.last_hidden_state
    _printt(embed_one_row, '- Embed one row', check_shape_only=True)

    """NOTE: Tokenizer put </s>=1 at the end of each tokenized sequence. Authors (paper) remove </s> from embed and don't use pad/mask"""
    # Take mean over seq len
    emb_w_pad_mask = embeds[idx, :last_idx[idx]].mean(dim=0)
    embed_one_row = embed_one_row.squeeze(0)[:last_idx[idx]].mean(dim=0) 
    _printt(emb_w_pad_mask, 'Embed pad mask')
    _printt(embed_one_row, 'Embed one row')

    #abs(a-b) <= atol + rtol * abs(b)
    #atol handles difference ~ 0
    #rtol will scale with the value itself (1000.01 or larger will give a lot scaling)
    print(f'Print TRUE if embed pad mask and one row are equivalent: {torch.allclose(emb_w_pad_mask, embed_one_row, atol=1e-6, rtol=0)}')

    diff = (emb_w_pad_mask - embed_one_row).abs()
    print(diff.max().item()) #1.2665987014770508e-07
    

    print('# =========================================== SANITY CHECK =========================================== #')

#first_sanity_test()
"""NOTE: THIS TEST PROVES THAT AGGREGATED TOKENIZING W/ PAD/MASK WILL YIELD EMBED W/ DIF ONLY ~1.2665987014770508e-07 PER ENTRY"""
"""NOTE: Tokenizer put </s>=1 at the end of each tokenized sequence. Authors (paper) remove </s> from embed and don't use pad/mask"""


def tokenizer_check(tokenizer=tokenizer):
    """Test tokenizer output: Length, special tokens, etc.."""
    string = "A"*10
    string_len = len(string)
    string = " ".join(string)
    tok = tokenizer(string, return_tensors='pt')['input_ids']
    _printt(tok, '\n# Test A*10')
    is_match = (tok.shape[-1] == string_len+1 and tok[:,-1] == 1).item()
    print(f'# Print TRUE if the length match (incl. </s>) and last token is </s> (1): {is_match}')

#tokenizer_check()
"""NOTE: the length match (incl. </s>) and last token is </s> (1)"""
#exit()

# =========================================== SANITY CHECK =========================================== #

#Now we can mass tokenize data
train_seq = train_data[train_data.columns[0]].tolist()
train_seq = [' '.join(list(i)) for i in train_seq]
print(f'\n>>> Train seq: Type = {type(train_seq)} -- Length = {len(train_seq)}')

#NOTE: I will pad the sequences here, so I also need to track the length for attention_mask later
tokenized_train_seq = tokenizer(train_seq, return_tensors='pt', padding=True, max_length=1000, truncation=True)['input_ids']
train_labels = torch.tensor(train_data['loc_num'].tolist())
train_len = torch.tensor(train_data['length'].tolist())
_printt(tokenized_train_seq, '# Tokenized train seq', check_shape_only=True)
_printt(train_labels, '# Train labels', check_shape_only=True)
_printt(train_len, '# Train len', check_shape_only=True)

train_dataset = ProteinDataset(tokenized_train_seq, train_labels, train_len)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# batch example
for batch in train_loader:
    print(f'\n# Example batch input: {batch["input_ids"].shape}')
    print(f'# Example batch label: {batch["labels"].shape}')
    break

val_seq = valid_data[valid_data.columns[0]].tolist()
val_seq = [' '.join(list(i)) for i in val_seq]
val_len = torch.tensor(valid_data['length'].tolist())
tokenized_val_seq = tokenizer(val_seq, return_tensors='pt', padding=True, max_length=1000, truncation=True)['input_ids']
val_labels = torch.tensor(valid_data['loc_num'].tolist())
val_dataset = ProteinDataset(tokenized_val_seq, val_labels, val_len)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
########################## DATA PROCESSING & TESTS ##########################


########################## IMPLEMENTATION & TRAIN ##########################

#add LoRA
add_lora(model)
model = T5LoRAClassifier(model, hidden_dim=512, num_classes=10).to(device)
#check_lora(model)
check_trainable_params(model)
print('\n>>> LORA ADDED SUCCESSFULLY!')

#training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-2)

model.train()
train_losses, val_losses, val_accs = train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=10)
plot_curves(train_losses, val_losses, val_accs)
save_model(model, "lora_t5_classifier.pth")

exit()









