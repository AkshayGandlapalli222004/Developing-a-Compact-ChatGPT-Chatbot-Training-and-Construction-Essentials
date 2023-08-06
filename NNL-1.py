import torch
import torch.nn as nn
from torch.nn import functional as F


batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
n_embd=32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
dropout=0.2


torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

arr_chars = sorted(list(set(text)))
vocab_size = len(arr_chars)

drusee = { ch:i for i,ch in enumerate(arr_chars) }
samee = { i:ch for i,ch in enumerate(arr_chars) }
encode = lambda s: [drusee[c] for c in s]
decode = lambda l: ''.join([samee[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.82*len(data))
training_data = data[:n]
valated_data = data[n:]

def get_batch(split):
    data_set = training_data if split == 'train' else valated_data
    ix = torch.randint(len(data_set) - block_size, (batch_size,))
    x = torch.stack([data_set[i:i+block_size] for i in ix])
    y = torch.stack([data_set[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):

    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias= False)
        self.value = nn.Linear(n_embd, head_size, bias= False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))

        self.dropout =nn.Dropout(dropout)

    def forward(self,x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C** -0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0,float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for j in range(num_heads)])

    def forward(self,x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd,n_embd),nn.ReLU())

    def forward(self,x):
          return self.net(x)



class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  #number of embedding directions
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        self.sa_heads = MultiHeadAttention(4,n_embd//4) # 4 communicationn channels so we need 8 dimensional of self attention
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)  #lm_head -- language modelling Head

    def forward(self, idx, targets=None):

        B,T = idx.shape
        token_embd = self.token_embedding_table(idx)
        pos_emd = self.position_embedding_table(torch.arange(T, device= device)) #(T,C)
        x = token_embd + pos_emd
        # To decode the blocks
        #x=self.blocks(x)
        x = self.sa_heads(x)
        x = self.ffwd(x)
        logits = self.lm_head(x)  # sa- Self Attention Head

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_condition = idx[:, -block_size:]
            logits, loss = self(idx_condition)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if(iter % eval_interval == 0):
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))