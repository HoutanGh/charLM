import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from dataclasses import dataclass



# preparing the data to work on it
class charDataset(Dataset):
    def __init__(self, words, chars, max_word_l):
        self.words = words
        self.chars = chars
        self.max_word_l = max_word_l

        # the mappings of char to index and index to char

        self.stoi = {char: i + 1 for i, char in enumerate(chars)}
        self.itos = {i: char for char, i in self.stoi.items()}

    def __len__(self):
        return len(self.words)
    
    def contains(self, word):
        return word in self.words # boolean?
    
    def get_vocab_size(self):
        return len(self.chars) + 1
    
    def get_output_length(self):
        return self.max_word_l + 1
    
    def encode(self, word):
        idx = torch.tensor([self.stoi[c] for c in word], dtype=torch.long)
        return idx

    def decode(self, idx):
        word = ''.join(self.itos[i] for i in idx)
        return word

    # this is called when using the DataLoader class in pytorch
    def __getitem__(self, idx):
        word = self.words[idx]
        i = self.encode(word)
        x = torch.zeros(self.max_word_l + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_l + 1, dtype=torch.long)
        x[1: 1 + len(i)] = i
        y[:len(i)] = i
        y[len(i) + 1:] = -1

        return x, y
    

# function to prepare the data

def create_dataset(text_file):
    with open(text_file, 'r') as f:
        data = f.read()

    words = data.splitlines()
    words = [w.strip() for w in words] # get rid off leading or trailing spaces
    words = [w for w in words if w] # any empty strings
    chars = sorted(list(set(''.join(words))))
    chars.insert(0, '.')
    max_word_length = max(len(w) for w in words)

    # display results to user

    print(f"number of words in dataset: {len(words)}")
    print(f"max word length: {max_word_length}")
    print(f"number of unique characters: {len(chars)}")
    print(f"vocabulary: {''.join(chars)}")

    # training and testing dataset prep

    test_set_size = int(len(words) * 0.1) # 10% might need to set an upper bound
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]
    print(f"number of training words: {len(train_words)}")
    print(f"number of testing words: {len(test_words)}")


    training_dataset = charDataset(train_words, chars, max_word_length)
    testing_dataset = charDataset(test_words, chars, max_word_length)

    return training_dataset, testing_dataset

# config, scalability for other models
@dataclass
class ModelConfig:
    vocab_size: int = None
    block_size: int = None
    model_save_path: str = "models/model.pth"
    epochs: int = 10
    n_layer: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    n_head: int = 4


# bigram model

class Bigram(nn.Module):
    def __init__(self, config):
        super().__init__()
        n = config.vocab_size
        self.logits = nn.Parameter(torch.zeros((n, n)))
    
    def get_block_size(self):
        return 1 # since model only uses the previous character
        

    def forward(self, idx, targets=None):
        # idx are input character indices, shape (B, T) batch size and sequence length (padded with zeros)
        logits = self.logits[idx]
        loss = None
        if targets is not None:
            # logits.view(-1, logits.size(-1)) reshapes the logits tensor from (B, T, vocab_size) to (B*T, vocab_size)
            # flattens all the time steps and batches into one dimension, making it compatible with PyTorch’s F.cross_entropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            

        return logits, loss

# MLP

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.block_size = config.block_size
        self.vocab_size = config.vocab_size

        # creates table that maps each token to an embedding
        # + 1 for padding
        self.embd = nn.Embedding(config.vocab_size + 1, config.n_embd) 

        self.mlp = nn.Sequential(
            nn.Linear(self.block_size * config.n_embd, config.n_embd2),
            nn.Tanh(),
            nn.Linear(config.n_embd2, self.vocab_size)
        )

    def get_block_size(self):
        return self.block_size  
    
    def forward(self, idx, targets=None):
        embds = []

        for i in range(self.block_size):
            tok_embd = self.embd(idx)
            idx = torch.roll(idx, 1, 1) # permutation of 1
            idx[:, ] = self.vocab_size # replaces the front token, self.vocab_size is a unique index outside of range
            embds.append(tok_embd)

        x = torch.cat(embds, -1) # (B, T, n_embd * block_size)
        logits = self.mlp(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss


        












