import torch
import torch.nn as nn
from torch.utils.data import Dataset



# need to be able to work with the data first 
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
        return self.max_world_l + 1
    
    def encode(self, word):
        idx = torch.tensor([self.stoi[c] for c in word], dtype=torch.long)
        return idx

    def decode(self, idx):
        word = ''.join(self.itos[i] for i in idx)
        return word

    def __getitem__(self, idx):
        word = self.words(idx)
        i = self.encode(word)
        x = torch.zeros(self.max_world_l + 1, dtype=torch.long)
        y = torch.zeros(self.max_world_l + 1, dtype=torch.long)
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













