import matplotlib.pyplot as plt
import torch

def plot_bigram(dataset):
    stoi = dataset.stoi
    itos = dataset.itos
    vocab_size = len(stoi) + 1

    N = torch.zeros((vocab_size, vocab_size), dtype=torch.long)
    
    for w in dataset.words:
        ch = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(ch, ch[1:]):
            i1 = stoi[ch1]
            i2 = stoi[ch2]
            N[i1, i2] += 1

    plt.figure(figsize=(16, 16))
    plt.imshow(N, cmap='Blues')

    for i in range(vocab_size):
        for j in range(vocab_size):
            chars = itos.get(i, '') + itos.get(j, '')
            plt.text(j, i, chars, ha="center", va="bottom", color="gray")
            plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")

    
    plt.axis('off')
    plt.show()
