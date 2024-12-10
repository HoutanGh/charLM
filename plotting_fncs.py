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


def plot_emb(model, dataset, figsize=(8,8)):
    itos = dataset.itos
    embd = model.embd.weight
    print(len(itos))

    if embd.shape[0] > len(itos):
            embd = embd[:len(itos)]

    if embd.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        reduced_embd = torch.tensor(pca.fit_transform(embd.detach().cpu().numpy()))
    else:
        reduced_embd = embd

    reduced_embd = reduced_embd.detach().cpu().numpy()


    assert len(itos) == reduced_embd.shape[0], "Mismatch between embeddings and tokens!"


    plt.figure(figsize=figsize)
    plt.scatter(reduced_embd[:, 0], reduced_embd[:, 1], s=200)

    for i in range(reduced_embd.shape[0]):
        token_label = itos[i] if i in itos else "<UNK>"
        plt.text(reduced_embd[i, 0], reduced_embd[i, 1], token_label, ha="center", va="center", color='white', fontsize=10)
        
    
    plt.grid(True, which='minor')
    plt.title("2D Visualization of Token Embeddings")
    plt.show()