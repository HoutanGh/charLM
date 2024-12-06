import torch
import torch.nn as nn
from torch.nn import functional as F
from charLM import ModelConfig, create_dataset, charDataset
from charLM import Bigram
from torch.utils.data import DataLoader

train_dataset, test_dataset = create_dataset("names.txt")

vocab_size = train_dataset.get_vocab_size()
block_size = 1 # for bigram

config = ModelConfig(vocab_size=vocab_size, block_size=block_size)

model = Bigram(config=config)

model.load_state_dict(torch.load("bigram_model.pth", weights_only=True))

optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

epochs = 10

for epoch in range(epochs):
    total_loss = 0
    for x, y in train_loader:
        optimiser.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{config.epochs}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), config.model_save_path)
print(f"Training complete. Model saved to {config.model_save_path}.")
