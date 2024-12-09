import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from charLM import ModelConfig, create_dataset, charDataset
from charLM import Bigram, MLP
from torch.utils.data import DataLoader

def train_model(model, model_config, dataset_path, model_save_path: None, epochs=10, batch_size=32, lr=0.01):
    train_dataset, test_dataset = create_dataset(dataset_path)

    if os.path.exists(model_config.model_save_path):
        print(f"Loading model weights from {model_config.model_save_path}")
        model.load_state_dict(torch.load(model_config.model_save_path, weights_only=True))
    else:
        print(f"No saved model found at {model_config.model_save_path}, training from scratch.")

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            optimiser.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Training complete. Model saved to {model_save_path}.")

# Example usage
if __name__ == "__main__":
    
    # Load dataset and model configurations
    train_dataset, _ = create_dataset("names.txt")
    vocab_size = train_dataset.get_vocab_size()
    block_size = None # for bigram


    # BIGRAM MODEL

    # bigram_config = ModelConfig(vocab_size=vocab_size, block_size=1)
    # bigram_model = Bigram(config=bigram_config)
    
    # train_model(
    #     model=bigram_model,
    #     model_config=bigram_config,
    #     dataset_path="names.txt",
    #     model_save_path="bigram_model.pth",
    #     epochs=10,
    #     batch_size=32,
    #     lr=0.01
    # )

    # MLP MODEL

    MLP_config = ModelConfig(vocab_size=vocab_size, 
                             block_size=3, n_embd=64, 
                             n_embd2=128, 
                             model_save_path="mlp_model.pth", 
                             epochs=10)
    
    MLP_model = MLP(config=MLP_config)

    train_model(
        model=MLP_model,
        model_config=MLP_config,
        dataset_path="names.txt",
        model_save_path="MLP_model.pth",
        epochs=10,
        batch_size=32,
        lr=0.01
    )

    
