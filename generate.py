import torch
from charLM import Bigram, ModelConfig, create_dataset
from torch.utils.data import DataLoader
from torch.nn import functional as F

# need to load the model
train_dataset, test_dataset = create_dataset("names.txt")

vocab_size = train_dataset.get_vocab_size()
block_size = 1 # for bigram

config = ModelConfig(vocab_size=vocab_size, block_size=block_size)

model = Bigram(config=config)

model.load_state_dict(torch.load("bigram_model.pth", weights_only=True))

# for future models
# block size condition
# temperature
# top_k condition

@torch.inference_mode
def generate(model, dataset, idx, max_new_tokens, num_samples=10):
    model.eval()

    samples = []
    for _ in range(num_samples):
        current_idx = idx.clone()  # Clone the starting index for each sequence
        for _ in range(max_new_tokens):

            # Forward pass to obtain logits
            logits, _ = model(current_idx[:, -1:])


            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample or take the most likely token
            idx_next = torch.multinomial(probs[:, -1, :], num_samples=1)

            # Append the next token to the sequence
            current_idx = torch.cat((current_idx, idx_next), dim=1)

        # Decode the generated sequence and add to the samples list
        sample_idx = current_idx[0, 1:].tolist()  # Skip the initial <START> token
        crop_idx = sample_idx.index(0) if 0 in sample_idx else len(sample_idx)
        decoded_sample = dataset.decode(sample_idx[:crop_idx])
        samples.append(decoded_sample)

    # Print the generated samples
    print("\nGenerated Samples:")
    print("\n".join(samples))

    return samples


@torch.inference_mode
def evaluate(model, dataset, batch_size=50):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, (X, y) in enumerate(loader):
        _, loss = model(X, y)
        losses.append(loss.item())
    
    mean_loss = torch.tensor(losses).mean().item()
    print(f"Evaluation Loss: {mean_loss:.4f}")
    return mean_loss

loss = evaluate(model, dataset=test_dataset)
idx_init = torch.zeros(1, 1, dtype=torch.long).to('cpu')

names = generate(model, idx=idx_init, dataset=train_dataset, max_new_tokens=20, num_samples=10)
