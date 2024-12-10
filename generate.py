import torch
from charLM import Bigram, ModelConfig, create_dataset, MLP
from torch.utils.data import DataLoader
from torch.nn import functional as F

# for future models
# block size condition
# temperature
# top_k condition

@torch.inference_mode
def generate(model, dataset, idx, max_new_tokens, num_samples=10, temperature=1.0, do_sample=True):
    model.eval()
    block_size= model.get_block_size()
    samples = []

    for _ in range(num_samples):
        current_idx = idx.clone()  # Clone the starting index for each sequence
        for _ in range(max_new_tokens):
            
            # print(current_idx)

            idx_cond = current_idx if current_idx.size(1) <= block_size else current_idx[:, -block_size:]

            # print(idx_cond)

            logits, _ = model(idx_cond)

            logits = logits[:, -1, :] / temperature

            probs = F.softmax(logits, dim=-1)

            # print(probs)
            
            if do_sample:
                # Sample from the probability distribution
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                # Select the most probable token
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            
            # take the most likely 
            idx_next = torch.multinomial(probs, num_samples=1)

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
    print(f"\nEvaluation Loss: {mean_loss:.4f}")
    return mean_loss


if __name__ == "__main__":
    # need to load the model
    train_dataset, test_dataset = create_dataset("data/names.txt")

    vocab_size = train_dataset.get_vocab_size()

    idx_init = torch.zeros(1, 1, dtype=torch.long).to('cpu')

    # BIGRAM model

    bigram_config = ModelConfig(vocab_size=vocab_size, block_size=1)

    bigram_model = Bigram(config=bigram_config)

    bigram_model.load_state_dict(torch.load("models/bigram_model.pth", weights_only=True))

    bigram_loss = evaluate(bigram_model, dataset=test_dataset)

    bigram_names = generate(bigram_model, idx=idx_init, dataset=train_dataset, max_new_tokens=20, num_samples=10)

    # MLP Model

    mlp_config = ModelConfig(
        vocab_size=vocab_size,
        block_size=3,  # Set block size for context window
        n_embd=64,
        n_embd2=128,
        model_save_path="models/MLP_model.pth"
    )

    mlp_model = MLP(config=mlp_config)

    # Load pre-trained weights if available
    mlp_model.load_state_dict(torch.load("models/MLP_model.pth", weights_only=True))

    # Evaluate the MLP model
    mlp_loss = evaluate(mlp_model, dataset=test_dataset)

    # Generate sequences using the MLP model
    mlp_names = generate(mlp_model, idx=idx_init, dataset=train_dataset, max_new_tokens=20, num_samples=10)
