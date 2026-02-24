"""
Step 4: Train Skip-gram on random walks (GPU) to produce 100D node embeddings.
Inputs: output/walks.npy (or .parquet), output/node_map.parquet.
Outputs: output/embeddings.parquet / embeddings.npy (N x 100).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from config_tune import (
    WALKS_PARQUET,
    NODE_MAP_PARQUET,
    EMBEDDINGS_PARQUET,
    EMBED_DIM,
    SKIPGRAM_WINDOW,
    SKIPGRAM_EPOCHS,
    SKIPGRAM_BATCH_SIZE,
    SKIPGRAM_NEGATIVE_SAMPLES,
    SKIPGRAM_LEARNING_RATE,
    SKIPGRAM_MAX_PAIRS,
)
from gpu_check import require_torch_cuda


class SkipGram(nn.Module):
    """Skip-gram with negative sampling. All on GPU."""

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.center_emb = nn.Embedding(vocab_size, embed_dim)
        self.context_emb = nn.Embedding(vocab_size, embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.center_emb.weight, -0.5 / self.embed_dim, 0.5 / self.embed_dim)
        nn.init.uniform_(self.context_emb.weight, -0.5 / self.embed_dim, 0.5 / self.embed_dim)

    def forward(self, center, context, neg_context):
        # center, context: (B,); neg_context: (B, K)
        v_center = self.center_emb(center)  # (B, D)
        v_context = self.context_emb(context)  # (B, D)
        pos_logits = (v_center * v_context).sum(dim=1)  # (B,)
        neg_vecs = self.context_emb(neg_context)  # (B, K, D)
        neg_logits = torch.bmm(v_center.unsqueeze(1), neg_vecs.transpose(1, 2)).squeeze(1)  # (B, K)
        return pos_logits, neg_logits


def extract_pairs(walks, window):
    """From walks (num_walks, walk_len), extract (center, context) pairs."""
    centers = []
    contexts = []
    for i in tqdm(range(walks.shape[0]), desc="Extracting (center, context) pairs", unit=" walks"):
        for j in range(walks.shape[1]):
            center = walks[i, j]
            for k in range(max(0, j - window), min(walks.shape[1], j + window + 1)):
                if k != j:
                    centers.append(center)
                    contexts.append(walks[i, k])
    return np.array(centers, dtype=np.int64), np.array(contexts, dtype=np.int64)


def main():
    device = require_torch_cuda()

    # Load walks
    npy_path = WALKS_PARQUET.with_suffix(".npy")
    if npy_path.exists():
        walks = np.load(npy_path)
    else:
        df = pd.read_parquet(WALKS_PARQUET)
        cols = [c for c in df.columns if c.startswith("step_")]
        walks = df[cols].values.astype(np.int32)
    print(f"Walks shape: {walks.shape}")

    node_map = pd.read_parquet(NODE_MAP_PARQUET)
    vocab_size = len(node_map)
    print(f"Vocabulary size: {vocab_size}")

    # Extract (center, context) pairs
    centers, contexts = extract_pairs(walks, SKIPGRAM_WINDOW)
    num_pairs = len(centers)
    print(f"Training pairs: {num_pairs}")

    rng = np.random.default_rng(42)
    if SKIPGRAM_MAX_PAIRS is not None and num_pairs > SKIPGRAM_MAX_PAIRS:
        idx = rng.choice(num_pairs, size=SKIPGRAM_MAX_PAIRS, replace=False)
        centers = centers[idx]
        contexts = contexts[idx]
        num_pairs = SKIPGRAM_MAX_PAIRS
        print(f"Subsampled to {num_pairs} pairs")

    # Negative sampling: sample random node ids
    neg_contexts = rng.integers(0, vocab_size, size=(num_pairs, SKIPGRAM_NEGATIVE_SAMPLES))

    # To GPU tensors
    centers_t = torch.from_numpy(centers).to(device)
    contexts_t = torch.from_numpy(contexts).to(device)
    neg_contexts_t = torch.from_numpy(neg_contexts).to(device)

    model = SkipGram(vocab_size, EMBED_DIM).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=SKIPGRAM_LEARNING_RATE)

    # Training loop (batched)
    num_batches = (num_pairs + SKIPGRAM_BATCH_SIZE - 1) // SKIPGRAM_BATCH_SIZE
    for epoch in tqdm(range(SKIPGRAM_EPOCHS), desc="Skip-gram epochs", unit=" epoch"):
        model.train()
        epoch_loss = 0.0
        perm = rng.permutation(num_pairs)
        batch_iter = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}", leave=False, unit=" batch")
        for b in batch_iter:
            start = b * SKIPGRAM_BATCH_SIZE
            end = min(start + SKIPGRAM_BATCH_SIZE, num_pairs)
            idx = perm[start:end]
            c = centers_t[idx]
            ctx = contexts_t[idx]
            neg = neg_contexts_t[idx]
            pos_logits, neg_logits = model(c, ctx, neg)
            pos_loss = torch.nn.functional.logsigmoid(pos_logits).neg().mean()
            neg_loss = torch.nn.functional.logsigmoid(-neg_logits).neg().mean()
            loss = pos_loss + neg_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            batch_iter.set_postfix(loss=f"{loss.item():.4f}")

    # Extract embedding matrix: use center embedding for each node
    model.eval()
    with torch.no_grad():
        all_ids = torch.arange(vocab_size, device=device, dtype=torch.long)
        emb = model.center_emb(all_ids).cpu().numpy()
    print(f"Embeddings shape: {emb.shape}")

    # Save
    np.save(EMBEDDINGS_PARQUET.with_suffix(".npy"), emb)
    emb_df = pd.DataFrame(emb, columns=[f"dim_{i}" for i in range(EMBED_DIM)])
    emb_df.to_parquet(EMBEDDINGS_PARQUET, index=False)
    print(f"Saved {EMBEDDINGS_PARQUET} and .npy")
    print("Step 4 done.")


if __name__ == "__main__":
    main()
