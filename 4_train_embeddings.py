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


def extract_pairs_vectorized(walks_t: torch.Tensor, window: int):
    """
    Vectorized extraction of (center, context) pairs from walks tensor.
    walks_t: (num_walks, walk_len) long tensor on GPU/CPU.
    Keeps only valid tokens (>=0) to avoid padding contamination.
    """
    centers = []
    contexts = []
    for offset in range(-window, window + 1):
        if offset == 0:
            continue
        if offset < 0:
            c = walks_t[:, -offset:]
            ctx = walks_t[:, :offset]
        else:
            c = walks_t[:, :-offset]
            ctx = walks_t[:, offset:]
        centers.append(c.reshape(-1))
        contexts.append(ctx.reshape(-1))
    centers_t = torch.cat(centers)
    contexts_t = torch.cat(contexts)
    valid = (centers_t >= 0) & (contexts_t >= 0)
    return centers_t[valid], contexts_t[valid]


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

    # Extract (center, context) pairs (vectorized)
    walks_t = torch.from_numpy(walks).to(device=device, dtype=torch.long)
    centers_t, contexts_t = extract_pairs_vectorized(walks_t, SKIPGRAM_WINDOW)
    del walks_t
    num_pairs = int(centers_t.numel())
    print(f"Training pairs: {num_pairs}")

    # Subsample to cap early to reduce compute/memory.
    # Uses sampling with replacement to avoid huge randperm allocations for 1B+ pairs.
    if SKIPGRAM_MAX_PAIRS is not None and num_pairs > SKIPGRAM_MAX_PAIRS:
        idx = torch.randint(0, num_pairs, (SKIPGRAM_MAX_PAIRS,), device=device)
        centers_t = centers_t[idx]
        contexts_t = contexts_t[idx]
        num_pairs = int(SKIPGRAM_MAX_PAIRS)
        print(f"Subsampled to {num_pairs} pairs")

    model = SkipGram(vocab_size, EMBED_DIM).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=SKIPGRAM_LEARNING_RATE)

    # Training loop (batched)
    num_batches = (num_pairs + SKIPGRAM_BATCH_SIZE - 1) // SKIPGRAM_BATCH_SIZE
    for epoch in tqdm(range(SKIPGRAM_EPOCHS), desc="Skip-gram epochs", unit=" epoch"):
        model.train()
        epoch_loss = 0.0
        perm = torch.randperm(num_pairs, device=device)
        batch_iter = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}", leave=False, unit=" batch")
        for b in batch_iter:
            start = b * SKIPGRAM_BATCH_SIZE
            end = min(start + SKIPGRAM_BATCH_SIZE, num_pairs)
            idx = perm[start:end]
            c = centers_t[idx]
            ctx = contexts_t[idx]
            neg = torch.randint(0, vocab_size, (c.size(0), SKIPGRAM_NEGATIVE_SAMPLES), device=device)
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
