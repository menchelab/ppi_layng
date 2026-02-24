"""
Step 4: Train Skip-gram on random walks (Optimized for RTX 5090 / Blackwell)
Inputs: output/walks.npy, output/node_map.parquet.
Outputs: output/embeddings.npy (N x 100).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

from config import (
    WALKS_PARQUET,
    NODE_MAP_PARQUET,
    EMBEDDINGS_PARQUET,
    EMBED_DIM,
    SKIPGRAM_WINDOW,
    SKIPGRAM_EPOCHS,
    SKIPGRAM_BATCH_SIZE,
    SKIPGRAM_NEGATIVE_SAMPLES,
    SKIPGRAM_LEARNING_RATE,
)
from gpu_check import require_torch_cuda

class SkipGram(nn.Module):
    """Skip-gram with negative sampling optimized for GPU throughput."""
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.center_emb = nn.Embedding(vocab_size, embed_dim)
        self.context_emb = nn.Embedding(vocab_size, embed_dim)
        # Initialization scaled for Blackwell tensor cores
        nn.init.xavier_uniform_(self.center_emb.weight)
        nn.init.xavier_uniform_(self.context_emb.weight)

    def forward(self, center, context, neg_context):
        # center: (B), context: (B), neg_context: (B, K)
        v_center = self.center_emb(center)  # (B, D)
        v_context = self.context_emb(context) # (B, D)
        
        # Positive score: dot product
        pos_logits = (v_center * v_context).sum(dim=1) # (B)
        
        # Negative scores: Batch Matrix Multiplication
        neg_vecs = self.context_emb(neg_context) # (B, K, D)
        neg_logits = torch.bmm(v_center.unsqueeze(1), neg_vecs.transpose(1, 2)).squeeze(1) # (B, K)
        
        return pos_logits, neg_logits

def extract_pairs_vectorized(walks_t, window):
    """
    Extracts (center, context) pairs using GPU-based slicing.
    This replaces the slow nested CPU loops.
    """
    centers = []
    contexts = []
    
    # We iterate over the window offset, not the individual nodes
    for i in range(-window, window + 1):
        if i == 0: continue
        
        if i < 0: # Context is to the left
            c = walks_t[:, -i:]
            ctx = walks_t[:, :i]
        else: # Context is to the right
            c = walks_t[:, :-i]
            ctx = walks_t[:, i:]
            
        centers.append(c.reshape(-1))
        contexts.append(ctx.reshape(-1))
        
    return torch.cat(centers), torch.cat(contexts)

def main():
    device = require_torch_cuda()
    
    # 1. Load Data
    npy_path = WALKS_PARQUET.with_suffix(".npy")
    print(f"Loading walks from {npy_path}...")
    walks_np = np.load(npy_path)
    
    # Move to GPU immediately as long integers
    walks_t = torch.from_numpy(walks_np).to(device, dtype=torch.long)
    
    node_map = pd.read_parquet(NODE_MAP_PARQUET)
    vocab_size = len(node_map)
    print(f"Vocab size: {vocab_size} | Device: {torch.cuda.get_device_name(0)}")

    # 2. Vectorized Pair Extraction (Instant on GPU)
    print("Extracting pairs on GPU...")
    centers_t, contexts_t = extract_pairs_vectorized(walks_t, SKIPGRAM_WINDOW)
    num_pairs = centers_t.size(0)
    print(f"Total training pairs: {num_pairs:,}")

    # 3. Model & Optimizer
    model = SkipGram(vocab_size, EMBED_DIM).to(device)
    # Adam handles the large sparse updates of embeddings much better than SGD
    optimizer = torch.optim.Adam(model.parameters(), lr=SKIPGRAM_LEARNING_RATE)
    
    # 4. Training Loop
    num_batches = (num_pairs + SKIPGRAM_BATCH_SIZE - 1) // SKIPGRAM_BATCH_SIZE
    
    for epoch in range(SKIPGRAM_EPOCHS):
        model.train()
        # Shuffle indices on GPU
        perm = torch.randperm(num_pairs, device=device)
        centers_t = centers_t[perm]
        contexts_t = contexts_t[perm]
        
        epoch_loss = 0
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{SKIPGRAM_EPOCHS}")
        
        for b in pbar:
            start = b * SKIPGRAM_BATCH_SIZE
            end = min(start + SKIPGRAM_BATCH_SIZE, num_pairs)
            
            c_batch = centers_t[start:end]
            ctx_batch = contexts_t[start:end]
            
            # Generate negative samples on the fly (saves 10GB+ of VRAM)
            neg_batch = torch.randint(0, vocab_size, (c_batch.size(0), SKIPGRAM_NEGATIVE_SAMPLES), device=device)
            
            optimizer.zero_grad(set_to_none=True) # Set to None is faster for 5090
            
            pos_logits, neg_logits = model(c_batch, ctx_batch, neg_batch)
            
            # Loss = -log(sigmoid(pos)) - sum(log(sigmoid(-neg)))
            pos_loss = F.logsigmoid(pos_logits).neg().mean()
            neg_loss = F.logsigmoid(-neg_logits).neg().mean()
            loss = pos_loss + neg_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if b % 100 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # 5. Export Embeddings
    print("Training complete. Exporting embeddings...")
    model.eval()
    with torch.no_grad():
        # We use the center embeddings as the final node representation
        final_embeddings = model.center_emb.weight.cpu().numpy()

    # Save as .npy for fast loading in downstream steps
    out_path_npy = EMBEDDINGS_PARQUET.with_suffix(".npy")
    np.save(out_path_npy, final_embeddings)
    
    # Save as Parquet for interoperability
    emb_df = pd.DataFrame(final_embeddings, columns=[f"dim_{i}" for i in range(EMBED_DIM)])
    emb_df.to_parquet(EMBEDDINGS_PARQUET, index=False)
    
    print(f"Saved embeddings to {EMBEDDINGS_PARQUET}")
    print("Step 4 complete.")

if __name__ == "__main__":
    main()