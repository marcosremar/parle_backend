"""
Multi-Embedding Fusion
Combines multiple Wav2Vec embeddings using multi-head attention.
Based on Lee et al. (Interspeech 2024).
"""

import torch
import torch.nn as nn
from typing import Tuple


class MultiEmbeddingFusion(nn.Module):
    """
    Combine multiple embeddings using multi-head attention.
    
    Based on Lee et al. (Interspeech 2024):
    - Uses multi-head attention to learn optimal combination
    - Combines native + learner embeddings
    - Can be extended to include phoneme embeddings
    
    Architecture:
    1. Stack embeddings (native, learner)
    2. Apply multi-head attention
    3. Layer normalization + residual
    4. Temporal pooling (mean)
    5. Concatenate final embeddings
    """
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 8):
        """
        Initialize fusion module.
        
        Args:
            embed_dim: Embedding dimension (768 for Wav2Vec Large)
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=False  # Wav2Vec outputs (seq_len, batch, embed_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Optional: Projection layer for final output
        self.projection = nn.Linear(embed_dim * 2, embed_dim)
    
    def forward(
        self, 
        native_emb: torch.Tensor,
        learner_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse native and learner embeddings.
        
        Args:
            native_emb: Native embedding (seq_len, batch, embed_dim)
            learner_emb: Learner embedding (seq_len, batch, embed_dim)
            
        Returns:
            Fused embedding (batch, embed_dim * 2) or (batch, embed_dim)
        """
        # Ensure same sequence length
        min_seq_len = min(native_emb.size(0), learner_emb.size(0))
        native_emb = native_emb[:min_seq_len]
        learner_emb = learner_emb[:min_seq_len]
        
        # Stack embeddings: (2, seq_len, batch, embed_dim)
        combined = torch.stack([native_emb, learner_emb], dim=0)
        # Reshape to: (2, seq_len, batch, embed_dim)
        # But attention expects (seq_len, batch, embed_dim) for each
        # So we need to process them together
        
        # Reshape for attention: (2 * seq_len, batch, embed_dim)
        batch_size = native_emb.size(1)
        combined_flat = combined.view(-1, batch_size, self.embed_dim)
        
        # Multi-head attention
        # Query, Key, Value all from combined embeddings
        attended, attention_weights = self.attention(
            query=combined_flat,
            key=combined_flat,
            value=combined_flat
        )
        
        # Layer norm + residual
        output = self.layer_norm(attended + combined_flat)
        
        # Reshape back: (2, seq_len, batch, embed_dim)
        output = output.view(2, min_seq_len, batch_size, self.embed_dim)
        
        # Temporal pooling (mean over sequence length)
        pooled = output.mean(dim=1)  # (2, batch, embed_dim)
        
        # Concatenate the two embeddings
        fused = torch.cat([pooled[0], pooled[1]], dim=-1)  # (batch, embed_dim * 2)
        
        # Optional: Project to original dimension
        if self.projection is not None:
            fused = self.projection(fused)  # (batch, embed_dim)
        
        return fused
    
    def forward_with_phoneme(
        self,
        native_emb: torch.Tensor,
        learner_emb: torch.Tensor,
        phoneme_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse native, learner, and phoneme embeddings.
        
        Args:
            native_emb: Native embedding (seq_len, batch, embed_dim)
            learner_emb: Learner embedding (seq_len, batch, embed_dim)
            phoneme_emb: Phoneme embedding (seq_len, batch, embed_dim)
            
        Returns:
            Fused embedding
        """
        # Stack all three embeddings
        min_seq_len = min(
            native_emb.size(0),
            learner_emb.size(0),
            phoneme_emb.size(0)
        )
        
        combined = torch.stack([
            native_emb[:min_seq_len],
            learner_emb[:min_seq_len],
            phoneme_emb[:min_seq_len]
        ], dim=0)
        
        # Process similar to forward()
        batch_size = native_emb.size(1)
        combined_flat = combined.view(-1, batch_size, self.embed_dim)
        
        attended, _ = self.attention(
            query=combined_flat,
            key=combined_flat,
            value=combined_flat
        )
        
        output = self.layer_norm(attended + combined_flat)
        output = output.view(3, min_seq_len, batch_size, self.embed_dim)
        
        pooled = output.mean(dim=1)  # (3, batch, embed_dim)
        
        # Concatenate all three
        fused = torch.cat([pooled[0], pooled[1], pooled[2]], dim=-1)
        
        return fused
