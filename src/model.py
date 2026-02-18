#!/usr/bin/env python3
"""Phase 5: Attention-Based Model for contact importance prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContactAttentionModel(nn.Module):
    """Attention-based model for predicting binding from per-contact features.

    Architecture (kept tiny for n=15):
    - Contact encoder: Linear(input_dim, 32) + ReLU
    - Multi-head attention: 1 layer, 4 heads, learned query
    - Attention-weighted pooling → prediction head → scalar

    Per-contact features (~15 dims):
    - contact type one-hot (6): CC, CP, CA, PP, PA, AA
    - distance (1)
    - AF3 contact_prob (1)
    - AF3 PAE (1)
    - is_scaffold (1)
    - GTPase region one-hot (3): switch_I, switch_II, other
    - burial (2): res_a_sasa, res_b_sasa (placeholder if not available)
    """

    def __init__(self, input_dim: int = 15, hidden_dim: int = 16,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        # Contact encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Learned query vector for attention
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8, 1),
        )

    def forward(self, contact_features, mask=None):
        """Forward pass.

        Args:
            contact_features: (batch, n_contacts, input_dim)
            mask: (batch, n_contacts) bool tensor, True = padding (to be masked)

        Returns:
            prediction: (batch, 1) binding value
            attention_weights: (batch, n_contacts)
        """
        batch_size = contact_features.shape[0]

        # Encode contacts
        encoded = self.encoder(contact_features)  # (B, N, hidden_dim)

        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)  # (B, 1, hidden_dim)

        # Multi-head attention
        # key_padding_mask: True positions are ignored
        attn_output, attn_weights = self.attention(
            query=query,
            key=encoded,
            value=encoded,
            key_padding_mask=mask,
        )
        # attn_output: (B, 1, hidden_dim)
        # attn_weights: (B, 1, N)

        # Predict
        prediction = self.predictor(attn_output.squeeze(1))  # (B, 1)
        attention = attn_weights.squeeze(1)  # (B, N)

        return prediction, attention

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
