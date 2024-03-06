import pytest
import torch

from multi_head_attention import (
    DotProductAttention,
    GeneralAttention,
    ScaledDotProductAttention,
    MultiHeadAttention
    )

def attention_test():
    # Prepare dummpy data.
    print(f"Prepare dummy data.")
    
    # Make dummy data.
    B = 3
    N_t = 16
    N_s = 20
    E = 384
    num_heads = 8
    attn_mask = None
    query = torch.randn(B, N_t, E)
    key = torch.randn(B, N_s, E)
    value = torch.randn(B, N_s, E)

    print(f"Start Attention test.")
    attentions = ["dot", "general", "scaled", "multihead"]

    for type_attention in attentions:
        # Compute attention value and weights.
        if type_attention == "dot":
            attn = DotProductAttention()
            attn_value, attn_weights = attn(query, value)
        
        elif type_attention == "general":
            attn = GeneralAttention(E)
            attn_value, attn_weights = attn(query, value)

        elif type_attention == "scaled":
            attn = ScaledDotProductAttention()
            attn_value, attn_weights = attn(query, key, value, attn_mask)
        
        elif type_attention == "multihead":
            attn = MultiHeadAttention(E, num_heads)
            attn_value, attn_weights = attn(query, key, value, attn_mask)

        # Check attention value and weights.
        assert attn_value.shape == (B, N_t, E)
        assert attn_weights.shape == (B, N_t, N_s) or attn_weights.shape == (B, num_heads, N_t, N_s)
    
        if type_attention == "multihead":
            for i in range(B):
                for j in range(num_heads):
                    for k in range(N_t):
                        assert round(float(attn_weights[i][j][k].sum()), 0) == 1.0
        else:
            for i in range(B):
                for j in range(N_t):
                    assert round(float(attn_weights[i][j].sum()), 0) == 1.0
    print("Test successfully")

if __name__ == "__main__":
    attention_test()