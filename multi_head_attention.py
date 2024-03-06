import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from typing import Tuple, Optional

class DotProductAttention(nn.Module):
    """
    This class is implementation of Dot-Product Attention mechanism
    proposed in "Effective Approaches to Attention-based Neural Machine Translation"(Luong et al., 2015).
    
    1. Compute attention score by dot-products of the query and values.
    2. Compute attention weights by softmax function on attention score.
    3. Compute attenion value(context vector) by weighted sum of attention weights and values.
    4. Return attention value and attention weights.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        query: Tensor,
        value: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            query:
                Query Embedding tensors.
            value:
                Value Embedding tensors.

        Returns:
            A tuple of attention value and attention weights about query and value.

        Shape:
            query: :math:`(B, N_t, E)` where 
                B is batch size,
                N_t is query length(target sequence length),
                E is embedding dimension.
            value: :math:`(B, N_s, E)` where
                B is batch size,
                N_s is value length(source sequence length),
                E is embedding dimension.
            
            Returns:
                attention value: :math:`(B, N_t, E)`
                attention weights: :math: `(B, N_t, N_s)`
        """
        # (B, N_t, E) x (B, E, N_s) -> (B, N_t, N_s)
        attn_score = torch.bmm(query, value.transpose(-2, -1))
        
        attn_weights = F.softmax(attn_score, dim=-1)
        
        # (B, N_t, N_s) x (B, N_s, E) -> (B, N_t, E)
        attn_value = torch.bmm(attn_weights, value)
        return attn_value, attn_weights


class GeneralAttention(nn.Module):
    """
    This class is implementation of General Attention mechanism
    proposed in "Effective Approaches to Attention-based Neural Machine Translation"(Luong et al., 2015).

    Very similar with DotProductAttention.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: Tensor,
        value: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            query:
                Query Embedding tensors.
            value:
                Value Embedding tensors.

        Returns:
            A tuple of attention value and attention weights about query and value.

        Shape:
            query: :math:`(B, N_t, E)` where 
                B is batch size,
                N_t is the target sequence length,
                E is embedding dimension.
            value: :math:`(B, N_s, E)` where 
                B is batch size,
                N_s is the source sequence length,
                E is embedding dimension.

            Returns:
                attention value: :math:`(B, N_t, E)`
                attention weights: :math:`(B, N_t, N_s)`
        """
        query = self.query_proj(query)
        
        # (B, N_t, E) x (B, E, N_s) -> (B, N_t, N_s)
        attn_score = torch.bmm(query, value.transpose(-2, -1))
        attn_weights = F.softmax(attn_score, dim=-1)
        
        # (B, N_t, N_s) x (B, N_s, E) -> (B, N_t, E)
        attn_value = torch.bmm(attn_weights, value)
        return attn_value, attn_weights


class ScaledDotProductAttention(nn.Module):
    """
    This class is implementation of Scaled Dot-Product Attention mechanism
    proposed in "Attention Is All You Need"(Vaswani et al., 2017).

    Computes scaled dot product attntion on query, key and value tensors.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            query:
                Query Embedding tensors.
            key:
                Key Embedding tensors.
            value:
                Value Embedding tensors.
            attn_mask:
                Optional tensor containing mask values to be added to calculated
                attention.

        Returns:
            A tuple of attention value and attention weights.

        Shape:
            query: :math:`(B, N_t, E)` where 
                B is batch size,
                N_t is the target sequence length,
                E is embedding dimension.
            key: :math:`(B, N_s, E)` where 
                B is batch size,
                N_s is the source sequence length,
                E is embedding dimension.
            value: :math:`(B, N_s, E)` where 
                B is batch size,
                N_s is the source sequence length,
                E is embedding dimension.
            attn_mask: either a 3D tensor of shape :math:`(B, N_t, N_s)` or a 2D tensor of
                shape :math:`(N_t,N_s)` where
                B is batch size,
                N_t is the target sequence length,
                N_s is the source sequence length.

            Returns:
                attention value: :math:`(B, N_t, E)`
                attention weights: :math:`(B, N_t, N_s)`
        """
        embed_dim = query.shape[-1]

        # (B, N_t, E) x (B, E, N_s) -> (B, N_t, N_s)
        attn_score = torch.bmm(query, key.transpose(-2, -1))
        attn_score = attn_score / math.sqrt(embed_dim)
        
        if attn_mask is not None:
            attn_score = torch.baddbmm(attn_mask, query, key.transpose(-2, -1))
        else:
            attn_score = torch.bmm(query, key.transpose(-2, -1))
            
        attn_score = attn_score / math.sqrt(embed_dim)

        attn_weights = F.softmax(attn_score, dim=-1)
        
        # (B, N_t, N_s) x (B, N_s, E) -> (B, N_t, E)
        attn_value = torch.bmm(attn_weights, value)
        
        return attn_value, attn_weights


class MultiHeadAttention(nn.Module):
    """
    This class is implementation of Multi-Head Attention
    proposed in "Attention Is All You Need"(Vaswani et al., 2017).
    
    Allows the modlr to jointly attend to information
    from different representation subspaces.
    
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            query:
                Query Embedding tensors.
            key:
                Key Embedding tensors.
            value:
                Value Embedding tensors.
            attn_mask:
                Optional tensor containing mask values to be added to calculated
                attention.

        Returns:
            A tuple of attention value and attention weights about query, key and value.

        Shape:
            query: :math:`(B, N_t, E)` where 
                B is batch size,
                N_t is the target sequence length,
                E is embedding dimension.
            key: :math:`(B, N_s, E)` where 
                B is batch size,
                N_s is the source sequence length,
                E is embedding dimension.
            value: :math:`(B, N_s, E)` where 
                B is batch size,
                N_s is the source sequence length,
                E is embedding dimension.
            attn_mask: either a 3D tensor of shape :math:`(B, N_t, N_s)` or a 2D tensor of
                shape :math:`(N_t,N_s)` where
                B is batch size,
                N_t is the target sequence length,
                N_s is the source sequence length.

            Returns:
                attention value: :math:`(B, N_t, E)`
                attention weights: :math:`(B, h, N_t, N_s)` where
                    h is a number of attention heads.
        """
        batch_size, target_len, _ = query.shape
        source_len = key.shape[1]
        
        # Compute Linear Projection
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)
        
        # Reshape query, key, value for multihead attention
        query = self._reshape_for_multihead_attn(query)
        key = self._reshape_for_multihead_attn(key)
        value = self._reshape_for_multihead_attn(value)

        # Reshape attention mask
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(self.head_num, 1, 1)
            
        attn_value, attn_weights = ScaledDotProductAttention()(query, key, value, attn_mask)
        
        # Reshape attention value
        # `(B * h, N_t, E // h)` -> `(N_t, B * h, E // h)`
        attn_value = attn_value.transpose(0, 1).contiguous()
        
        # `(N_t, B * h, E // h)` -> (N_t, B, E)`
        attn_value = attn_value.view(target_len, batch_size, self.embed_dim)
        
        # `(N_t, B, E)` -> `(B, N_t, E)`
        attn_value = attn_value.transpose(0, 1)
        
        # Compute out projection(W^O)
        attn_value = self.out_proj(attn_value)
        
        # Reshape attention weights
        #`(B * h, N_t, N_s)` -> `(B, h, N_t, N_s)`
        attn_weights = attn_weights.view(batch_size, self.num_heads, target_len, source_len)
        
        return attn_value, attn_weights
        
        
    def _reshape_for_multihead_attn(
        self,
        emb_tensor: Tensor
    ) -> Tensor:
        """
        Reshape emb_tensor(query, key or value) for multihead attention.
        
        Args:
            emb_tensor:
                The embedding tensor.
                Query, Key or Value Tensor.
            
        Returns:
            Reshaped emb_tensor for multihead attention.
            
        Shape:
            emb_tensor: :math`(B, N, E)` where
                B is batch size,
                N is sequence(target or source) length,
                E is embedding dimension.
            
            Returns: :math`(B * h, N, E // h)` where
                B is batch size,
                h is a number of attention heads,
                N is sequence(target or source) length,
        """
        batch_size, seq_len, _ = emb_tensor.shape
        
        # `(B, N, E)` -> `(N, B, E)`
        reshaped_tensor = emb_tensor.transpose(0, 1).contiguous()
        
        # `(N, B, E)` -> `(N, B * h, E // h)`
        reshaped_tensor = reshaped_tensor.view(seq_len, batch_size * self.num_heads, self.head_dim)
        
        # `(N, B * h, E // h)` -> `(B * h, N, E // h)`
        reshaped_tensor = reshaped_tensor.transpose(0, 1)
        
        return reshaped_tensor