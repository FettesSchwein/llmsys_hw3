import numpy as np
from .tensor import tensor, tensor_from_numpy
from .module import Module, Parameter
from .modules_basic import (
    Embedding,
    Dropout,
    LayerNorm1d,
    Linear
)
from .tensor_ops import TensorBackend
from .nn import (
    max,
    softmax,
    dropout,
    GELU,
)
from typing import Any, Dict, Optional, Sequence, Tuple

datatype = np.float32


class MultiHeadAttention(Module):
    def __init__(self, n_embd: int, n_head: int, causal: bool=True, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """Implements Multi-Head Attention as described in "Attention Is All You Need"

        Args:
            n_embd: Dimensionality of embeddings and hidden states
            n_head: Number of heads
            p_dropout: Dropout ratio for dropout layer
            causal: If True, then apply a causal mask during self-attention
            bias: If True, then apply a bias in Linear layers
        
        Attributes:
            q_projection: Linear layer projecting input to Q matrix
            k_projection: Linear layer projecting input to K matrix
            v_projection: Linear layer projecting input to V matrix
            out_projection: Linear output projection layer
            dropout: Dropout layer
        """
        self.backend = backend
        self.n_embd = n_embd 
        self.n_head = n_head
        self.causal = causal
        self.attn_hidden_dim = n_embd // n_head

        ### BEGIN ASSIGN3_3
        self.q_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.k_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.v_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.out_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.dropout = Dropout(p_dropout)
        ### END ASSIGN3_3

    def create_causal_mask(self, seq_len):
        """
        Create a causal mask for self-attention to prevent information leakage.
        
        Generates a triangular mask where each position can only attend to previous
        positions and itself. Upper triangle contains -inf, lower triangle contains 0.

        Args:
            seq_len (int): Length of the sequence

        Returns:
            Tensor: Causal mask of shape (1, 1, seq_len, seq_len) with -inf above
                    diagonal and 0 on/below diagonal. Will be broadcasted to full
                    attention tensor shape during computation.
        """
        # Returns a 1x1xTxt triangular causal mask for Q @ K^T (You will implicitly broadcast it to BxHxTxT)
        mask = np.triu(np.ones((1, 1, seq_len, seq_len), dtype=datatype), k=1)
        mask = np.where(mask == 1, -np.inf, 0.0).astype(datatype)
        return tensor_from_numpy(mask, backend=self.backend)

    def project_to_query_key_value(self, x):
        """
        Project input embeddings to Query, Key, and Value matrices for self-attention.
        
        Args:
            x (Tensor): Input embeddings of shape (batch_size, seq_len, n_embd)

        Returns:
            tuple: (q, kT, v) where:
                - q: Query matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
                - kT: Transposed key matrix of shape (batch_size, num_heads, attn_hidden_dim, seq_len)
                - v: Value matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN ASSIGN3_3
        # 1. Project to Q, K, V -> Shape: (batch, seq, n_embd)
        # Note: We flatten the batch and sequence dimensions for the Linear layer input
        x_flat = x.view(batch_size * seq_len, n_embd)
        
        q = self.q_projection(x_flat).view(batch_size, seq_len, n_embd)
        k = self.k_projection(x_flat).view(batch_size, seq_len, n_embd)
        v = self.v_projection(x_flat).view(batch_size, seq_len, n_embd)

        # 2. Reshape to Heads -> Shape: (batch, seq, n_head, attn_hidden_dim)
        q = q.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim)
        k = k.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim)
        v = v.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim)

        # 3. Permute for Attention -> 
        # q, v shape: (batch, n_head, seq, attn_hidden_dim)
        q = q.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # k transposed shape: (batch, n_head, attn_hidden_dim, seq)
        # Do this in a single permute to get proper gradient flow
        kT = k.permute(0, 2, 3, 1)
        ### END ASSIGN3_3
        return q, kT, v
    
    def self_attention(self, q, kT, v):
        """
        Compute self-attention: softmax((q @ kT) / sqrt(attn_hidden_dim)) @ v.
        
        Args:
            q (Tensor): Query matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
            kT (Tensor): Transposed key matrix of shape (batch_size, num_heads, attn_hidden_dim, seq_len)
            v (Tensor): Value matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)

        Returns:
            Tensor: Attention output of shape (batch_size, seq_len, n_embd)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, k_dim, _ = kT.shape
        _, _, _, v_dim = v.shape
        assert q_dim == k_dim == v_dim
        result = None
        
        ### BEGIN ASSIGN3_3
        # 1. Scaled Dot-Product Attention Scores
        # (batch, head, seq, dim) @ (batch, head, dim, seq) -> (batch, head, seq, seq)
        scores = (q @ kT) / np.sqrt(q_dim)

        # 2. Apply Causal Mask
        if self.causal:
            mask = self.create_causal_mask(queries_len)
            scores = scores + mask

        # 3. Softmax and Dropout
        attn_weights = softmax(scores, dim=3)
        attn_weights = self.dropout(attn_weights)

        # 4. Weighted Sum
        # (batch, head, seq, seq) @ (batch, head, seq, dim) -> (batch, head, seq, dim)
        attn_output = attn_weights @ v

        # 5. Reshape and Concatenate Heads
        # Permute back to (batch, seq, head, dim)
        attn_output = attn_output.permute(0, 2, 1, 3)
        # Reshape to (batch, seq, n_embd)
        result = attn_output.contiguous().view(batch_size, queries_len, self.n_embd)
        ### END ASSIGN3_3

        return result

    def forward(self, x):
        """
        Compute multi-head attention with optional causal masking.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN ASSIGN3_3
        q, kT, v = self.project_to_query_key_value(x)
        attn_out = self.self_attention(q, kT, v)
        
        # Final output projection
        # View as 2D for linear layer then reshape back
        attn_out_flat = attn_out.contiguous().view(batch_size * seq_len, n_embd)
        output = self.out_projection(attn_out_flat)
        return output.view(batch_size, seq_len, n_embd)
        ### END ASSIGN3_3


class FeedForward(Module):
    def __init__(self, n_embd: int, middle_dim: int=256, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """
        Initialize a feed-forward network module.
        
        Args:
            n_embd (int): Input and output dimension
            middle_dim (int): Hidden layer dimension, default 256
            p_dropout (float): Dropout probability, default 0.1
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            linear_in (Linear): First linear layer
            linear_out (Linear): Second linear layer
            dropout (Dropout): Dropout layer
        """
        ### BEGIN ASSIGN3_3
        self.linear_in  = Linear(n_embd, middle_dim, bias=bias, backend=backend)
        self.linear_out = Linear(middle_dim, n_embd, bias=bias, backend=backend)
        self.dropout    = Dropout(p_dropout)
        ### END ASSIGN3_3

    def forward(self, x):
        """
        Forward pass through feed-forward network with GELU activation and dropout.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape

        ### BEGIN ASSIGN3_3
        # Flatten for Linear layer: (batch * seq, n_embd)
        x_flat = x.view(batch_size * seq_len, n_embd)
        
        # Linear -> GELU
        hidden = GELU(self.linear_in(x_flat))
        
        # Linear -> Dropout
        out = self.dropout(self.linear_out(hidden))
        
        # Reshape back to (batch, seq, n_embd)
        return out.view(batch_size, seq_len, n_embd)
        ### END ASSIGN3_3
    

class TransformerLayer(Module):
    def __init__(self, n_embd: int, n_head: int, p_dropout: float=0.1, ln_eps: float=1e-5, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """
        Initialize a transformer layer with pre-layer normalization.
        
        Args:
            n_embd (int): Embedding dimension
            n_head (int): Number of attention heads
            p_dropout (float): Dropout probability, default 0.1
            ln_eps (float): Layer normalization epsilon, default 1e-5
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            ln_1 (LayerNorm1d): First layer normalization before attention
            ln_2 (LayerNorm1d): Second layer normalization after attention
            attention (MultiHeadAttention): Multi-head attention layer
            ff (FeedForward): Feed-forward network layer
        """
        ### BEGIN ASSIGN3_3
        self.ln_1 = LayerNorm1d(n_embd, eps=ln_eps, backend=backend)
        self.ln_2 = LayerNorm1d(n_embd, eps=ln_eps, backend=backend)
        self.attention = MultiHeadAttention(n_embd, n_head, p_dropout=p_dropout, bias=bias, backend=backend)
        self.ff = FeedForward(n_embd, p_dropout=p_dropout, bias=bias, backend=backend)
        ### END ASSIGN3_3

    def forward(self, x):
        """
        Forward pass through transformer layer with pre-layer normalization.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)
        
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN ASSIGN3_3
        # 1. Layer Norm -> Attention -> Residual Connection (PRE-LN)
        x_norm = self.ln_1(x.contiguous().view(batch_size * seq_len, n_embd))
        x = x + self.attention(x_norm.contiguous().view(batch_size, seq_len, n_embd))
        
        # 2. Layer Norm -> FeedForward -> Residual Connection (PRE-LN)
        x_norm = self.ln_2(x.contiguous().view(batch_size * seq_len, n_embd))
        x = x + self.ff(x_norm.contiguous().view(batch_size, seq_len, n_embd))
        return x
        ### END ASSIGN3_3


class DecoderLM(Module):
    def __init__(
        self, 
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        p_dropout: float=0.1,
        ln_eps: float=1e-5, 
        bias: bool=True,
        backend: TensorBackend=None
    ):
        super().__init__()
        """
        Initialize a decoder-only transformer language model.
        
        Args:
            n_vocab (int): Vocabulary size
            n_embd (int): Embedding dimension
            n_head (int): Number of attention heads
            n_positions (int): Maximum sequence length
            p_dropout (float): Dropout probability, default 0.1
            ln_eps (float): Layer normalization epsilon, default 1e-5
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            token_embeddings (Embedding): Token embedding layer
            position_embeddings (Embedding): Position embedding layer
            t_layer_1 (TransformerLayer): First transformer layer
            t_layer_2 (TransformerLayer): Second transformer layer
            t_layer_3 (TransformerLayer): Third transformer layer
            t_layer_4 (TransformerLayer): Fourth transformer layer
            dropout (Dropout): Dropout layer before transformer layers
            ln (LayerNorm1d): Final layer normalization
            lm_head (Linear): Language model head for vocabulary projection
        """
        self.backend = backend
        self.n_embd = n_embd
        self.n_vocab = n_vocab
        ### BEGIN ASSIGN3_3
        self.token_embeddings = Embedding(n_vocab, n_embd, backend=backend)
        self.position_embeddings = Embedding(n_positions, n_embd, backend=backend)
        
        self.t_layer_1 = TransformerLayer(n_embd, n_head, p_dropout=p_dropout, ln_eps=ln_eps, bias=bias, backend=backend)
        self.t_layer_2 = TransformerLayer(n_embd, n_head, p_dropout=p_dropout, ln_eps=ln_eps, bias=bias, backend=backend)
        self.t_layer_3 = TransformerLayer(n_embd, n_head, p_dropout=p_dropout, ln_eps=ln_eps, bias=bias, backend=backend)
        self.t_layer_4 = TransformerLayer(n_embd, n_head, p_dropout=p_dropout, ln_eps=ln_eps, bias=bias, backend=backend)
        
        self.dropout = Dropout(p_dropout)
        self.ln = LayerNorm1d(n_embd, eps=ln_eps, backend=backend)
        self.lm_head = Linear(n_embd, n_vocab, bias=bias, backend=backend)
        ### END ASSIGN3_3
    
    def forward(self, idx):
        """
        Forward pass through decoder-only transformer language model.
        
        Args:
            idx (Tensor): Input token indices of shape (batch_size, seq_len)
        
        Returns:
            Tensor: Logits of shape (batch_size, seq_len, n_vocab)
        """
        
        batch_size, seq_len = idx.shape

        ### BEGIN ASSIGN3_3
        # 1. Token embeddings
        tok_emb = self.token_embeddings(idx) # (batch, seq, n_embd)
        
        # 2. Positional embeddings
        # Create position indices [0, 1, ..., seq_len-1]
        # We need to create a tensor manually since we don't have arange in minitorch yet
        # Using numpy to create indices and converting to tensor
        pos_indices_np = np.arange(seq_len)
        # Add batch dimension to match embedding requirement or broadcast: (1, seq)
        pos_indices = tensor_from_numpy(pos_indices_np, backend=self.backend).view(1, seq_len)
        
        pos_emb = self.position_embeddings(pos_indices) # (1, seq, n_embd)
        
        # 3. Add embeddings
        x = tok_emb + pos_emb
        
        # 4. Dropout
        x = self.dropout(x)
        
        # 5. Transformer layers
        x = self.t_layer_1(x)
        x = self.t_layer_2(x)
        x = self.t_layer_3(x)
        x = self.t_layer_4(x)
        
        # 6. Final Layer Norm
        # Reshape for LN: (batch * seq, n_embd)
        x_flat = x.view(batch_size * seq_len, self.n_embd)
        x_norm = self.ln(x_flat)
        
        # 7. Project to vocab size
        logits = self.lm_head(x_norm)
        
        # Reshape back to (batch, seq, n_vocab)
        return logits.view(batch_size, seq_len, self.n_vocab)
        ### END ASSIGN3_3