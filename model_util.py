    
import math
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
import torch.nn.functional as F

def masked_softmax(X, valid_lens):
    """Performs softmax operation by masking elements on the last axis"""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # Masked elements on the last axis are replaced with a very large negative value, making their softmax output 0
        X = X.reshape(-1, shape[-1])
        mask = torch.arange(shape[-1], device=X.device)[None, :] < valid_lens[:, None]
        X[~mask] = -1e6  # Use a very large negative value
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class AdditiveAttention(nn.Module):
    """Additive Attention"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        """
        Args:
            queries: (batch_size, num_queries, query_size)
            keys: (batch_size, num_key_value_pairs, key_size)
            values: (batch_size, num_key_value_pairs, value_size)
            valid_lens: (batch_size,) or (batch_size, num_queries)

        Returns:
            output: (batch_size, num_queries, value_size)
            attention_weights: (batch_size, num_queries, num_key_value_pairs)
        """
        queries = self.W_q(queries)  # (batch_size, num_queries, num_hiddens)
        keys = self.W_k(keys)        # (batch_size, num_key_value_pairs, num_hiddens)
        # Expand queries and keys to compute attention scores
        # queries shape: (batch_size, num_queries, 1, num_hiddens)
        # keys shape: (batch_size, 1, num_key_value_pairs, num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # Compute attention scores
        scores = self.w_v(features).squeeze(-1)  # (batch_size, num_queries, num_key_value_pairs)
        # Apply masked softmax
        attention_weights = masked_softmax(scores, valid_lens)
        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)
        # Compute the context vector
        output = torch.bmm(attention_weights, values)  # (batch_size, num_queries, value_size)
        return output, attention_weights

    
class DotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries shape: (batch_size, number of queries, d)
    # keys shape: (batch_size, number of key-value pairs, d)
    # values shape: (batch_size, number of key-value pairs, value dimension)
    # valid_lens shape: (batch_size,) or (batch_size, number of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Compute attention scores
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        # Compute attention weights
        attention_weights = masked_softmax(scores, valid_lens)
        # Weighted sum of values
        output = torch.bmm(self.dropout(attention_weights), values)
        return output, attention_weights

class HierarchicalGatedExpertNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model

        self.alpha_mlp = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )

        # Also define a multi-layer MLP for calculating beta
        self.beta_mlp = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )

        # Define a linear layer for calculating gating weights
        self.gate_layer = AdditiveAttention(self.d_model, self.d_model, self.d_model, dropout=0.1)

    def forward(self, style_long_repr, interest_long_repr, style_short_repr, interest_short_repr, article_repr):
        """
        Args:
            style_long_repr: (batch_size, d_model)
            interest_long_repr: (batch_size, d_model)
            style_short_repr: (batch_size, d_model)
            interest_short_repr: (batch_size, d_model)
            article_repr: (batch_size, d_model)
        
        Returns:
            user_feature: (batch_size, d_model)
        """

        # Concatenate long and short term interest representations with article representation as input for alpha_mlp
        interest_inputs = torch.cat([interest_long_repr, interest_short_repr, article_repr], dim=-1)  # (batch_size, d_model * 3)
        alpha = self.alpha_mlp(interest_inputs)  # (batch_size, 1)

        # Calculate interest representation
        interest_repr = alpha * interest_long_repr + (1 - alpha) * interest_short_repr  # (batch_size, d_model)
        

        # Concatenate long and short term style representations with article representation as input for beta_mlp
        style_inputs = torch.cat([style_long_repr, style_short_repr, article_repr], dim=-1)  # (batch_size, d_model * 3)
        beta = self.beta_mlp(style_inputs)  # (batch_size, 1)

        # Calculate style representation
        style_repr = beta * style_long_repr + (1 - beta) * style_short_repr  # (batch_size, d_model)

        # Calculate gating weights
        # article_repr as query, combine style and interest representations as key-value pairs, calculate attention gating weights
        gate_inputs = torch.stack([style_repr, interest_repr], dim=1)
        user_feature = self.gate_layer(article_repr.unsqueeze(1), gate_inputs, gate_inputs)[0].squeeze(1)  # (batch_size, d_model)

        return user_feature

class PoolingMethod(nn.Module):
    def __init__(self, config, method='mean'):
        super().__init__()
        self.method = method
        if method == 'attention':
            # Attention pooling: using two linear transformations
            self.attn_fc1 = nn.Linear(config.d_model, config.d_model // 2)  # First linear transformation
            self.attn_fc2 = nn.Linear(config.d_model // 2, 1)  # Second linear transformation
            self.init_weights()
        elif method == 'query':
            # Query-based attention pooling
            self.attention = AdditiveAttention(config.d_model, config.d_model, config.d_model, dropout=0.1)
        elif method == 'self_voting':
            # Self-voting attention pooling
            pass # No specific layers needed for self-voting in __init__

    def init_weights(self):
        if self.method == 'attention':
            nn.init.xavier_uniform_(self.attn_fc1.weight)
            nn.init.xavier_uniform_(self.attn_fc2.weight)
            if self.attn_fc1.bias is not None:
                nn.init.zeros_(self.attn_fc1.bias)
            if self.attn_fc2.bias is not None:
                nn.init.zeros_(self.attn_fc2.bias)

    def forward(self, hidden_states, attention_mask=None, query=None):
        """
        Args:
            hidden_states: (batch_size, seq_length, d_model)
            attention_mask: (batch_size, seq_length) or None
            query: (batch_size, d_model) or None
        Returns:
            pooled_output: (batch_size, d_model)
        """
        if self.method == 'mean':
            # Mean pooling
            if attention_mask is not None:
                attention_mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch_size, seq_length, 1)
                mask_sum = attention_mask_expanded.sum(dim=1) + 1e-8  # Prevent division by zero
                pooled_output = torch.sum(hidden_states * attention_mask_expanded, dim=1) / mask_sum
            else:
                pooled_output = hidden_states.mean(dim=1)

        elif self.method == 'attention':
            # Two-layer linear attention pooling
            e = self.attn_fc1(hidden_states)  # First linear transformation (batch_size, seq_length, hidden_dim // 2)
            e = torch.tanh(e)  # Non-linear activation function (batch_size, seq_length, hidden_dim // 2)
            alpha = self.attn_fc2(e)  # Second linear transformation, compute attention scores (batch_size, seq_length, 1)

            # Apply mask to alpha if attention mask is provided
            if attention_mask is not None:
                alpha = alpha * attention_mask.unsqueeze(2)

            # Normalize attention weights using softmax
            attention_weights = F.softmax(alpha, dim=1)  # (batch_size, seq_length, 1)

            # Weighted sum
            pooled_output = torch.bmm(hidden_states.permute(0, 2, 1), attention_weights).squeeze(2)  # (batch_size, d_model)        
        
        elif self.method == 'query':
            # Query-based attention pooling
            assert query is not None, "query pooling requires a query tensor."
            
            # Expand query to (batch_size, 1, d_model)
            query = query.unsqueeze(1)
            # Attention mask handling
            if attention_mask is not None:
                # Calculate valid lengths
                valid_lens = attention_mask.sum(dim=1)
            else:
                valid_lens = None
            # Use AdditiveAttention for query-based pooling
            attention_output, _ = self.attention(query, hidden_states, hidden_states, valid_lens=valid_lens)
            pooled_output = attention_output.squeeze(1)  # (batch_size, d_model)
            
        elif self.method == 'self_voting':
            # Self-voting attention pooling
            voting_scores = torch.bmm(hidden_states, hidden_states.transpose(1, 2))  # (batch_size, seq_length, seq_length)
            if attention_mask is not None:
                # Create a mask to exclude self-attention scores (diagonal)
                diag_mask = torch.eye(voting_scores.size(1), device=hidden_states.device).bool().unsqueeze(0)
                voting_scores = voting_scores.masked_fill(diag_mask, -1e9) # Use a large negative value for masked elements
                # Apply attention mask to voting scores
                mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2) # (batch_size, seq_length, seq_length)
                voting_scores = voting_scores.masked_fill(mask == 0, -1e9)

            aggregated_scores = voting_scores.sum(dim=1) # Sum scores for each token across all other tokens
            attention_weights = nn.functional.softmax(aggregated_scores, dim=-1) # Normalize scores
            attention_weights = attention_weights.unsqueeze(-1)  # (batch_size, seq_length, 1)
            pooled_output = torch.sum(hidden_states * attention_weights, dim=1)  # (batch_size, d_model)
        else:
            raise ValueError("Invalid pooling method.")
        return pooled_output
    
class NewsEmbedding:
    def __init__(self, save_path=None):
        """
        Initializes interest and style embedding weights from DataFrame and mapping dictionary.
        """
    
        # Use relative path for save_path if not provided
        self.save_path = save_path if save_path is not None else "./data/embeddings"
        os.makedirs(self.save_path, exist_ok=True)  # Ensure the directory exists
        
        # Use relative paths for data files
        emb_df_path = "./data/interest_style_embeddings.feather"
        mapping_path = "./data/mappings/news_id_to_idx.pkl"
        combined_weights_path = f"{self.save_path}/combined_weights.safetensors"
        
        # Check if the combined weights file exists for faster loading
        if os.path.exists(combined_weights_path):
            weights = load_file(combined_weights_path)
            
            self.interest_weights = weights["interest_weights"]
            self.style_weights = weights["style_weights"]
            
            self.vocab_size = self.interest_weights.size(0)
            self.embedding_dim = self.interest_weights.size(1)
            
            print("Loaded pre-initialized embeddings.")
        else:
            print("Pre-initialized embeddings not found. Initializing from scratch.")
            
            # If not found, initialize from scratch
            df = pd.read_feather(emb_df_path)

            # Convert numpy arrays in DataFrame to lists if necessary
            if not isinstance(df['Interest Embedding'].iloc[0], list):
                df['Interest Embedding'] = df['Interest Embedding'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
                df['Style Embedding'] = df['Style Embedding'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
            
            # Load news ID to index mapping
            with open(mapping_path, 'rb') as f:
                self.newsid_to_idx = pickle.load(f)

            self.vocab_size = len(self.newsid_to_idx)
            # Determine embedding dimension from sample data
            sample_interest = df['Interest Embedding'].iloc[0]
            self.embedding_dim = len(sample_interest)

            # Initialize weight tensors
            interest_weights = torch.zeros((self.vocab_size, self.embedding_dim), dtype=torch.float32)
            style_weights = torch.zeros((self.vocab_size, self.embedding_dim), dtype=torch.float32)
            
            # Initialize special tokens (PAD, UNK) weights
            pad_idx = self.newsid_to_idx.get('PAD', 0) # Default to 0 if 'PAD' not in mapping
            unk_idx = self.newsid_to_idx.get('UNK', self.vocab_size - 1) # Default to last index if 'UNK' not in mapping
            nn.init.normal_(interest_weights[unk_idx], mean=0.0, std=0.02)
            nn.init.normal_(style_weights[unk_idx], mean=0.0, std=0.02)

            # Map news IDs from DataFrame to indices and populate weights
            non_special_news_ids = df.loc[~df['News ID'].isin(['PAD', 'UNK']), 'News ID']
            filtered_df = df[df['News ID'].isin(non_special_news_ids)].copy()
            filtered_df['idx'] = filtered_df['News ID'].map(self.newsid_to_idx)
            
            interest_emb_tensor = torch.from_numpy(np.stack(filtered_df['Interest Embedding'].values)).float()
            style_emb_tensor = torch.from_numpy(np.stack(filtered_df['Style Embedding'].values)).float()
            indices = filtered_df['idx'].values
            
            interest_weights[indices] = interest_emb_tensor
            style_weights[indices] = style_emb_tensor
            
            self.interest_weights = interest_weights
            self.style_weights = style_weights
            
            # Save both weights in a single .safetensors file for faster future loading
            weights = {
                "interest_weights": self.interest_weights,
                "style_weights": self.style_weights
            }
            save_file(weights, combined_weights_path)
            print("Saved initialized embeddings for faster future loading.")

    def get_interest_weights(self):
        """Returns the initialized interest embedding weights."""
        return self.interest_weights

    def get_style_weights(self):
        """Returns the initialized style embedding weights."""
        return self.style_weights
