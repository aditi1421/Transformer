import torch 
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) #Input embedding

    def forward(self,x):
        return self.embedding * math.sqrt(self.d_model)

class PositionEncoding(nn.Module):

    def __init__(self, d_model:int, seq_len:int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

       #Develop a matrix of shape( seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len,1) -> This is for the formula 
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        #Apply the sin to even positions
        pe[:,0::2] = torch.sin(position * div_term)
        #cosine to odd positions 
        pe[:,1::2] = torch.cos(position * div_term)


        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe) #pe is stored in the model without being updated during training

    def forward(self, x): 
        x = x + (self.pe[:,:x.shape[1], :]).requires_grad_(False) #We only need positional encodings up to the actual sequence length of x.
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) ->None: 
        super().init() 
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) #Added
    
    def forward(self,x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean)/ (std + self.eps) + self.bias 

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model:int, d_ff:int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.linear(d_model, d_ff) #W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) #W2 and B2

    def forward(self,x):
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_len, d_ff) --> (Batch, Seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
        
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model) #Wq, Wk, Wv

        self.w_o = nn.Linear(d_model, d_model) #Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query,key,value,mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        #(Batch, h, Seq_len, d_k) @ (Batch, h, d_k, Seq_len) --> (Batch, h, Seq_len, Seq_len)
        attn_scores =(query @ key.transpose(-2,-1)) / math.sqrt(d_k) 
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9) #Replace all the values with 0 with -1e9
        attn_scores = attn_scores.softmax(dim = -1) #Softmax over the last dimension

        if dropout is not None:
            attn_scores = dropout(attn_scores)

        return (attn_scores @ value), attn_scores 
        


    def forward(self, q, k, v, mask = None): #Mask: If we want some words to be ignored
        query = self.w_q(q) #(Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)
        key = self.w_k(k)   #Same as query 
        value = self.w_v(v)

        # (Batch, Seq_len, d_model) --> (Batch, h, Seq_len, d_k) --> (Batch, h, Seq_len, d_k)
        query = query.view(query.shape[0],query.shape[1], self.h, self.d_k).transpose(1,2) 
        key = key.view(key.shape[0],key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1], self.h, self.d_k).transpose(1,2)

        x,self.attn_scores = MultiHeadAttentionBlock.attention(query,key,value,mask, self.dropout)

        # (Batch, h, Seq_len, d_k) --> (Batch, Seq_len, h, d_k) --> (Batch, Seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k) #d_k is d_model // h

        # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)
        return self.w_o(x) #Output matrix 

class ResidualConnection(nn.Module):

    def __init__(self, size: int, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNormalization()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) #Sublayer is the block that we want to apply the residual connection to

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block):