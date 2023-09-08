import torch
import torch.nn as nn
import math
import pytorch_lightning as pl

#embedding layer map each word in the vocab to a vector of dimension d_model
class inputEmbedding(pl.LightningModule):

    #d_mode =  dimension of model
    #vocab_size = how many words are there in vocabulary
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size=vocab_size 
        self.embedding = nn.Embedding(vocab_size,d_model)
    #Forwar method  is responsible for carrying out the forward pass of the neural network. The forward pass is the 
    # process of taking the input data and transforming it into the output data.
    def forward(self,x):
        #as mentioned in the paper we have to multiply embedding layer with sqrt(d_model) . It is done for scaling purpose
        return self.embedding(x) *math.sqrt(self.d_model)
    

class PositionalEncodeing(pl.LightningModule):
    def __init__(self,d_model:int,seq_len:int,droput)->None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(droput)

        #Create a matrix of dimension (seq_len,d_model)
        #seq_len = if we have a sentence with 5 words, the seq_len would be 5.
        pe = torch.zeros(seq_len,d_model) #pe is positional encodng
        #Create a vector of shape seq_len
        position = torch.arange(0,seq_len,dtype=torch.float32).unsqueeze(1)#creating tesnor of shape (seq_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)
    #we need to add positional encoding to every word inside the sentence
    def forward(self, x):
        #self.pe is buffer registered during intilazation with shape (1,seq_len_d_model)
        #x dimesion -> batch_size,seq_len,d_model
        #x.shape[1] will be seq_len
        positional_encodings = self.pe[:, :x.shape[1], :]#retreive position encoding for input sequence shape (1,seq_len,d_model)
        x = x+positional_encodings#element wise addition
        #x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)#Dropout helps prevent overfitting by randomly setting some elements of the tensor to zero during training


class LayerNormalization(pl.LightningModule):#in transformer it is called Add & Norm
    def __init__(self, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps#we need eps as in layer normalization formula we have eps=> x = x-u/math.sqrt((sigma)^2+eps)
        #nn.Parameter means it should be treated as learnable parameter during training. here PyTorch automatically tracks gradients for that tensor and includes it in the optimization process.
        self.alpha = nn.Parameter(torch.ones(1)) # alpha is a learnable parameter - multiplicative parameter
        self.bias = nn.Parameter(torch.zeros(1)) # bias is a learnable parameter - additive parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    

class FeedForwardBlock(pl.LightningModule):#Feed forwaard is fully conncted layer
    #d_model = 512 matrix
    #d_ff = 2048 matrix
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        #First linear transformation is used for dimeanionality transformation. It increases the dimension from d_model to d_ff
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1 #512,2048 # w1 have shape = d_model,d_ff bias of shape = (d_ff,)
        self.dropout = nn.Dropout(dropout)
        #Second linear transformation is used for Dimensionality Reduction and Reshaping
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2 #2048,512
        #Overall, the purpose of having two linear transformations in the feedforward block is to enable the model to learn 
        # complex relationships within the data by increasing dimensionality and then reducing it

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        #x is of shape (batch, seq_len, d_model)
        #self.linear_1(x):The operation involves multiplying x by the weight matrix w1 and adding the bias b1 element-wise.
        #result of the above operation is (batch_size, seq_len, d_ff)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

class MultiHeadAttentionBlock(pl.LightningModule):
    #h - no of head
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod# means we do not need to create instance of this finction
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]#Last dimesion of query 
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)#As in line 131 dimesion is (batch, h, seq_len, d_k)
        #so we are taking last two dimesion i.e. seq_len,d_k and transpose it hence d_k,seq_len
        if mask is not None:#before applying softmax we apply mask
            # replace it with very small value
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        #mask if we dont want some word to interact with another word we replace it with small value and when we do softmax these value will
        #become zero hence we hide attention for those words hence this is called masking
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)======> Q``
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model) =====>K`
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)=====>V`

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)# we did transpose as we want h as second dimenion
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # omputes the attention scores between the query, key, and value matrices.
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)#contiguos -?

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)
    
class ResidualConnection(pl.LightningModule):#skip connection
    
        def __init__(self, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization()
    
        def forward(self, x, sublayer):#sublayer - previos layer
            return sublayer(x)
        

class EncoderBlock(pl.LightningModule):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])#here 2 means make two instance of residual connection
        #ResidualConnection(dropout) means in each instance of residual connection we must have dropout
    def forward(self, x, src_mask):
        #src_mask :mask required for input of the encoder as we dont want padding word to interact with other word
        #Here x is query,key value since this is self attention hence all the value are same 
        #x: This is input data, often representing some kind of sequence like words in a sentence or tokens in a document.
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(pl.LightningModule):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers#ayers are expected to be stacked on top of each other to process input sequentially.
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)# Layer normalization ensures that the output is standardized and stable.
    

class DecoderBlock(pl.LightningModule):
    #Block named mask multi head attention is self attention and block name
    #  multi head attantion is cross attection as it is taking two output(key and value) from encoder
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    #src_mask: mask aplied on ecoder
    #tgt_mask: mask applied on decoder
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))#(x,key,value,src_mask)
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(pl.LightningModule):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)#here output dimesion will be (batch,seq_len,d_model)
    

class ProjectionLayer(nn.Module):#Linear block in transformer image

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)
    

class ProjectionLayer(nn.Module):#Linear block in transformer image

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)
    
class Transformer(pl.LightningModule):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: inputEmbedding, tgt_embed: inputEmbedding, src_pos: PositionalEncodeing, tgt_pos: PositionalEncodeing, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)#First embdedding
        src = self.src_pos(src)#then apply posional encoding
        return self.encoder(src, src_mask)#then apply encoder
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)

#N - No of encoder and decoder block according to paper here it is 6
#d_ff  - hidden - feed forward   
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = inputEmbedding(d_model, src_vocab_size)
    tgt_embed = inputEmbedding(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncodeing(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncodeing(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    e1,e2,e3 = encoder_blocks[:3]
    d1,d2,d3 = decoder_blocks[3:]

    encoder_blocks1 = [e1,e2,e3,e1,e2,e3]
    decoder_blocks1 = [d1,d2,d3,d1,d2,d3]
    
    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks1))
    decoder = Decoder(nn.ModuleList(decoder_blocks1))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters using xavier uniform to make the training faster
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer