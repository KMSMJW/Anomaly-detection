from tensorflow.keras import layers
import tensorflow as tf
import math
import numpy as np
from tensorflow import keras

class PositionalEmbedding(layers.Layer):
    def __init__(self, d_model, max_len = 5000):
        super(PositionalEmbedding, self).__init__()
        pe = tf.Variable(tf.zeros([max_len,d_model], tf.float32), trainable=False) # pe = (max_len,d_model)
        position = tf.range(0,max_len,dtype=tf.float32)[:, tf.newaxis] # position = (max_len,1)
        div_term = tf.math.exp(tf.range(0,d_model,2,tf.float32)*-(tf.math.log(10000.0)/d_model)) # div_term = (d_model/2,)

        pe = pe[:,0::2].assign(tf.math.sin(position*div_term))
        pe = pe[:,1::2].assign(tf.math.cos(position*div_term))

        self.pe = pe[tf.newaxis,:] # pe = (1,max_len,d_model)

    def call(self, x):
        # inputs = (batch_size, seqs, channels)
        return self.pe[:,:tf.shape(x)[1]] # (1,seqs,d_model)

class TokenEmbedding(layers.Layer):
    def __init__(self, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = tf.keras.layers.Conv1D(filters=d_model, kernel_size=3, padding='valid', use_bias=False, kernel_initializer=tf.keras.initializers.HeUniform())

    def call(self, x):
        # x = (batch_size, seqs, channels) 
        x = tf.concat([x[:,-1][:,tf.newaxis], x, x[:,0][:,tf.newaxis]], axis=1) # x = (batch_size, seqs, d_model)
        x = self.tokenConv(x) # x = (batch_size, seqs, d_model)
        return x

class DataEmbedding(layers.Layer):
    def __init__(self,d_model, dropout=0.0):
        super(DataEmbedding,self).__init__()

        self.value_embedding = TokenEmbedding(d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = tf.keras.layers.Dropout(rate = dropout)

    def call(self, x):
        # inputs = (batch_size, seqs, channels)
        x = self.value_embedding(x) + self.position_embedding(x) # x = (batch_size, seqs, d_model)
        return self.dropout(x) 

class TriangularCasualMask():
    def __init__(self, B, L):
        mask_shape = [B, 1, L, L]
        self._mask = tf.experimental.numpy.triu(tf.ones(mask_shape), k=1)
    
    @property
    def mask(self):
        return tf.cast(self._mask, tf.bool)

class AnomalyAttention(layers.Layer):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = tf.keras.layers.Dropout(rate = attention_dropout)
        window_size = win_size
        distances = tf.Variable(tf.zeros([window_size,window_size]), trainable=False)
        for i in range(window_size):
            for j in range(window_size):
                distances = distances[i,j].assign(abs(i-j)) # distances = (seqs, seqs)
        self.distances = tf.constant(distances)
    
    def call(self, queries, keys, values, sigma, attn_mask):
        # queries, keys, values = (batch_size, seqs, n_heads, d_keys), sigma = (batch_size, seqs, n_heads)
        B, L, H, E = queries.shape # L=seqs, H=n_heads, E=d_keys
        _, S, _, D = values.shape # S=seqs, D=d_keys
        scale = self.scale or 1./ math.sqrt(E)

        scores = tf.einsum("blhe,bshe->bhls", queries, keys) # scores = (batch_size, n_heads, seqs, seqs)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCasualMask(B, L)
            scores = tf.where(attn_mask.mask, -np.inf, scores)
        attn = scale * scores # attn = (batch_size, n_heads, seqs, seqs)

        sigma = tf.transpose(sigma, perm=[0,2,1]) # sigma = (batch_size, n_heads, seqs)
        window_size = tf.shape(attn)[-1] # window_size = seqs
        sigma = tf.math.sigmoid(sigma*5) + 1e-5
        sigma = tf.math.pow(3,sigma) - 1
        sigma = tf.tile(sigma[...,tf.newaxis],[1,1,1,window_size]) # sigma = (batch_size, n_heads, seqs, seqs)
        prior = tf.tile(self.distances[tf.newaxis,...][tf.newaxis,...], [sigma.shape[0],sigma.shape[1],1,1]) # prior = (batch_size, n_heads, seqs, seqs)
        prior = 1.0 / (math.sqrt(2*math.pi)*sigma)*tf.exp(-prior**2/2/(sigma**2)) # (batch_size, n_heads, seqs, seqs)

        series = self.dropout(tf.nn.softmax(attn, axis=-1)) # series = (n_heads, seqs,seqs)
        V = tf.einsum("bhls,bshd->blhd", series, values) # V = (batch_size, seqs, n_heads, d_keys)

        if self.output_attention:
            return (V, series, prior, sigma)
        else:
            return (V, None)

class AttentionLayer(layers.Layer):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model//n_heads)
        self.inner_attention = attention
        self.query_projection = tf.keras.layers.Dense(d_keys*n_heads)
        self.key_projection = tf.keras.layers.Dense(d_keys*n_heads)
        self.value_projection = tf.keras.layers.Dense(d_values*n_heads)
        self.sigma_projection = tf.keras.layers.Dense(n_heads)
        self.out_projection = tf.keras.layers.Dense(d_model)

        self.n_heads = n_heads

    def call(self, queries, keys, values, attn_mask):
        # queries = keys = values = (batch_size, seqs, d_model)
        B, L, _ = queries.shape # L = seqs
        _, S, _ = keys.shape # S = seqs
        H = self.n_heads
        x = queries # x = (batch_size, seqs, d_model)
        queries = tf.reshape(self.query_projection(queries),[B,L,H,-1]) # (batch_size, seqs, d_model) -> (batch_size, seqs, n_heads, d_keys)
        keys = tf.reshape(self.key_projection(keys), [B,S,H,-1]) # (batch_size, seqs, d_model) -> (batch_size, seqs, n_heads, d_keys)
        values = tf.reshape(self.value_projection(values), [B, S,H,-1]) # (batch_size, seqs, d_model) -> (batch_size, seqs, n_heads, d_keys)
        sigma = tf.reshape(self.sigma_projection(x), [B,L,H]) # (batch_size, seqs, n_heads) -> (batch_size, seqs, n_heads)

        out, series, prior, sigma = self.inner_attention(queries, keys, values, sigma, attn_mask) # out=(seqs,n_heads,d_keys) series,prior,sigma=(n_heads,seqs,seqs)
        out = tf.reshape(out,[B, L,-1]) # out = (batch_size, seqs, d_model)

        return self.out_projection(out), series, prior, sigma

class EncoderLayer(layers.Layer):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = tf.keras.layers.Conv1D(d_ff, kernel_size=1)
        self.conv2 = tf.keras.layers.Conv1D(d_model, kernel_size=1)
        self.norm1 = keras.layers.LayerNormalization()
        self.norm2 = keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.activation = tf.nn.relu if activation == 'relu' else tf.nn.gelu

    def call(self, x, attn_mask=None):
        # x = (batch_size, seqs, d_model)
        new_x, attn, mask, sigma = self.attention(x,x,x,attn_mask=attn_mask) # new_x=(batch_size,seqs,d_model) attn=(batch_size,n_heads,seqs,seqs) mask=(batch_size,n_heads,seqs,seqs) sigma=(batch_size,n_heads,seqs,seqs)
        x = x + self.dropout(new_x) # x = (batch_size, seqs, d_model)
        y = x = self.norm1(x) # y = (batch_size, seqs, d_model)
        y = self.dropout(self.activation(self.conv1(y))) # y = (batch_size, seqs, d_model)
        y = self.dropout(self.conv2(y)) # y = (batch_size, seqs, d_model)

        return self.norm2(x+y), attn, mask, sigma

class Encoder(layers.Layer):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.norm = norm_layer
        self.attn_layers = attn_layers

    def call(self, x, attn_mask=None):
        # x = (batch_size, seqs, d_model)
        series_list = list()
        prior_list = list()
        sigma_list = list()
        for attn_layer in self.attn_layers.layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)
        
        return x, series_list, prior_list, sigma_list

class AnomalyTransformer(tf.keras.Model):
    def __init__(self, win_size, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512, dropout=0.0, activation='gelu', output_attention=True):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        self.embedding = DataEmbedding(d_model, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout,output_attention=output_attention), 
                        d_model, n_heads),
                    d_model, 
                    d_ff, 
                    dropout=dropout, 
                    activation=activation
                ) for l in range(e_layers)
            ], 
            norm_layer=tf.keras.layers.LayerNormalization()
        )

        self.projection = tf.keras.layers.Dense(c_out)
    
    def call(self,x):
        x = self.embedding(x)
        x, series, prior, sigmas = self.encoder(x)
        x = self.projection(x)

        if self.output_attention:
            return x, series, prior, sigmas
        else:
            return x