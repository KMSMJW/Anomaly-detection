import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean , z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        length = tf.shape(z_mean)[1]
        dim = tf.shape(z_mean)[2]
        epsilon = tf.keras.backend.random_normal(shape=(batch,length,dim))
        return z_mean + tf.exp(0.5*z_log_var)*epsilon

class Encoder(layers.Layer):
    def __init__(self, in_channels, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.lstm1 = tf.keras.layers.LSTM(max(int(in_channels*2/3),1), return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(max(int(in_channels*2/3),1), return_sequences=True)
        self.Dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(max(int(in_channels/3),1)))
        self.dense_mean = tf.keras.layers.Dense(units=max(int(in_channels/5),1))
        self.dense_log_var = tf.keras.layers.Dense(units=max(int(in_channels/5),1))
        self.sampling = Sampling()

    def call(self,inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        x = self.Dense(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

class Decoder(layers.Layer):
    def __init__(self, in_channels, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.lstm1 = tf.keras.layers.LSTM(max(int(in_channels/3),1), return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(max(int(in_channels*2/3),1), return_sequences=True)
        self.lstm3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(in_channels))

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        x = self.lstm3(x)
        return x

class VariationalAutoEncoder(keras.Model):
    def __init__(self, in_channels, name="autoencoder", **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(in_channels)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        kl_loss = -0.0005*tf.reduce_mean(tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1, axis=-1),axis=-1)
        # kl_loss = -0.0005*tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed