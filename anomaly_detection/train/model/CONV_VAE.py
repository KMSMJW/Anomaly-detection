import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.autonotebook import tqdm
from IPython import display
from scipy.io import wavfile

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean , z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch,dim))
        return z_mean + tf.exp(0.5*z_log_var)*epsilon

class Encoder(tf.keras.layers.Layer):
    def __init__(self, channel, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.conv0 = tf.keras.layers.Conv1D(filters=max(channel*2//3, 1), kernel_size=2, padding='valid', strides=2, activation='relu')
        self.conv1 = tf.keras.layers.Conv1D(filters=max(channel//3, 1), kernel_size=2, padding='valid', strides=2, activation='relu')
        self.flat = tf.keras.layers.Flatten()
        self.dense_mean = tf.keras.layers.Dense(units=max(channel//5,1))
        self.dense_log_var = tf.keras.layers.Dense(units=max(channel//5,1))
        self.sampling = Sampling()

    def call(self,inputs):
        x = self.conv0(inputs)
        x = self.conv1(x)
        x = self.flat(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

class Decoder(tf.keras.layers.Layer):
    def __init__(self, seqs, channel, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense = tf.keras.layers.Dense(units=max((seqs//4),1)*max((channel//3),1), activation='relu')
        self.reshape = tf.keras.layers.Reshape(target_shape=(max(seqs//4,1),max(channel//3,1)))
        self.conv1 = tf.keras.layers.Conv1DTranspose(filters=max(channel*2//3,1), kernel_size=2, padding='valid', strides=2, activation='relu')
        self.conv2 = tf.keras.layers.Conv1DTranspose(filters=channel, kernel_size=2, padding='valid', strides=2, activation='relu')

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class VariationalAutoEncoder(tf.keras.Model):
    def __init__(self, seqs, channel, name="autoencoder", **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.encoder = Encoder(channel)
        self.decoder = Decoder(seqs, channel)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # kl_loss = -0.0005*tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        kl_loss = -0.0005*(tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1, axis=-1))
        self.add_loss(kl_loss)
        return reconstructed