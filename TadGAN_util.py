import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from tqdm.autonotebook import tqdm
from IPython import display
import tensorflow as tf
import logging
import math
import numpy as np
import pandas as pd
import pickle
import keras
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional, LSTM, Flatten, Dense, Reshape, UpSampling1D, TimeDistributed
from tensorflow.keras.layers import Activation, Conv1D, LeakyReLU, Dropout, Add, Layer
from tensorflow.keras.optimizers import Adam
from functools import partial
from scipy import integrate, stats

class RandomWeightedAverage(Layer):
    def _merge_function(self, inputs):
        alpha = K.random_uniform((64,1,1))
        return (alpha)*inputs[0] + (1-alpha)*inputs[1]

def build_encoder_layer(input_shape, encoder_reshape_shape):

    input_layer = layers.Input(shape=(input_shape))

    x = layers.Bidirectional(LSTM(units=100, return_sequences=True))(input_layer)
    x = layers.Flatten()(x)
    x = layers.Dense(encoder_reshape_shape[0]*encoder_reshape_shape[1])(x)
    x = layers.Reshape(target_shape=encoder_reshape_shape)(x)
    model = keras.models.Model(input_layer, x, name='encoder')

    return model

def build_generator_layer(input_shape, generator_reshape_shape):

    input_layer = layers.Input(shape=input_shape)

    x = layers.Flatten()(input_layer)
    x = layers.Dense(generator_reshape_shape[0]*generator_reshape_shape[1])(x)
    x = layers.Reshape(target_shape=generator_reshape_shape)(x)
    x = layers.Bidirectional(LSTM(units=64, return_sequences=True), merge_mode='concat')(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Bidirectional(LSTM(units=64, return_sequences=True), merge_mode='concat')(x)
    x = layers.TimeDistributed(layers.Dense(input_shape[-1]))(x)
    x = layers.Activation(activation='tanh')(x)
    model = keras.models.Model(input_layer, x, name='generator')
    
    return model

def build_critic_x_layer(input_shape):

    input_layer = layers.Input(shape=input_shape)

    x = layers.Conv1D(filters=64, kernel_size=5)(input_layer)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(rate=0.25)(x)
    x = layers.Conv1D(filters=64, kernel_size=5)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(rate=0.25)(x)
    x = layers.Conv1D(filters=64, kernel_size=5)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(rate=0.25)(x)
    x = layers.Conv1D(filters=64, kernel_size=5)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(rate=0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=1)(x)
    model = keras.models.Model(input_layer, x, name='critic_x')

    return model

def build_critic_z_layer(input_shape):

    input_layer = layers.Input(shape=input_shape)

    x = layers.Flatten()(input_layer)
    x = layers.Dense(units=4000)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Dense(units=4000)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Dense(units=1)(x)
    model = keras.models.Model(input_layer, x, name='critic_z')

    return model

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

def critic_x_train_on_batch(x,z, critic_x, generator, batch_size, critic_x_optimizer):
    with tf.GradientTape() as tape:

        valid_x = critic_x(x)
        x_ = generator(z)
        fake_x = critic_x(x_)

        alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        interpolated = alpha*x + (1-alpha)*x_

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = critic_x(interpolated)

        grads = gp_tape.gradient(pred, interpolated)
        grad_norm = tf.norm(tf.reshape(grads, (batch_size, -1)), axis=1)
        gp_loss = 10.0*tf.reduce_mean(tf.square(grad_norm -1.))

        loss1 = wasserstein_loss(-tf.ones_like(valid_x), valid_x)
        loss2 = wasserstein_loss(tf.ones_like(fake_x), fake_x)

        loss = loss1 + loss2 + gp_loss

    gradients = tape.gradient(loss, critic_x.trainable_weights)
    critic_x_optimizer.apply_gradients(zip(gradients, critic_x.trainable_weights))
    return loss

def critic_z_train_on_batch(x,z, encoder, critic_z, batch_size, critic_z_optimizer):
    with tf.GradientTape() as tape:

        z_ = encoder(x)
        valid_z = critic_z(z)
        fake_z = critic_z(z_)

        alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        interpolated = alpha*z + (1-alpha)*z

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = critic_z(interpolated, training=True)
        
        grads = gp_tape.gradient(pred, interpolated)
        grad_norm = tf.norm(tf.reshape(grads, (batch_size, -1)), axis=1)
        gp_loss = 10.0*tf.reduce_mean(tf.square(grad_norm -1.))

        loss1 = wasserstein_loss(-tf.ones_like(valid_z), valid_z)
        loss2 = wasserstein_loss(tf.ones_like(fake_z), fake_z)
        loss = loss1 + loss2 + gp_loss

    gradients = tape.gradient(loss, critic_z.trainable_weights)
    critic_z_optimizer.apply_gradients(zip(gradients, critic_z.trainable_weights))
    return loss

def enc_gen_train_on_batch(x,z, encoder, generator, critic_x, critic_z, encoder_optimizer, generator_optimizer):
    with tf.GradientTape() as enc_tape:

        z_gen_ = encoder(x, training=True)
        x_gen_ = generator(z, training=False)
        x_gen_rec = generator(z_gen_, training=False)

        fake_gen_x = critic_x(x_gen_, training=False)
        fake_gen_z = critic_z(z_gen_, training=False)

        loss1 = wasserstein_loss(fake_gen_x, -tf.ones_like(fake_gen_x))
        loss2 = wasserstein_loss(fake_gen_z, -tf.ones_like(fake_gen_z))
        loss3 = 10.0*tf.reduce_mean(tf.keras.losses.MSE(x, x_gen_rec))

        enc_loss = loss1 + loss2 + loss3

    gradients_encoder = enc_tape.gradient(enc_loss, encoder.trainable_weights)
    encoder_optimizer.apply_gradients(zip(gradients_encoder, encoder.trainable_weights))

    with tf.GradientTape() as gen_tape:

        z_gen_ = encoder(x, training=False)
        x_gen_ = generator(z, training=True)
        x_gen_rec = generator(z_gen_, training=True)

        fake_gen_x = critic_x(x_gen_, training=False)
        fake_gen_z = critic_z(z_gen_, training=False)

        loss1 = wasserstein_loss(fake_gen_x, -tf.ones_like(fake_gen_x))
        loss2 = wasserstein_loss(fake_gen_z, -tf.ones_like(fake_gen_z))
        loss3 = 10.0*tf.reduce_mean(tf.keras.losses.MSE(x, x_gen_rec))

        gen_loss = loss1 + loss2 + loss3

    gradients_generator = gen_tape.gradient(gen_loss, generator.trainable_weights)
    generator_optimizer.apply_gradients(zip(gradients_generator, generator.trainable_weights))
    return enc_loss, gen_loss