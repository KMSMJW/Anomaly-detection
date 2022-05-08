import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.autonotebook import tqdm
from IPython import display
from scipy.io import wavfile
from tensorflow.keras import layers
from tensorflow import keras

def ae(time_steps, in_channels):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(max(in_channels*2//3,1), kernel_size=2, padding='valid', strides=2, activation='relu'))
    model.add(tf.keras.layers.Conv1D(max(in_channels//3,1), kernel_size=2, padding='valid', strides=2, activation='relu'))
    model.add(tf.keras.layers.Conv1DTranspose(max(in_channels*2//3,1), kernel_size=2, padding='valid', strides=2, activation='relu'))
    model.add(tf.keras.layers.Conv1DTranspose(max(in_channels,1), kernel_size=2, padding='valid', strides=2, activation='relu'))
    return model