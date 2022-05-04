import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.io import wavfile
from tqdm.autonotebook import tqdm
from IPython import display

def LSTM(seqs, channel):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer((seqs,channel)))
    model.add(tf.keras.layers.LSTM(channel*2, return_sequences=True))
    model.add(tf.keras.layers.LSTM(channel*2, return_sequences=True))
    model.add(tf.keras.layers.LSTM(channel*2, return_sequences=True))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(channel)))
    return model