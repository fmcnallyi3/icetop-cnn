#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from keras import callbacks
from keras import backend

def hex_filter(shape, dtype=None):
    zeros=np.zeros
    #f = np.array([
    #[[[0]],[[1]],[[1]]],
    #[[[1]],[[1]],[[1]]],
    #[[[1]],[[1]],[[0]]]
    #])
    zeros = np.zeros((shape[2],shape[3]))
    ones = np.ones((shape[2],shape[3]))
    f = np.array([
    [zeros,ones,ones],
    [ones,ones,ones],
    [ones,ones,zeros]
    ])
    assert f.shape == shape
    return backend.variable(f,dtype='float32')