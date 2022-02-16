#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from keras import callbacks
from keras import backend
import functools

hex_filter = np.array([
    [ 0 , 1 , 1 ],
    [ 1 , 1 , 1 ],
    [ 1 , 1 , 0 ]
], dtype='float32').reshape([3,3,1,1])

def hex_init(shape, dtype=None):
    init=tf.convert_to_tensor(hex_filter,dtype=dtype)
    init=np.repeat(np.repeat(hex_filter, shape[2], axis=2),shape[3],axis=3)
    return init

#def hex_filter(shape, dtype=None):
#    zeros=np.zeros
#    #f = np.array([
#    #[[[0]],[[1]],[[1]]],
#    #[[[1]],[[1]],[[1]]],
#    #[[[1]],[[1]],[[0]]]
#    #])
#    zeros = np.zeros((shape[2],shape[3]))
#    ones = np.ones((shape[2],shape[3]))
#    f = np.array([
#    [zeros,ones,ones],
#    [ones,ones,ones],
#    [ones,ones,zeros]
#    ])
#    assert f.shape == shape
#    return backend.variable(f,dtype='float32')

#class MaskedConv2D(tf.keras.layers.Layer):
#    def __init__(self, *args, **kwargs):
#        super(MaskedConv2D, self).__init__()
#        self.conv2d = tf.keras.layers.Conv2D(*args, **kwargs)
#        self.mask = np.array([
#            [0,1,1],
#            [1,1,1],
#            [1,1,0]
#        ])
#        self.mask = tf.reshape(self.mask, self.mask.shape + (1,1))
#        
#    def build(self, input_shape):
#        self.conv2d.build(input_shape[0])
#        self._convolution_op = self.conv2d._convolution_op
#        
#    def masked_convolution_op(self, filters, kernel, mask):
#        return self._convolution_op(filters, tf.math.multiply(kernel, self.mask))
#        
#    def call(self, inputs):
#        x = inputs
#        self.conv2d._convolution_op = functools.partial(self.masked_convolution_op, mask=self.mask)
#        return self.conv2d.call(x)

#class MaskedConv2D(layers.Conv2D):
#    def call(self, inputs):
#        mask=np.array([
#            [0,1,1],
#            [1,1,1],
#            [1,1,0]])
#        result = self.convolution_op(inputs, tf.math.multiply(kernel,mask))
#        if self.use_bias:
#            result = result + self.bias
#        return result

class MaskedConv2D(layers.Conv2D):
    def convolution_op(self, inputs, kernel):
        #mask=np.array([
        #    [0,1,1],
        #    [1,1,1],
        #    [1,1,0]],dtype='float32')
        #mask = tf.reshape(mask,mask.shape+(1,1))
        #print(kernel)
        #print(tf.math.multiply(kernel,mask))
        return tf.nn.conv2d(
            inputs,
            tf.math.multiply(kernel,hex_filter),
            padding="SAME",
            strides=list(self.strides),
            name=self.__class__.__name__,
        )

#class MaskedConv2D(tf.keras.layers.Layer):
#    def __init__(self, *args, **kwargs):
#        super(MaskedConv2D, self).__init__()
#        self.conv2d = tf.keras.layers.Conv2D(*args, **kwargs)
#        
#    def build(self, input_shape):
#        self.conv2d.build(input_shape[0])
#        self._convolution_op = self.conv2d._convolution_op
#        
#    def masked_convolution_op(self, filters, kernel, mask):
#        return self._convolution_op(filters, tf.math.multiply(kernel, tf.reshape(mask, mask.shape + [1,1] )))
#        
#    def call(self, inputs):
#        x = inputs
#        self.conv2d._convolution_op = functools.partial(self.masked_convolution_op, mask=hex_filter)
#        return self.conv2d.call(x)