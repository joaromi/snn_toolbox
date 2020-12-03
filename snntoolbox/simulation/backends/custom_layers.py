import tensorflow as tf
import numpy as np
import keras
from keras import backend as K
from keras.layers import Layer

class Normalizable_Add(Layer):
    
   def __init__(self, **kwargs): 
        self.concat = keras.layers.Concatenate(axis=-1)
        super(Normalizable_Add, self).__init__(**kwargs) 

   def build(self, input_shape): 
        n = len(input_shape)
        self.output_dim = input_shape[0][-1]
        self.b = [None]*n
        for i in range(len(self.b)):
             self.b[i] = self.add_weight(
                         shape = (self.output_dim,), 
                         initializer = "zeros", trainable = False
                         )
        
        weights_conv = np.zeros([1, 1, n*self.output_dim, self.output_dim])
        for k in range(self.output_dim):
            weights_conv[:, :, k::self.output_dim, k] = 1
        
        self.conv = keras.layers.Conv2D(
                filters=self.output_dim,
                kernel_size=1, 
                weights=(weights_conv, np.zeros(self.output_dim))
                )

        super(Normalizable_Add, self).build(input_shape)

   def call(self, input_data):
        tensor = [None]*len(self.b)
        for i,image in enumerate(input_data):
            tensor[i] = image+self.b[i]
            
        out = self.concat(tensor)
        out = self.conv(out)
        
        return out

   def compute_output_shape(self, input_shape): 
        return input_shape[0] + (self.output_dim,)
