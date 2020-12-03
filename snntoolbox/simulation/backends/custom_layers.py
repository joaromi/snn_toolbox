import tensorflow as tf
import numpy as np
import keras
from keras import backend as K
from keras.layers import Layer
from tensorflow.python.keras import activations

class Normalizable_Add(Layer):
    
     def __init__(self, activation=None, **kwargs):
          self.activation = activations.get(activation)
          self.concat = keras.layers.Concatenate(axis=-1)
          super(Normalizable_Add, self).__init__(**kwargs) 

     def build(self, input_shape): 
          n = len(input_shape)
          self.filters = input_shape[0][-1]
          self.b = [None]*n
          for i in range(len(self.b)):
               self.b[i] = self.add_weight(
                              shape = (self.filters,), 
                              initializer = "zeros", trainable = False
                              )
          
          weights_conv = np.zeros([1, 1, n*self.filters, self.filters])
          for k in range(self.filters):
               weights_conv[:, :, k::self.filters, k] = 1
          
          self.conv = keras.layers.Conv2D(
                    filters=self.filters,
                    kernel_size=1, 
                    weights=(weights_conv, np.zeros(self.filters)),
                    )    

          super(Normalizable_Add, self).build(input_shape)

     def call(self, input_data):
          tensor = [None]*len(self.b)
          for i,image in enumerate(input_data):
               tensor[i] = image+self.b[i]
               
          out = self.concat(tensor)
          out = self.conv(out)
          
          if self.activation is not None:
               return self.activation(out)
          return out

     def compute_output_shape(self, input_shape): 
          return input_shape[0] + (self.filters,)

     def get_config(self):
        config = super().get_config().copy()
        config.update({
            'activation': self.activation
        })
        return config
