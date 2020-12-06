import tensorflow as tf
import numpy as np
import keras
from keras import backend as K
from keras.layers import Layer
from tensorflow.python.keras import activations

class NormAdd(Layer):
    
     def __init__(self, activation=None, **kwargs):
          self.activation = activations.get(activation)
          self.concat = keras.layers.Concatenate(axis=-1)
          super(NormAdd, self).__init__(**kwargs) 

     def build(self, input_shape): 
          n = len(input_shape)
          self.filters = input_shape[0][-1]
          self.b = [None]*n
          for i in range(len(self.b)):
               self.b[i] = self.add_weight(
                              name="unshift"+str(i),
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

          super(NormAdd, self).build(input_shape)

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
        config['weights'] = self.get_weights()
        config.update({
            'activation': self.activation,
            #'filters': self.filters
        })
        return config


class NormReshape(Layer):
    
     def __init__(self, target_shape, **kwargs):
          self.target_shape = target_shape
          self.resh = keras.layers.Reshape(self.target_shape, **kwargs)
          #self.built = False
          super(NormReshape, self).__init__(**kwargs) 

     def build(self, input_shape): 
          self.in_channels = input_shape[-1]

          self.lmbda = self.add_weight(
                         name="lambda",
                         shape = (self.in_channels,), 
                         initializer = "ones", trainable = False
                         )
          self.shift = self.add_weight(
                         name = "shift",
                         shape = (self.in_channels,), 
                         initializer = "zeros", trainable = False
                         )
          super(NormReshape, self).build(input_shape)
          #self.built = True

     def call(self, input_data):
          #if not self.built: self.build(tf.shape(input_data))
          out = input_data*(self.lmbda-self.shift)+self.shift   
          out = self.resh(out)
          return out

     # def compute_output_shape(self, input_shape): 
     #      return input_shape[0] + (self.filters,)

     def get_config(self):
        config = super().get_config().copy()
        config.update({
            'target_shape': self.target_shape
        })
        return config

     def get_weights(self):
          return [self.lmbda, self.shift]
     
     def set_weights(self, weights):
          try:
               self.lmbda = weights[0]
               self.shift = weights[1]
          except ValueError:
               print('Weights need to be of shape: [',tf.shape(self.lmbda),',',tf.shape(self.shift),'].')

