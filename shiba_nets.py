#==============================================================================#
#                                  DEPENDENCIES                                #
#==============================================================================#

import random, math, time, json
import numpy as np, pandas as pd
from np import expand_dims, mean, ones
from np.random import randn, randint

import tensorflow as tf
import tensorlayer as tl
from tf import keras, backend

from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose
from keras.layers import LeakyReLU, BatchNormalization

from keras.initializers import RandomNormal
from keras.constraints import Constraint

#==============================================================================#
#                                   GENERATOR                                  #
#==============================================================================#

class ClipConstraint(constraints.Constraint):
    """Class to constrain model weights using a paramterized value."""
    
    def __init__(self, clip_val : float=0.01):
        self.clip_val = clip_val
        
    def __call__(self, weights):
        """Clip the model weights."""
        return backend.clip(weights, -self.clip_value, self.clip_value)
    
    def get_config(self):
        """Return the config value."""
        return {'clip_value': self.clip_value}

class Generator:
    """The generator network of the WGAN architecture."""
    
    def __init__(self, latent_dim):
        self.w = initializers.RandomNormal(stddev=0.02)
        
        self.build(latent_dim)
        self.network.summary()
        
    def build(self, latent_dim) -> None:
        """Constructs and stores the keras network architecture."""
        
        # Construct the network with keras blocks
        net = models.Sequential()
        
        # Foundation layer for Hi-C matrix
        n_nodes = 128 * 7 * 7
        net.add(Dense(n_nodes, kernel_initializer=self.w, input_dim=latent_dim))
        net.add(LeakyReLU(alpha=0.2))
        net.add(Reshape((7, 7, 128)))

        def upsample(self, dim : int) -> None:
            """Adds an upsampling layer to the network."""
            net.add(Conv2DTranspose(128, (dim,dim), strides=(2,2), padding='same', kernel_initializer=self.w))
            net.add(BatchNormalization())
            net.add(LeakyReLU(alpha=0.2))
        
        # Upsample...
        net.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=self.w))
        net.add(BatchNormalization())
        net.add(LeakyReLU(alpha=0.2))
        
        # Output layer
        net.add(Conv2D(1, (7,7), activation='tanh', padding='same', kernel_initializer=self.w))
        
        self.network = net

#==============================================================================#
#                                     CRITIC                                   #
#==============================================================================#

class Critic:
    """The 'critic' of the WGAN architecture."""
    
    def __init__(self, shape):
        self.w = RandomNormal(stddev=0.02)
        
        self.build(shape)
        self.network.summary()
        
    def build(self, shape) -> None:
        """Constructs and stores the keras network architecture."""
        
        # Constrain the weights using class
        const = ClipConstraint(0.01)
        
        # Construct the network with keras blocks
        net = Sequential()
        
        # Downsample...
        net.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=self.w, kernel_constraint=const, input_shape=shape))
        net.add(BatchNormalization())
        net.add(LeakyReLU(alpha=0.2))
        
        # Downsample...
        net.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=self.w, kernel_constraint=const))
        net.add(BatchNormalization())
        net.add(LeakyReLU(alpha=0.2))
        
        # Dense layer for scoring/activation
        net.add(Flatten())
        net.add(Dense(1))
        
        # Compile the network with Wasserstein loss
        net.compile(loss=wasserstein_loss, optimizer=RMSprop(lr=0.00005))
        
        self.network = net