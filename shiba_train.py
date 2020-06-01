#==============================================================================#
#                                  DEPENDENCIES                                #
#==============================================================================#

import random, math, time, json
import numpy as np, pandas as pd

import tensorflow as tf
import tensorlayer as tl
from tf import keras, backend

from shiba_nets import Generator, Critic, ClipConstraint

#==============================================================================#
#                             WASSERSTEIN TRAINING                             #
#==============================================================================#

class TrainingModule:
    """Class for cross-training both WGAN networks."""
    
    def __init__(self, gen, critic, latent_dim : int, l_rate : float=0.001):
        self.gen, self.critic = gen, critic
        self.learn = l_rate
        self.latent_dim = latent_dim
        
    def load_dataset() -> None:
        """Load the ground-truth Hi-C matrices."""
        pass
    
    def split(self, dataset) -> np.ndarray:
        """Splits the dataset into training, validation, and test sets."""
        
        # TODO: Implement data splitting
        
        test_set = []
        
        self.train_set, self.val_set = [], []
        return test_set
    
    def w_loss(self, y_true, y_pred) -> float:
        """Calculates Wasserstein loss for each network."""
        return backend.mean(y_true * y_pred)
    
    def _compile_GAN(self) -> None:
        """Combine the two adversarial networks into one WGAN."""
        
        wgan = models.Sequential()
        # Add the two component models
        wgan.add(gen.network); wgan.add(critic.network)
        # Compile the model
        opt = optimizers.RMSprop(lr=0.00005)
        model.compile(loss=wasserstein_loss, optimizer=opt)
        
        self.wgan = wgan
    
    def generate_latent_points(self):
        """Generates random points in the latent space for generation."""
        
        for base in self.latent_dim:
            
        
    
    def train(self, dataset, num_epochs : int=500, batch_size : int=3):
        """Trains each network for the given number of iterations."""
        self.critic.trainable = False
        self._compile_GAN()
        
        bat_per_epoch = int(dataset.shape[0] / batch_size)
        itr = bat_per_epoch * num_epochs
        
        for i in range(itr):
            pass
        
        
        
            if ((i+1) % bat_per_epo == 0):
                

#==============================================================================#
#                                   MAIN METHOD                                #
#==============================================================================#

def main():
    
    LATENT_DIM = 128; CRIT_SHAPE = (1, 1)
    
    # Initialize adversarial networks
    gen, critic = Generator(LATENT_DIM), Critic(CRIT_SHAPE)
    gen.network.summary(); critic.network.summary()
    
    # Initialize training module
    trainer = TrainingModule(gen, critic, LATENT_DIM)
    # Load and split the dataset
    dataset = trainer.load('seq.txt')
    test_set = trainer.split(dataset)
    
main()