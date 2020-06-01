#==============================================================================#
#                             IMPLEMENTATION NOTES                             #
#==============================================================================#
#                                                                              #
#       The adversarial networks are instantiated as their own classes. Their  #
#   parameters, hyperparameters and architectures are attributes of these      #
#   classes. The two classes comprise a WGAN which uses Wasserstein loss to    #
#   predict 3D genome folding from linear sequence.                            #
#                                                                              #
#                                 shibaGAN.sh                                  #
#                                                                              #
#   Wasserstein GAN (WGAN) predicts Hi-C matrices from linear sequence data.   #
#                                                                              #
#   USAGE:                                                                     #
#   shibaGAN.sh <DATADIR>                                                      #
#                                                                              #
#==============================================================================#

#!/bin/bash

echo "Loading data..."
get_data.sh

echo "Instantiating convolutional networks..."
python shiba_nets.py

echo "Training shibaGAN..."
python shiba_train.py

python shiba_eval.py

echo "Done!"
