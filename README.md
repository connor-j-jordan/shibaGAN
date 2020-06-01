## shibaGAN: Predicting 3D genome folding from DNA sequence
___

This project is largely adapted from [hicGAN](https://github.com/kimmo1019/hicGAN) and [Akita](https://github.com/calico/basenji/tree/master/manuscripts/akita) under the MIT and Apache licenses, respectively.

shibaGAN substitutes the L<sub>2</sub> MSE objective function used by Akita for an adversarial approach, constructing a Wasserstein GAN to predict 3D genome folding patterns (in a Hi-C matrix) from linear sequence data.
___

### Requirements/Dependencies

- Numpy, Scipy, Pandas
- TensorFlow >= 1.15
- TensorLayer >= 1.9.1

___

### Installation

shibaGAN can be downloaded in shell:
```
git clone https://github.com/connor-j-jordan/shibaGAN
```

Installation has been tested on a Linux platform, as well as Windows running the Ubuntu subsystem.
___

### Reproducing Results

___

### Training shibaGAN on new data

___

### License

This project is licensed by the MIT License - see LICENSE for details.
