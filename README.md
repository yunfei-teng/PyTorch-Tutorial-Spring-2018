# PyTorch Tutorial
PyTorch tutorial codes for course EL-9133 Advanced Machine Learning, NYU, Spring 2018

# Requirements:
Pytorch 0.2(newest version is stable 1.0)

## Notes:
We have 2 recitations in total:
1. Introduction to Neural Networks and PyTorch.
2. GANs, ResNets, Autoencoders, VisualBackProp, Skip connections, Batch normalization, and more examples.

### Recitation 1:
1. cosine_sim: Calculating cosine similarity of vectors.
2. classification: MNIST digit classification.
3. regression: Regression for polynomial functions.

### Recitation 2:
1. Autoencoder: Autoencoder + Unet.
2. classification: STL10 image classification + ResNet + Visual BackProp.
3. GAN: generative adversarial nets.

## Requirments
Please install [PyTorch](http://pytorch.org/) as indicated. Please be careful about the *version* of Python, PyTorch and Cuda. I strongly recommend Python3 instead of Python2. Before you run the codes, check whether your machine supports GPU or not.

## Run
The dataset should be downloaded automatically. STL10 is a large dataset, and it may take several minutes.
```
python3 train.py
```
## Homework Tips
1. You need to modify the codes and add functions like plotting for your homework. 
2. Take care of regression part, because you need to deal with two variables.
3. Read option.py since you may need to adjust the parameters.

If you have any question, please email yt1208@nyu.edu. 

## Thanks
All the codes are inspired by [PyTorch official examples](https://github.com/pytorch/examples). 
