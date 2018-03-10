# Autoencoder + Unet
There are three folders:
1. Pipeline: Contains the codes of a series of necesary commands and functions. 
2. Architecture: Contains the codes for building the models.
3. Data: Contains the codes for loading and processing data.
## Notes
This experiment helps you understand the power of skip connection. You can run the model twice with and without U-Net respectively. 
By default, plain autoencoder is applied. To run U-Net, use command:
```
python3 train --use_unet
```
