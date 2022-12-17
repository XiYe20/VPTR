# VPTR in Simulation of Rayleigh-Bénard convection (RBC)

The aim of this custom folder is:

- Ultilize the original code base of autoregressive model
- Fit the model to RBC dataset

## Dataset

Run the file ```data.sh```:

- Download the dataset to current folder
- Split the original dataset into 7 time series dataset of reasonable size.

Format of the dataset:

```data_i = T x C x H x W ``` with:

|---|------------------------------------------------------------------|
| T | number of time stamps                                            |
| C | number of channels (1 channel for $v_x$ and 1 channel for $v_y$) |
| H | vertical size of the frame                                       |
| W | horizontal size of the frame                                     |

## Training

Run the following command:

```
>>> python3 ./customTrain_AutoEncoder.py \\
        --data_dir ... \\
        --hyper_params ... \\
        --checkpoint_dir ...                 
```