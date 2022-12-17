import numpy as np
import torch

# read data
data = torch.load("rbc_data.pt")

# standardization
std = torch.std(data)
avg = torch.mean(data)
data = (data - avg)/std
data = data[:,:,::4,::4]

# divide each rectangular snapshot into 7 subregions
# data_prep shape: num_subregions * time * channels * w * h
data_prep = torch.stack([data[:,:,:,k*64:(k+1) * 64] for k in range(7)])

# use sliding windows to generate 9870 samples
# training 6000, validation 2000, test 1870
for i in range(7):
    torch.save(torch.FloatTensor(data_prep[i]), "rbc_data/sample_" + str(i) + ".pt")