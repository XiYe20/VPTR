# VPTR: Efficient Transformers for Video Prediction
Video future frames prediction based on Transformers. Published on ICPR2022, https://ieeexplore.ieee.org/abstract/document/9956707

# Video prediction by efficient transformers
Published on Image and Vision Computing. https://arxiv.org/pdf/2212.06026.pdf

The overall framework for video prediction.
![Alt text](./docs/Framework.png?raw=true "Title")

Fully autoregressive (left) and non-autoregressive VPTR (right).

![Alt text](./docs/VPTR.png?raw=true "Title")

## Pretrained-models
Download the checkpoints from here: https://polymtlca0-my.sharepoint.com/:f:/g/personal/xi_ye_polymtl_ca/EuxjSddJ7wNIsiSTOfB-u7AB7qQhP5H0iX2a5mbaowiSZw?e=IEj1bd

See Test_AutoEncoder.ipynb and Test_VPTR.ipynb for the detatiled test functions.

## Training
### Stage 1: train_AutoEncoder.py
Train the autoencoder firstly, save the ckpt, load it for stage 2


### Stage 2: Train Transformer for the video prediction
train_FAR.py: Fully autoregressive model \
train_FAR_mp.py: multiple gpu training (single machine) \
train_NAR.py: Non-autoregressive model \
train_NAR_mp.py: multiple gpu training (single machine)




### Dataset folder structure
/MovingMNIST \
  &nbsp;&nbsp;&nbsp;&nbsp; moving-mnist-train.npz \
  &nbsp;&nbsp;&nbsp;&nbsp; moving-mnist-test.npz \
  &nbsp;&nbsp;&nbsp;&nbsp; moving-mnist-val.npz

/KTH \
  &nbsp;&nbsp;&nbsp;&nbsp; boxing/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; person01_boxing_d1/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; image_0001.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; image_0002.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; person01_boxing_d2/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; image_0001.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; image_0002.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... 

  &nbsp;&nbsp;&nbsp;&nbsp; handclapping/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... \
  &nbsp;&nbsp;&nbsp;&nbsp; handwaving/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... \
  &nbsp;&nbsp;&nbsp;&nbsp; jogging_no_empty/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... \
  &nbsp;&nbsp;&nbsp;&nbsp; running_no_empty/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... \
  &nbsp;&nbsp;&nbsp;&nbsp; walking_no_empty/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ...


/BAIR \
  &nbsp;&nbsp;&nbsp;&nbsp; test/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; example_0/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0000.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0001.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; example_1/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0000.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0001.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; example_... \
&nbsp;&nbsp;&nbsp;&nbsp; train/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; example_0/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0000.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0001.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; example_... 

### Citing
   
Please cite the paper if you find our work is helpful.
```
@inproceedings{ye2022vptr,
  title={VPTR: Efficient Transformers for Video Prediction},
  author={Ye, Xi and Bilodeau, Guillaume-Alexandre},
  booktitle={2022 26th International Conference on Pattern Recognition (ICPR)},
  pages={3492--3499},
  year={2022},
  organization={IEEE}
}
```
```
@article{ye2022video,
  title={Video prediction by efficient transformers},
  author={Ye, Xi and Bilodeau, Guillaume-Alexandre},
  journal={Image and Vision Computing},
  pages={104612},
  year={2022},
  publisher={Elsevier}
}
```

### Correction about the paper

Recently, we found a mistake in our ICPR paper. For the BAIR experiments, the previous papers predict 28 future frames instead of 10. Specifically, the results in "TABLE II: Results on BAIR" are for 10 future frames instead of 28. The results for 28 predicted frames are updated here, see the following correct table.

![Alt text](./docs/Table2_Corrected.png?raw=true "Title")

We apologize for the mistake, the correction does not affect our conclusions.
