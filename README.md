# Turbid Water Imaging Code & Datasets

This repository contains code for the paper _Reconstruction Method of Scattering Light Information for depth imaging in Turbid Water using Deep Learning and TSIDF_ by YuJie Fang, JunMing Wu, et al. The captured datasets and the results of network training can be downloaded separately from our [project webpage](https://pan.baidu.com/s/1NGP_sAvqMGkbPY5XlHPKIA ).

If required, email fang_yj@bitzh.edu.cn to obtain passwords for all datasets and experimental results.

## Experimental Setup & Datasets
The dataset is collected using the OPT8241CDK TOF 3D camera, where different concentrations of milk are used to simulate varying levels of turbidity in the turbid water medium. The blurred image data is collected through turbid water, while the corresponding clear images are obtained through a clear water collection method. The target for testing was a plaster statue characterized by complex geometric features and uniform optical reflectivity.
<div align="center">
  <img src="https://github.com/fyj0202/USI/blob/main/figure1.png" width="70%">
</div> <br />

## Code Description
The code folder contains model files, training code, and testing software. The testing software is used to view the model's 3D reconstruction results, while the training code is used to train our model using paired clear and blurred images, with the clear images serving as prior reference knowledge. Our code is written in MATLAB.

## Reconstruction Results of Targets at Different Concentrations
We achieved 3D reconstruction of the target in milk turbid liquids with concentrations of 0.16%, 0.24%, and 0.40%. Our experimental results demonstrated excellent performance.
<div align="center">
  <img src="https://github.com/fyj0202/USI/blob/main/figure2.png" width="80%">
</div> <br />
