# Automatic Parameter Tuning for Plug-and-Play Algorithms Using Generalized Cross Validation and Stein's Unbiased Risk Estimation for Linear Inverse Problems in Computational Imaging

[Canberk Ekmekci](https://cekmekci.github.io/) and Mujdat Cetin

[SDIS Lab](https://labsites.rochester.edu/sdis), Rochester, NY USA

This repo contains a demo illustrating the use of the proposed GCV-based tuning method for MRI reconstruction. 

## Paper

**Abstract:** We propose two automatic parameter tuning methods for Plug-and-Play (PnP) algorithms that use CNN denoisers. We focus on linear inverse problems and propose an iterative algorithm to calculate generalized cross-validation (GCV) and Stein?s unbiased risk estimator (SURE) functions for a half-quadratic splitting-based PnP (PnP-HQS) algorithm that uses a state-of- the-art CNN denoiser. The proposed methods leverage forward mode automatic differentiation to calculate the GCV and SURE functions and tune the parameters of a PnP-HQS algorithm automatically by minimizing the GCV and SURE functions using grid search. Because linear inverse problems appear frequently in computational imaging, the proposed methods can be applied in various domains. Furthermore, because the proposed methods rely on GCV and SURE functions, they do not require access to the ground truth image and do not require collecting an additional training dataset, which is highly desirable for imaging applications for which acquiring data is costly and time-consuming. We evaluate the performance of the proposed methods on deblurring and MRI experiments and show that the GCV-based proposed method achieves comparable performance to that of the oracle tuning method that adjusts the parameters by maximizing the structural similarity index between the ground truth image and the output of the PnP algorithm. We also show that the SURE-based proposed method often leads to worse performance compared to the GCV-based proposed method.

[[Paper Link]](https://library.imaging.org/ei/articles/35/14/COIMG-170)

## Requirements

- PyTorch 1.13.1

## Data and Denoiser

- The MRI data used in the demo is obtained from the [IXI Dataset](https://brain-development.org/ixi-dataset/).

- The CNN denoiser used in the demo is the [DRUNet](https://ieeexplore.ieee.org/abstract/document/9454311) denoiser. The code for the DRUNet is obtained from the [DPIR](https://github.com/cszn/DPIR) repo and put in "/models/". The required MIT licence notice from [DPIR](https://github.com/cszn/DPIR) is also provided in "/models/".

## Use

Before running the demo, please download the pre-trained weights of the denoiser ("drunet_gray.pth") from [DPIR](https://github.com/cszn/DPIR/tree/master/model_zoo) repo and place it inside the "/models/" folder. Then, the demo can be run by running the "main_mri_demo.py" file. With the default parameters, the runtime is around 1 hour on a Tesla T4 GPU.


## Note

This repository contains the "revised" version of the code used to run the experiments in the published manuscript because the previous version of the code was held together with duct tape. You will be better off using the implementation provided in this repo. If you have any problems or questions, please feel free to contact me via [e-mail](https://cekmekci.github.io/contact/). 

## Citation (TBD)
----------
If you use the code for your research, please consider citing the following paper:

```BibTex
 @inproceedings{Ekmekci2023PnPTuning,
   title={Automatic Parameter Tuning for {P}lug-and-{P}lay Algorithms Using Generalized Cross Validation and {S}tein's Unbiased Risk Estimation for Linear Inverse Problems in Computational Imaging},
   author={Ekmekci, Canberk and Cetin, Mujdat},
   booktitle={Electronic Imaging 2023},
   pages={170-1--170-6},
   year={2023},
 }
```



