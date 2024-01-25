# CT Augmentation

## Introduction

Using U-Net model as basis to generate slices by nearby data to improve the performace of detection models in the latter stages.

## Table of Contents

```bash
/Data Augmentation
├── README.md
├── Final_Choose
│ ├── preprocess files
│ │ ├── z1_slice_count.csv (Count the slice number of all image files.)
│ │ ├── z4_all_nodule.csv (Nodule position.)
│ │ ├── z2_HU_slice_cut.ipynb (Proprocess CT slice dataset.)
│ │ └── z3_Mask_slice_cut_new.ipynb (Proprocess nodule mask dataset.)
│ ├── model parameters
│ │ ├── NoduleMask_best.ckpt (Nodule mask model best parameters trained.)
│ │ └── CTSlice_best.ckpt (CT Slice model best parameters trained.)
│ ├── NoduleMask_Training.ipynb (Nodule Mask model training.)
│ ├── NoduleMask_Testing.ipynb (Nodule Mask model testing.)
│ ├── CTSlice_Training.ipynb (CT Slice model training.)
│ ├── CTSlice_Testing.ipynb (CT Slice model testing.)
│ └── CTSlice_InserttoImage.ipynb (Insert Augmented CT Slice back to the original file.)
└── Experiments (Some experiments during training.)
```

## Getting Started

Please run the preprocess files first and then run the two training and testing files respectively to get the final result you need. Or you can just use the model parameters \*.ckpt to get the result you need.

### Prerequisites

```bash
# Import necessary packages.
!pip install numpy
!pip install pandas
!pip install torch torchvision
!pip install PIL

!pip install tqdm
!pip install matplotlib
!pip install SimpleITK as sitk
```
