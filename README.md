# NTUIMProject

This is an integrated lung cancer detection AI model built by 6 students from the department of Information Management at National Taiwan University.

## Datasets
Dataset from **LUng Nodule Analysis 2016(LUNA 16)**.

## Research Flow
### 1. Data Augmentation
   
Because the data are not enough, we use the **U-Net** model in the CT Slice Augmentation phase to augment the data.

### 2. Lung Segmentation

Divide into two ways to compare whether the Lung Segmentation has any influence.

* One apply the Lung Segmentation.
* The other didn't.

We use the **ResBCDU-Net** to perform the lung segmentation.

### 3. Nodule Processing

Divide into two ways to compare which processing works the best.

* One apply the **Nodule Segmentation** (**U-Det model**) first, and apply the **Nodule Classification** (**ResNet Model**) afterwards.
* The other apply **Nodule Detection** (**YOLO v8** & **Detectron2** (Faster R-CNN)).

## Result
Based on the comparison of the model results, both methods have their own advantages.

* The two models using Detection can achieve an accuracy of over **90%** in detecting tumors.
* The other group of models, U-Det + Classification, can also detect most tumors and provide more information than just bounding boxes.
* Data augmentation led to a substantial improvement in model accuracy.
* Lung segmentation allowed for more precise detection of tumors located at the edges. 
