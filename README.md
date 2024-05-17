# Xray Abnormalities Detection
## Table of Contents
- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Preprocessing](#preprocessing)
- [Project Ideology](#project-ideology)
- [Models and Results](#models-and-results)
  - [Classifier](#classifier)
  - [Detection Model](#detection-model)
- [Deployment](#deployment)

## Introduction

X-ray image analysis is one of the most challenging tasks in medical sciences. Even experts sometimes fail to accurately identify diseases. This project aims to create an assistance tool for radiologists to improve the accuracy and efficiency of disease detection in X-ray images.

## Dataset Description

The dataset used in this project was released by VingBigData, which is a subset of the original dataset created by Stanford University. It includes:
- **Training Set:** 15,000 images
- **Testing Set:** 3,000 images
- **Disease Distribution:** Approximately 11,000 images with no findings, and the rest with one of 14 disease classes including aortic enlargement and pneumothorax.
- **File Format:** DICOM
- **Annotations:** Provided by 17 radiologists, with a focus on annotations from 3 radiologists who had the maximum annotations. We primarily used annotations from R9 (the radiologist with the maximum annotations).

## Preprocessing

The following preprocessing steps were applied to the dataset:
- **CLAHE (Contrast Limited Adaptive Histogram Equalization):** To improve the contrast of the images.
- **Normalization:** To standardize the pixel values.
- **Conversion:** DICOM files were converted to JPEG format.
- **Resizing:** Images were resized for uniform input dimensions.
- **Exclusion of Patient Identifiers (PI):** No patient identifiers were considered to maintain privacy.

## Project Ideology

### Initial Approach
- **Direct Detection Model:** Applied directly to the full dataset but yielded poor results, biased towards images with no findings.

### Improved Approach
- **Dual-Stage Approach:** Implemented a combination of a classifier and a detector to improve accuracy.

## Models and Results

### Classifier

1. **DenseNet:**
   - **Validation Accuracy:** 75%
   - **Precision and Recall:** Approximately 50%

2. **Stack Model (MobileNetV2 + DenseNet):**
   - **Test Accuracy:** 89%
   - **False Negatives:** 69

3. **DaVit (by Microsoft):**
   - **Training:** Pre-trained on ImageNet-1k, further trained on our dataset.
   - **Accuracy:** 92%
   - **False Negatives:** 52

### Detection Model

1. **YOLOv8 Detection X Model:**
   - **Initial mAP (Mean Average Precision) 50-95:** 15%

2. **With Augmentation (rotation, brightness, contrast adjustments):**
   - **mAP:** 33% after 100 epochs
   - **Optimizer:** Adam
   - **Learning Rate:** 3e-4

## Deployment

The model was deployed using Streamlit, providing an interactive web interface for radiologists to upload and analyze X-ray images.


## Video Demonstration

<video width="320" height="240" controls>
  <source src="https://github.com/k-Rohit/Vinbig-Lungs-Xray/assets/93335681/9fbba4f9-a0dd-4625-a765-6c7aaf79bded.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

https://github.com/k-Rohit/Vinbig-Lungs-Xray/assets/93335681/9fbba4f9-a0dd-4625-a765-6c7aaf79bded

