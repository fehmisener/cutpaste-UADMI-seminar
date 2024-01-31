# UADMI Seminar Project - CutPaste Implementation

This project is the implementation of ["CutPaste: Self-Supervised Learning for Anomaly Detection and Localization" ](https://arxiv.org/abs/2104.04015) within the scope of the Master-Seminar: Unsupervised Anomaly Detection in Medical Imaging (IN2107, IN45010) seminar course.

## Project Setup

First you have to initialize environment

> **Note**
> Using virtual environment would be better

```bash
pip install -r requirements.txt
```

Download and extract the data

```bash
wget <link of the data>
unzip data.zip
```

## Results

The tests were conducted based on the parameters outlined in the table below.

| **Epoch Count** | **Batch Size** | **Learning Rate** | **Momentum** | **Input Size** | **Weight Decay** | **Algorithm**  |
|-----------------|----------------|-------------------|--------------|----------------|------------------|----------------|
| 256             | 96             | 0.03              | 0.9          | 256x256        | 0.00003          | 3-Way CutPaste |

![Algorithm Evaluation Results](results/evaluation_results.png "Algorithm Evaluation Results")

Comprehensive ROC Curve figures and detailed results for each pathology are available in the [results folder](results/).


