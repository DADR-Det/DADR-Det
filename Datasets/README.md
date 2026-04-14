# Datasets

This directory contains dataset-related utilities and configuration files used in this project.
The DIOR-R dataset is available at DIOR and DIOR-R datasets
https://www.kaggle.com/datasets/redzapdos123/dior-r-dataset-yolov11-obb-format

The HRSC dataset is available at https://www.kaggle.com/datasets/guofeng/hrsc2016?utm_source=chatgpt.com
## Contents

- `DIOR-R Filtered Subset.py`  
  Script for filtering the original DIOR-R dataset and keeping only three categories:
  - airplane
  - vehicle
  - ship

  The script also remaps the original class ids to new ids:
  - airplane -> 0
  - vehicle -> 1
  - ship -> 2

- `data.yaml`  
  Dataset configuration file for training and validation.

## Filtered Dataset

The filtered dataset keeps only images that contain at least one object from the selected categories.  
Images without these three categories are discarded.

## Output Class Mapping

```text
0: airplane
1: vehicle
2: ship