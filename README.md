# Implementation of *[ICLR2021 Oral] Free Lunch for Few-shot Learning: Distribution Calibration* for Chinese Text Classification

paper link: https://arxiv.org/abs/2101.06395

**Original work is validated in image classification. This project implements the algorithm for Chinese text classification**

## Overview

### Data

The data file *data.txt* contains 200,000 Chinese new titles in 10 classes. 7 classes (Finance and Economics(财经), Real Estate(房产), Spots(体育), Entertainment(娱乐), Science and Technology(科技), Society(社会), Computer Games(游戏), are selected as base classes; 3 classes (Stock(股票), Current Politics(时政), Education(教育)) are selected as novel classes

### Backbone

A tiny version of ALBERT for Chinese is chosen as the feature extractor:

https://huggingface.co/clue/albert_chinese_tiny/tree/main



## Usage

1. Download pretrained weights files to 'pretrained_weights' folder

2. Split and preprocess the dataset, run:

   ```bash
   python data_preprocess.py
   ```

3. Fine-tune backbone, extract features and evaluate distribution calibration, run:

   ```bash
   python main.py
   ```

   