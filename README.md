<!-- ## Visual Question Answering in Medical Domain.

1. Run preprocess_data.ipynb to preprocess data which will save the images and questions in pickle format if already not present.
2. Execute extract_features.ipynb to save image features in secondory memory which can use in training later to make it faster.
3. Execute mixup.ipynb to generate mixed images and mixed dataset along with original which will store in secondory memory.
4. Execute mixed_exp.ipynb to train the model on original+mixed dataset and also for testing.
5.  -->




<div align="center">
<!-- <h1>Predictive_Analysis_of_Parkinsons_Disease_from_Gait_Sensor_Data_and_Brain_MRI_Images: Predictive Analysis of Parkinsons Disease from Gait Sensor Data and Brain MRI Images</h1> -->
<h2><a href="MTP Report.pdf">Deep Learning based Approaches for Medical Visual Question Answering using Transformer and Mixup</a></h2>
    
[Anurag Jaiswal](https://github.com/arunava5764)<sup></sup>,        [Dr. Deepti R. Bathula](https://www.iitrpr.ac.in/deepti-r-bathula)<sup></sup>
    
<sup></sup>[Indian Institute of Technology, Ropar](https://www.iitrpr.ac.in/)
</div>

<p align="center">
<img src="Diagrams/siamese.PNG" width=100% height=80% 
class="center">
</p>

We implement the visualBERT architecture with a data augmentation technique on ImageCLEF 2019 dataset and subset of 2020. We also implement the MFB with co-attention based architecture along with mixup technique as a baseline to compare with visualBERT based Model. Both models are trained on original dataset, mixed dataset and original+mixed dataset. 
We also used Contrastive Loss as an additional loss along with crossEntropy Loss which improves the test accuracy for VisualBERT based Model. The basic aim of Med_VQA is provide help to medical doctors and practioners to take decision related to medical images such as MRI,Xray-scans which can benefit society.

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#File-Description">File Description</a></li>
    <li><a href="#Dataset Description">Datasets</a></li>
    <li><a href="#Steps-to-Reproduce Results">Steps to Reproduce Results</a></li>
    <li><a href="#Training-and-Validation">Training and Validation</a></li>
    <li><a href="#Test-Flow">Test Flow</a></li>
    <li><a href="#Comparison-Between-Multi-Modality-and-Baseline-Single-Modality">Comparison Between Multi-Modality and Baseline Single Modality</a></li>
    <li><a href="#Comaprison-With-Previous-PPMI-Experiment">Comaprison With Previous PPMI Experiment</a></li>
    <li><a href="#Comaprison-With-Previous-Gait-Experiment">Comaprison With Previous Gait Experimen</a></li>
    <li><a href="#acknowledgement">Acknowledgement</a></li>
  </ol>
</details>


## File Description

The content of each folder of this repository is described as follows.
- [x] **pre_processes_data.ipynb** This file contains the code for preprocessing the data and save them in pickle format.
- [x] **train_dataset_pickle** This folder contains the train data and train image feature in pickle format. 
- [x] **test_dataset_pickle** This folder contains the test images dataframe and test image features in pickle format.
- [x] **valid_dataset_pickle** This folder contains the validation images dataframe and valid image features in pickle format.
- [x] **data** This folder contains the ImageCLEF dataset.
- [x] **imgs** This folder contains all the images such as architecture image, loss and accuracy images for each experiments.
- [x] **extract_featues.ipynb** This notebook is used to extract features from images and save it in pickle format. 
- [x] **mixup.ipynb** This notebook will generate the mixed data and also prepare original_mixed data.
- [x] **mfb_trainexp.py** This script is used for train the model and also test on test data.

## Datasets

we used ImageCLEF 2019 dataset and subset of 2020 dataset. The dataset should reside in data folder.

## Steps to Reprodue the Results

To reproduce the results please follow the steps vlbelow.

- Download the `ImageCLEF datasest` from the sites if not present in data folder.
- Run `preprocess_data.ipynb` to preprocess data and save the data in pickle format save in train_dataset_pickle folder.
- Run `mixup.ipynb` to generate mixed images and save them in train_dataset_pickle format.
- Run `extract_features.ipynb` to extract features from images and save them in secondory memory which later use in training to make it faster.
- Run `mixed_exp.py` for training and testing.

## Training and Validation

We have prepared pickle file for training on original data,mixed data and original+mixed data.By providing the path of pickle data in `mixed_exp.py` file.
Similary for training on mfb base architecture provide path of pickle format and execute the `mfb_trainexp.py`.


<!-- <p align="center">
<img src="Diagrams/train_test_ppt.png" width=80% height=100% 
class="center">
</p>

<p align="center">
<img src="Diagrams/test_ppt.png" width=100% height=100% 
class="center">
</p> -->

## Experimental Result

So, if we consider our previous experiment on individual dataset using individual model it is evident that, proposed architecture has able to improve the accuracy along with Cohen Kappa score and ROC AUC value using those multi-modal features. The same scenario is described in the below comparison chart.

| Topic | Gait Model | PPMI Model | Multi-Model |
|---|:---:|:---:|:----:|
| Features | 128 Features | 1595 Features | 1721 Features |
| Parameters | 2214 | 211,852 | 218,676 |
| Loss Function | Sparse Categorical CrossEntropy | Sparse Categorical CrossEntropy | Triplet Loss |
| Optimizer | RMSprop | RMSprop | RMSprop |
| Accuracy | 89.61% | 92.13 | Gait : 98.7, PPMI : 99.11 |
| ROC AUC Score | 85% | 84% | Gait : 0.9782, PPMI : 0.9864 | 
| Cohen Kappa Score | 0.7310 | 0.74 | Gait : 0.9686, PPMI : 0.9613 | 



## Comparison between MFB with Co-attention base architecture & VisualBERT base model with Mixup

With the help of Mixup technique the VisualBERT model performes better on test data. Also use the contrastive loss as an additional loss with crossEntropy loss improved the accuracy further on test set.

## Comaprison With Previous Gait Experiment
While considering Gait dataset, the below table describes little bit different scenario compared to PPMI result. Recent studies have already got an highest accuracy over 99% and our experiment is almost similar to those results. Not only that, sensitivity, specificity and F1 Score, all these evaluation metrics are also near about similar to recent research studies. So we can say that as all the recent research experiment already achieved near to 100% accuracy that is why our framework does not achieved much more than that result. So there is no scope of improvement as of now for Gait Dataset.

| Authors | Accuracy | Sensitivity | Specificity | F1 Score | 
|-------|:--:|:--:|:--:|:--:|
| Zeng et al. (2016) [[Link](https://pubmed.ncbi.nlm.nih.gov/27693437/)<sup></sup>] | 98.8 | 98.92 | 98.63 | NR |
| Açici et al. (2017) [[Link](https://link.springer.com/chapter/10.1007/978-3-319-65172-9_51)<sup></sup>] | 98 | 99.1 | 95.7 | 0.98 |
| A¸suro˘glu et al. (2018) [[Link](https://linkinghub.elsevier.com/retrieve/pii/S0208521617304321)<sup></sup>] | 99 | 97.8 | 99.5 | NR |
| Zhao et al. (2018) [[Link](https://www.sciencedirect.com/science/article/abs/pii/S0925231218303242)<sup></sup>] | 98.61 | NR | NR | NR |
| Noella et al. (2019) [[Link](https://www.sciencedirect.com/science/article/pii/S1877050920300107)<sup></sup>] | 97 | NR | NR | NR |
| Veeraragavan et al. (2020) [[Link](https://pubmed.ncbi.nlm.nih.gov/33240106/)<sup></sup>] | 97.7 | 97.05 | 97.41 | 0.97 |
| Xia et al. (2020) [[Link](https://pubmed.ncbi.nlm.nih.gov/31603824/)<sup></sup>] | 99.07 | 99.1 | 99.01 | NR |
| Priya et al. (2020) [[Link](https://ieeexplore.ieee.org/abstract/document/9075785)<sup></sup>] | 98.82 | NR  | NR | NR |
| Ghaderyan and Fathi (2021) [[Link](https://www.sciencedirect.com/science/article/pii/S0263224121002591)<sup></sup>] | 97.22 | 98.22 | 95.86 | NR |
| Liu et al. (2021) [[Link](https://link.springer.com/article/10.1007/s10489-020-02182-5)<sup></sup>] | 99.22 | 98.04 | 100 | 0.99 |
| Tong et al. (2021) [[Link](https://www.mdpi.com/2076-3417/11/4/1834)<sup></sup>] | 99.23 | NR | NR | NR |
| Proposed Method | 99.2 | 98.52 | 100 | 0.99 |



## Acknowledgement

I would like to thank my guide [Dr. Deepti R. Bathula](https://www.iitrpr.ac.in/deepti-r-bathula)<sup></sup> (Associate Professor, Indian Institute
of Technology, Ropar) for her continuous support and motivation throughout my project. I
would also like to thank [Mr. Abhishek Singh Sambyal](https://abhisheksambyal.github.io/)<sup></sup> (PhD Student, Computer Science and
Engineering Department, IIT Ropar) for his guidance.

<p align="right">(<a href="#top">back to top</a>)</p>
