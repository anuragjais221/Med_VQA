<!-- ## Visual Question Answering in Medical Domain.

1. Run preprocess_data.ipynb to preprocess data which will save the images and questions in pickle format if already not present.
2. Execute extract_features.ipynb to save image features in secondory memory which can use in training later to make it faster.
3. Execute mixup.ipynb to generate mixed images and mixed dataset along with original which will store in secondory memory.
4. Execute mixed_exp.ipynb to train the model on original+mixed dataset and also for testing.
5.  -->

<div align="center">
<!-- <h1>Deep Learning based Approaches for Medical Visual Question Anseering using VisualBERT and Mixup</h1> -->
<h2><a href="MTP Report.pdf">Deep Learning based Approaches for Medical Visual Question Answering using VisualBERT and Mixup</a></h2>
    
[Anurag Jaiswal](https://github.com/anuragjais221)<sup></sup>,        [Dr. Deepti R. Bathula](https://www.iitrpr.ac.in/deepti-r-bathula)<sup></sup>
    
<sup></sup>[Indian Institute of Technology, Ropar](https://www.iitrpr.ac.in/)
</div>

<p align="center">
<img src="imgs/visualBERT.PNG" width=100% height=100% 
class="center">
</p>

We implement the visualBERT architecture with a data augmentation technique on ImageCLEF 2019 dataset and subset of 2020. We also implement the MFB with co-attention based architecture along with mixup technique as a baseline to compare with visualBERT based Model. Both models are trained on original dataset, mixed dataset and original+mixed dataset. 
We also used Contrastive Loss as an additional loss along with crossEntropy Loss which improves the test accuracy for VisualBERT based Model. The basic aim of Med_VQA is provide help to medical doctors and practioners to take decision related to medical images such as MRI,Xray-scans which can benefit society.



## Acknowledgement

I would like to thank my guide [Dr. Deepti R. Bathula](https://www.iitrpr.ac.in/deepti-r-bathula)<sup></sup> (Associate Professor, Indian Institute
of Technology, Ropar) for her continuous support and motivation throughout my project. I
would also like to thank [Mr. Abhishek Singh Sambyal](https://abhisheksambyal.github.io/)<sup></sup> (PhD Student, Computer Science and
Engineering Department, IIT Ropar) for his guidance.

<p align="right">(<a href="#top">back to top</a>)</p>