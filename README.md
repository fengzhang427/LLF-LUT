# LLF-LUT(Lookup Table meets Local Laplacian Filter)

The implementation of NeurIPS 2023 paper "[Lookup Table meets Local Laplacian Filter: Pyramid Reconstruction Network for Tone Mapping](https://arxiv.org/abs/2310.17190)" and its journal (TPAMI) version "[High-resolution Photo Enhancement in Real-time: A Laplacian Pyramid Network](~)".

## :sparkles: News
* 2025/10/12: Release our pretrained models of TPAMI version at [GoogleDrive](https://drive.google.com/file/d/1eDxI9Y_VNz2euNymdNYLYA6w8WLX3A5e/view?usp=sharing) and [Baidudisk(code:qh5w)](https://pan.baidu.com/s/1AwiHVLF0xlezGOfGu3XhLA?pwd=qh5w)
* 2025/10/12: The comprehensive version of this work was accepted to *IEEE Transactions on Pattern Analysis and Machine Intelligence* (TPAMI)

## Highlights
<img width="400" alt="image" src='./asset/0215.gif'> <img width="400" alt="image" src='./asset/1224.gif'>
<img width="400" alt="image" src='./asset/1247.gif'> <img width="400" alt="image" src='./asset/1874.gif'>

🚀🚀 Welcome to the repo of **LLF-LUT** 🚀🚀 

LLF-LUT is an effective end-to-end framework for the **HDR image tone mapping** task performing global tone manipulation while preserving local edge details. Specifically, we build a lightweight transformer weight predictor on the bottom of the Laplacian pyramid to predict the pixel-level content-dependent weight maps. The input HDR image is trilinear interpolated using the basis 3D LUTs and then multiplied with weighted maps to generate a coarse LDR image. To preserve local edge details and reconstruct the image from the Laplacian pyramid faithfully, we propose an image-adaptive learnable local Laplacian filter (LLF) to refine the high-frequency components while minimizing the use of computationally expensive convolution in the high-resolution components for efficiency.

🛄🛄 Disclaimer 🛄🛄

"The disparities observed between the results of CLUT in our study and the original research can be attributed to differences in the fundamental tasks. Specifically, our study focuses on the transformation of 16-bit High Dynamic Range (HDR) images into 8-bit Low Dynamic Range (LDR) images. In contrast, the original paper primarily addressed 8-bit to 8-bit image enhancement. Furthermore, CLUT's parameter count stands at 952K in our paper, a result of the utilization of sLUT as the backbone for CLUT. Notably, when the backbone is modified to LUT, the parameter count is reduced to 292K."

## 🌟 Structure

The model architecture of LLF-LUT is shown below. Given an input 16-bit HDR image, we initially decompose it into an adaptive Laplacian pyramid, resulting in a collection of high-frequency components and a low-frequency image. The adaptive Laplacian pyramid employs a dynamic adjustment of the decomposition levels to match the resolution of the input image. This adaptive process ensures that the low-frequency image achieves a proximity of approximately 64 × 64 resolution. The described decomposition process possesses invertibility, allowing the original image to be reconstructed by incremental operations.

<img width="900" alt="image" src='./asset/framework.png'>

## :bookmark_tabs:Intallation
Download the HDR+ dataset and MIT-Adobe FiveK dataset at the following links:

HDR+ (Original Size (4K)): [download (37 GB)](https://hdrplusdata.org/dataset.html) [Baiduyun(code:vcha)](https://pan.baidu.com/s/18iuX4eoYc0CaMzeN9O-KvA); (480p)[download (1.38 GB)](https://drive.google.com/file/d/1w5pFeqBX1U5v6qA-OS9CMzEEhnJv-vbi/view?usp=sharing)

MIT-Adobe FiveK (Original Size (4K)):  [download (50 GB)](https://data.csail.mit.edu/graphics/fivek/) [Baidudisk(code:a9av)](https://pan.baidu.com/s/15_Fp-a8DLU6npjQt4iURvA ); (480p)[download (12.51 GB)](https://drive.google.com/file/d/1Z4krjqK8a_k5eBEXmOSQc2JS_5Ioc023/view?usp=sharing)

* Install the conda environment
```
conda create -n llf-lut python=3.9
conda activate llf-lut
```
* Install Pytorch
```commandline
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch
```
## :heavy_check_mark:Pretrained Models
Release our pretrained models at [GoogleDrive](https://drive.google.com/file/d/1eDxI9Y_VNz2euNymdNYLYA6w8WLX3A5e/view?usp=sharing) and [Baidudisk(code:qh5w)](https://pan.baidu.com/s/1AwiHVLF0xlezGOfGu3XhLA?pwd=qh5w)(TPAMI version pretrained model).
Due to company policies, we regret that we cannot release the code and pre-trained models for the NeurIPS version.

## :car:Run
The code and the checkpoints will release soon.


## :book: Citation
If you find our LLF-LUT model useful for you, please consider citing :mega:
```bibtex
@article{zhang2023lookup,
  title={Lookup table meets local laplacian filter: pyramid reconstruction network for tone mapping},
  author={Zhang, Feng and Tian, Ming and Li, Zhiqiang and Xu, Bin and Lu, Qingbo and Gao, Changxin and Sang, Nong},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={57558--57569},
  year={2023}
}
```

## :email:Contact
If you have any question, feel free to email fengzhangaia@gmail.com.
