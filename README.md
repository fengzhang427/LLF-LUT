# Lookup Table meets Local Laplacian Filter: Pyramid Reconstruction Network for Tone Mapping (NeurIPS 2023)

**Technical Report**: <a href='https://arxiv.org/abs/2310.17190'><img src='https://img.shields.io/badge/paper-PDF-green'></a>

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

HDR+ (Original Size (4K)): [download (37 GB)](https://hdrplusdata.org/dataset.html) ; (480p)[download (1.38 GB)](https://drive.google.com/drive/folders/1Y1Rv3uGiJkP6CIrNTSKxPn1p-WFAc48a)

MIT-Adobe FiveK (Original Size (4K)):  [download (50 GB)](https://data.csail.mit.edu/graphics/fivek/) ; (480p)[download (12.51 GB)](https://drive.google.com/drive/folders/1Y1Rv3uGiJkP6CIrNTSKxPn1p-WFAc48a)

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
The code and the checkpoints will release soon.

## :car:Run
The code and the checkpoints will release soon.

## :book: Citation
If you find our LLF-LUT model useful for you, please consider citing :mega:
```bibtex
@misc{zhang2023lookup,
      title={Lookup Table meets Local Laplacian Filter: Pyramid Reconstruction Network for Tone Mapping}, 
      author={Feng Zhang and Ming Tian and Zhiqiang Li and Bin Xu and Qingbo Lu and Changxin Gao and Nong Sang},
      year={2023},
      eprint={2310.17190},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## :email:Contact
If you have any question, feel free to email fengzhangaia@hust.edu.cn.
