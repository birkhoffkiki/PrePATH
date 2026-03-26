# PrePATH: A Toolkit for Preprocessing Whole Slide Images

<p align="center">
	<img src="assets/prepath_logo.svg" alt="PrePath logo" width="680" />
</p>

<hr>

<div align="center" style="line-height: 1.6;">
	<a href="https://pathbench.org" target="_blank"><img alt="Live Benchmark" src="https://img.shields.io/badge/🌐_Live_Benchmark-pathbench.org-4A90E2?style=for-the-badge"/></a>
	<a href="https://github.com/birkhoffkiki/PathBench-MIL" target="_blank"><img alt="PathBench-MIL" src="https://img.shields.io/badge/PathBench--MIL-Repository-181717?style=for-the-badge&logo=github"/></a>
	<a href="https://arxiv.org/abs/2505.20202" target="_blank"><img alt="Paper" src="https://img.shields.io/badge/📄_Paper-arXiv-B31B1B?style=for-the-badge"/></a>
</div>

<br/>

> [!TIP]
> **🚀 Contribute Your Foundation Model!**
> We welcome submissions of new pathology foundation models to our benchmark.
> 👉 **[Submit Your Model Here](documents/SUBMIT_MODEL.md)** — Help advance the field by adding your model to PrePATH!

<br/>

PrePATH is a comprehensive preprocessing toolkit for whole slide images (WSI), built upon [CLAM](https://github.com/mahmoodlab/CLAM) and [ASlide](https://github.com/MrPeterJin/ASlide).

## TODO 
- H0-mini
- OpenMidnight
- GenBio-PathFM
- TITAN (Slide level)

## Installation

### Prerequisites
- Anaconda or Miniconda
- `openslide-tools` (system dependency)

### Setup Instructions

The following instructions demonstrate installation for the GPFM model. For other foundation models, please refer to their respective repositories for environment-specific requirements.

```bash
git clone https://github.com/birkhoffkiki/PrePATH.git
cd PrePATH
conda create --name gpfm python=3.10
conda activate gpfm
pip install -r requirements/gpfm.txt
cd models/ckpts/
wget https://github.com/birkhoffkiki/GPFM/releases/download/ckpt/GPFM.pth
```

**Notes:**
- ASlide should be installed as a Python package from [GitHub](https://github.com/MrPeterJin/ASlide) and is included in `requirements/gpfm.txt`.
- Environment configurations for other foundation models should be referenced from their respective repositories.

## Usage

### ⚡Using PrePATH to extract Patch-Level Features

#### Step 1: Coordinate Extraction
Extract coordinates of foreground patches from whole slide images:

```bash
# Configure variables in the script before execution
bash scripts/get_coors/example.sh
```

#### Step 2: Feature Extraction

Extract patch-level features using the selected foundation model:
```bash
# Refer to the script for detailed configuration options
bash scripts/extract_feature/one_gpu_example.sh
```
If you have multiple GPUs, you can use the `exe.sh` script for parallel processing:

```bash
bash scripts/extract_feature/exe.sh
```
#### Step 3: (Optional) Extract patches and pack them into HDF5 files
This is useful for pretraining or if you meet the `Corrupt JPEG data` error during feature extraction.  
This may happen for `kfb` or `sdpc` images due to limited support in multiprocessing.  
```bash
# Refer to the script for detailed configuration options
bash scripts/crop_image/example_packed2h5.sh
```
#### Step 4: (Optional) Extract features from HDF5 packed patches
If you have packed patches into HDF5 files in Step 3, you can extract features from them directly:
```bash
# Refer to the script for detailed configuration options
bash scripts/extract_feature/one_gpu_from_h5_example.sh
```

### ⚡Extract patches directly without feature extraction (e.g., for pretraining)
#### Step 1: Coordinate Extraction
Extract coordinates of foreground patches from whole slide images:
```bash
# Configure variables in the script before execution
bash scripts/get_coors/example.sh
```
#### Step 2: Patch Extraction
Extract patches based on the coordinates:  
We strongly recommend packing all patches using the HDF5 method for efficient storage and retrieval.
```bash
# Refer to the script for detailed configuration options
bash scripts/crop_image/example_packed2h5.sh
```

## Supported Foundation Models

**Note:** Each foundation model requires its corresponding Python environment to be properly configured.
| Model | Identifier | Reference |
|-------|------------|-----------|
| ResNet50 | `resnet50` | Standard ImageNet pretrained model |
| GPFM | `gpfm` | [GitHub](https://github.com/birkhoffkiki/GPFM) |
| CTransPath | `ctranspath` | [GitHub](https://github.com/Xiyue-Wang/TransPath) |
| PLIP | `plip` | [GitHub](https://github.com/PathologyFoundation/plip) |
| CONCH | `conch` | [HuggingFace](https://huggingface.co/MahmoodLab/CONCH) |
| CONCH-1.5 | `conch15` | [HuggingFace](https://huggingface.co/MahmoodLab/conchv1_5) |
| UNI | `uni` | [HuggingFace](https://huggingface.co/MahmoodLab/UNI) |
| UNI-2 | `uni2` | [HuggingFace](https://huggingface.co/MahmoodLab/UNI2-h) |
| mSTAR | `mstar` | [GitHub](https://github.com/Innse/mSTAR) |
| Phikon | `phikon` | [HuggingFace](https://huggingface.co/owkin/phikon) |
| Phikon2 | `phikon2` | [HuggingFace](https://huggingface.co/owkin/phikon-v2) |
| Virchow-2 | `virchow2` | [HuggingFace](https://huggingface.co/paige-ai/Virchow2) |
| Prov-GigaPath | `gigapath` | [HuggingFace](https://huggingface.co/prov-gigapath/prov-gigapath) |
| CHIEF | `chief` | [GitHub](https://github.com/hms-dbmi/CHIEF/tree/main) |
| H-Optimus-0 | `h-optimus-0` | [HuggingFace](https://huggingface.co/bioptimus/H-optimus-0) |
| H0-mini | `h0-mini` | [HuggingFace](https://huggingface.co/bioptimus/H0-mini) |
| H-Optimus-1 | `h-optimus-1` | [HuggingFace](https://huggingface.co/bioptimus/H-optimus-1) |
| OpenMidnight | `openmidnight` | [HuggingFace](https://huggingface.co/SophontAI/OpenMidnight) |
| GenBio-PathFM | `genbio-pathfm` | [HuggingFace](https://huggingface.co/genbio-ai/genbio-pathfm) |
| Lunit | `lunit` | [GitHub](https://github.com/lunit-io/benchmark-ssl-pathology) |
| Hibou-L | `hibou-l` | [GitHub](https://github.com/HistAI/hibou) |
| MUSK | `musk` | [HuggingFace](https://huggingface.co/xiangjx/musk) |
| OmiCLIP | `omiclip` | [Github](https://github.com/GuangyuWangLab2021/Loki) |
| PathoCLIP | `pathoclip` | [Github](https://github.com/wenchuan-zhang/patho-r1) |
---

## Supported WSI Formats

PrePATH supports the following whole slide image formats:

- **KFB** (.kfb)
- **SDPC** (.sdpc)
- **TRON** (.tron)
- All formats supported by OpenSlide (including .svs, .tiff, .ndpi, .vms, .vmu, .scn, .mrxs, .tif, .bif, and others)

