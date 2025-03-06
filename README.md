# PrePATH: A Toolkit for Preprocessing Whole Slide Images 
This toolkit is built on [CLAM](https://github.com/mahmoodlab/CLAM) and [Aslide](https://github.com/MrPeterJin/ASlide).

## Step 1: Patching
We need to find the coordinates of patches with foreground in the WSI

```bash
# segment the tissue and get the coors, see the shell script for details
bash scripts/get_coors/SAL/sal.sh
```
## Step 2: Extracting features
```bash
# extract features, see scripts for details
bash scripts/extract_feature/exe.sh
```

## Supported Foundation Models (patch-level feature extractors)
If you want to extract feature using **ResNet50** and **GPFM**, set `models="resnet50 gpfm"` in the `script/extract_feature/exe.sh`  
* **ResNet50 (resnet50)**
* **GPFM (gpfm)** (https://github.com/birkhoffkiki/GPFM)
* **CTransPath (ctranspath)** (https://github.com/Xiyue-Wang/TransPath)
* **PLIP (plip)** (https://github.com/PathologyFoundation/plip)
* **CONCH (conch)** (https://huggingface.co/MahmoodLab/CONCH)
* **CONCH-1.5 (conch15)** (https://huggingface.co/MahmoodLab/conchv1_5)
* **UNI (uni)** (https://huggingface.co/MahmoodLab/UNI)
* **UNI-2 (uni2)** (https://huggingface.co/MahmoodLab/UNI2-h)
* **mSTAR (mstar)** (https://github.com/Innse/mSTAR)
* **Phikon (phikon)** (https://huggingface.co/owkin/phikon)
* **Phikon2 (phikon2)** (https://huggingface.co/owkin/phikon-v2)
* **Virchow-2 (virchow2)** (https://huggingface.co/paige-ai/Virchow2)
* **Prov-Gigapath (gigapath)** (https://huggingface.co/prov-gigapath/prov-gigapath)
* **CHIEF (chief)** (https://github.com/hms-dbmi/CHIEF/tree/main)
* **H-Optimus-0 (h-optimus-0)** (https://huggingface.co/bioptimus/H-optimus-0)
* **H-Optimus-1 (h-optimus-1)** (https://huggingface.co/bioptimus/H-optimus-1)
* **Lunit (lunit)** (https://github.com/lunit-io/benchmark-ssl-pathology) 
* **Hibou-L (hibou-l)** (https://github.com/HistAI/hibou)
* **MUSK (musk)** (https://huggingface.co/xiangjx/musk)

## Supported WSI formats
* kfb
* all format supported by `openslide`
* sdpc

