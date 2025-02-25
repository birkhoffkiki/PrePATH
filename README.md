# PrePath: A Toolkit for Preprocess Whole Slide Images 

# Step 1: Patching
We need to find the coordinates of patches with foreground in the WSI

```bash
# segment the tissue and get the coors, see the shell script for details
bash scripts/get_coors/SAL/sal.sh
```
# Step 2: Extracting features
```bash
# extract features, see scripts for details
bash scripts/extract_feature/exe.sh
```
# Suppored Foundation Models
* resnet50
* gpfm
* ctranspath
* plip
* conch
* uni
* mstar
* phikon
* virchow2
* gigapath
* chief
* h-optimus-0

