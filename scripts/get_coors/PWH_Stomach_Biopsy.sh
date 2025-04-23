# Most data are 20x
save_dir="/jhcnas4/Pathology/Patches/PWH_Stomach_Biopsy"
wsi_dir="/jhcnas5/Pathology/PWH/Stomach_Biopsy/Stomach_Biopsy"
wsi_format="svs;mrxs"
log_name="PWH.log"


# to set the patch size, please set it at `configs/resolution.py`
python create_patches_fp.py \
        --source $wsi_dir \
        --save_dir $save_dir\
        --preset tcga.csv \
        --patch_level 0 \
        --wsi_format $wsi_format \
        --seg \
        --patch \
        --stitch