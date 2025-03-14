save_dir="/jhcnas3/Pathology/code/PrePath/temp"
wsi_dir="/jhcnas5/Pathology/SAL/1"
wsi_format="svs"
log_name="SAL.log"


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