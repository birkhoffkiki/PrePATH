save_dir="/jhcnas4/Pathology/Patches/Nanfang_CRC"
wsi_dir="/mnt/hdd2/CRC_survival"
wsi_format="svs"
log_name="Nanfang_CRC.log"


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