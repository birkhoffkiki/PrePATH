# Most data are 20x
save_dir="/jhcnas4/Pathology/Patches/Nanfang_GC"
wsi_dir="/jhcnas5/Pathology/NanfangHospital/WSIs/GC"
wsi_format="svs"
log_name="Nanfang_Lung_GC.log"


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