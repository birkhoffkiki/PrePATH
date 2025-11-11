save_dir="/jhcnas3/Pathology/code/PrePath/temp"
wsi_dir="/data2/xyx/data/NACT/GZFPH/zssy-NAC影像+穿刺病理（n=118）/新辅助WSI/"
wsi_format="sdpc"
patch_size=512
log_name="sdpc.log"

# to set the patch size, please set it at `configs/resolution.py`

python create_patches_fp.py \
        --source $wsi_dir \
        --save_dir $save_dir\
        --preset tcga.csv \
        --patch_level 0 \
        --wsi_format $wsi_format \
        --seg \
        --patch 
        # --stitch