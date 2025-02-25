save_dir="/jhcnas3/Pathology/code/PrePath/temp"
source_dir="/jhcnas5/Pathology/SAL/1"
wsi_format="svs"
patch_size=512
log_name="SAL.log"

# nohup python create_patches_fp.py \
#         --source $source_dir \
#         --save_dir $save_dir\
#         --preset tcga.csv \
#         --patch_level 0 \
#         --patch_size $patch_size \
#         --step_size $patch_size \
#         --wsi_format $wsi_format \
#         --seg \
#         --patch \
#         --stitch > $log_name 2>&1 &

python create_patches_fp.py \
        --source $source_dir \
        --save_dir $save_dir\
        --preset tcga.csv \
        --patch_level 0 \
        --patch_size $patch_size \
        --step_size $patch_size \
        --wsi_format $wsi_format \
        --seg \
        --patch \
        --stitch