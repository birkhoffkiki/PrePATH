export LD_LIBRARY_PATH=wsi_core/Aslide/kfb/lib:$LD_LIBRARY_PATH # kfb file support
export LD_LIBRARY_PATH=wsi_core/Aslide/sdpc/so:wsi_core/Aslide/sdpc/so/ffmpeg:wsi_core/Aslide/sdpc/so/jpeg:$LD_LIBRARY_PATH # sdpc file support

save_dir="/jhcnas3/Pathology/code/PrePath/temp"
wsi_dir="/data2/xyx/data/NACT/GZFPH/zssy-NAC影像+穿刺病理（n=118）/新辅助WSI/"
wsi_format="sdpc"
patch_size=512
log_name="sdpc.log"

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
        --source $wsi_dir \
        --save_dir $save_dir\
        --preset tcga.csv \
        --patch_level 0 \
        --patch_size $patch_size \
        --step_size $patch_size \
        --wsi_format $wsi_format \
        --seg \
        --patch 
        # --stitch