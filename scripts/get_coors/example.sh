# Most data are 20x
save_dir="./temp_results"
wsi_dir="/jhcnas7/Pathology/original_data/Breast/ACROBAT2023/tiff"
wsi_format="tiff"


# ====== Segmentation with DeepLabV3+ ======
# recommended for better tissue segmentation
# ==========================================

# remove following env if you do not want to use AI segmentation
# export CUDA_VISIBLE_DEVICES=7
# export ENABLE_AI_SEGMENTATION=1
# # set a proper downsample rate for segmentation to speed up
# export DOWNSAMPLE_FOR_SEGMENTATION=64
# # set a proper confidence threshold for segmentation
# export AI_SEG_CONFIDENCE_THRESHOLD=0.35
# # define batch size for segmentation
# export AI_SEG_BATCH_SIZE=48 
# ===========================================


# to set the patch size, please set it at `configs/resolution.py`
python create_patches_fp.py \
        --source "$wsi_dir" \
        --save_dir "$save_dir" \
        --preset tcga.csv \
        --patch_level 0 \
        --wsi_format "$wsi_format" \
        --seg \
        --patch \
        --stitch