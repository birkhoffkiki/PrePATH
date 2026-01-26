# configuration

# enable color correction, remove if you do not need this
export COLOR_CORRECTION_FLAG="TRUE"
# threshold to drop slide if too many corrupted tiles (the slide will be skipped if the ratio of corrupted tiles is higher than this threshold)
export DROP_SLIDE_THRESHOLD=0.1

export OPENCV_IO_MAX_IMAGE_PIXELS=10995116277760


wsi_root="/jhcnas5/jmabq/Pathology/PWH/garywsi_he"
wsi_format="svs"
log_path="crop_image_scripts/garywsi_he.log"
root=/jhcnas5/jmabq/Pathology/PWH/Patches/garywsi_he
cpu_cores=48
h5_root=$root"/patches"
save_root=$root"/images"

nohup python extract_images_and_pack2h5.py \
        --wsi_format $wsi_format \
        --cpu_cores $cpu_cores \
        --h5_root $h5_root \
        --save_root $save_root \
        --wsi_root $wsi_root > $log_path 2>&1 &