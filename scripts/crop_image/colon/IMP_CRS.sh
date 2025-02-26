# configuration
export OPENCV_IO_MAX_IMAGE_PIXELS=10995116277760
wsi_dir=/jhcnas5/jmabq/Pathology/IMP-CRS-2024
wsi_format=svs
log_path=scripts/crop_image/logs
coor_root=/jhcnas4/Pathology/Patches/IMP-CRS-2024
save_dir=/mnt/hdd1/jmabq/IMP-CRS-2024/images
datatype="auto"
level=0
size=512
cpu_cores=48


h5_dir=$coor_root"/patches"

python extract_images.py \
        --datatype $datatype \
        --wsi_format $wsi_format \
        --level $level \
        --cpu_cores $cpu_cores \
        --h5_root $h5_dir \
        --save_root $save_dir \
        --wsi_root $wsi_dir > IMP-CRS.log
