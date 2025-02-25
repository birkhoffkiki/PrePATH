# configuration

wsi_root="/jhcnas5/jmabq/Pathology/NanfangHospital/WSIs/LargeModel"
wsi_format="svs"
log_path="crop_image_scripts/large_model.log"
root=/jhcnas5/jmabq/Pathology/NanfangHospital/LargeModel
datatype="auto"
level=0
size=1024
cpu_cores=48
h5_root=$root"/patches"
save_root=$root"/images"
export OPENCV_IO_MAX_IMAGE_PIXELS=10995116277760

nohup python extract_images.py \
        --datatype $datatype \
        --wsi_format $wsi_format \
        --level $level \
        --cpu_cores $cpu_cores \
        --h5_root $h5_root \
        --save_root $save_root \
        --wsi_root $wsi_root > $log_path 2>&1 &