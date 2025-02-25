# configuration
export OPENCV_IO_MAX_IMAGE_PIXELS=10995116277760
wsi_root="/jhcnas5/Pathology/ZhongShanZhongliu/frozen_slides/BD"
wsi_format="svs"
log_path="crop_image_scripts/"
coor_root=/jhcnas5/Pathology/ZhongShanZhongliu/Patches/BD
save_root=/jhcnas5/Pathology/ZhongShanZhongliu/Images
datatype="auto"
level=0
size=1024
cpu_cores=48


for dir in "$wsi_root"/*; do
        if [ -d "$dir" ]; then
                dir_name=$(basename "$dir")
                wsi_dir=$dir
                save_dir=$save_root/$dir_name
                echo "Source directory: $source_dir"
                echo "Save directory: $save_dir"

                h5_dir=$coor_root"/"$dir_name"/patches"
                save_dir=$save_root"/"$dir_name"/images"

                python extract_images.py \
                        --datatype $datatype \
                        --wsi_format $wsi_format \
                        --level $level \
                        --cpu_cores $cpu_cores \
                        --h5_root $h5_dir \
                        --save_root $save_dir \
                        --wsi_root $wsi_dir &> $log_path"/"$dir_name".log"

        fi
done