
image_folder=/jhcnas5/Pathology/ZhongShanLiuYuan/images/liuyuan_II_slide_svs/images
target_folder=/scratch/vcompath/Colon/Patches/liuyuan_II_slide_svs
cache_folder=/mnt/hdd2/liuyuan_II_slides_svs
hostname=superpod.ust.hk
port=22
username=jmabq
keyfile=/home/jmabq/.ssh/id_rsa
processes=512


python pack_and_upload.py \
    --image_folder $image_folder \
    --target_folder $target_folder \
    --cache_folder $cache_folder \
    --hostname $hostname \
    --port $port \
    --username $username \
    --key_filename $keyfile \
    --processes $processes