image_folder=/mnt/hdd1/jmabq/IMP-CRS-2024/images
target_folder=/scratch/vcompath/Colon/Patches/IMP-CRS-2024
cache_folder=/mnt/hdd1/jmabq/IMP-CRS-2024
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