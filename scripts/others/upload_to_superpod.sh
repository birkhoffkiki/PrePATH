data_root=/jhcnas4/wangyihui/colon_fm/haplox_colon/images
tar_temp_root=/mnt/hdd2/haplox_colon
target_root=/scratch/vcompath/Colon/Patches/haplox_colon
address=superpod.ust.hk
mkdir -p $tar_temp_root
max_jobs=16

# 打包函数
pack_slide() {
    slide_id=$1
    echo "Packing: $slide_id"
    zip -0 -r $tar_temp_root/$slide_id.zip $data_root/$slide_id
    echo "Uploading $slide_id to superpod..."
    rsync -avuP $tar_temp_root/$slide_id.zip jmabq@$address:$target_root
}

# 遍历所有 slide_id 并并行处理
for slide_id in $(ls $data_root); do
    pack_slide $slide_id &
    
    # 控制并行进程数
    if [[ $(jobs -r -p | wc -l) -ge $max_jobs ]]; then
        wait -n
    fi
done