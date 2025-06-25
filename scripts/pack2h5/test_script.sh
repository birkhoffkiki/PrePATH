# configuration


wsi_dir=/home/jmabq/svs
h5_dir=/jhcnas3/Pathology/code/PrePath/temp/patches/patches
wsi_format=svs  # for multiple formats, use `svs;kfb`
save_dir=/jhcnas3/Pathology/code/PrePath/temp/packed_h5/temp


cpu_cores=48
export OPENCV_IO_MAX_IMAGE_PIXELS=10995116277760
export LD_LIBRARY_PATH=wsi_core/Aslide/kfb/lib:$LD_LIBRARY_PATH # kfb file support
export LD_LIBRARY_PATH=wsi_core/Aslide/sdpc/so:$LD_LIBRARY_PATH # sdpc file support
export PYTHONPATH=.:$PYTHONPATH


python extract_images_and_pack2h5.py \
        --wsi_format $wsi_format \
        --cpu_cores $cpu_cores \
        --h5_root $h5_dir \
        --save_root $save_dir \
        --wsi_root $wsi_dir
