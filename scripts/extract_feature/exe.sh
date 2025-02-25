#!/bin/bash

# --- You Can Change Following Parameters ----
TASK_NAME=SAL1
wsi_dir=/jhcnas5/Pathology/SAL/1
slide_ext=.svs
feat_dir=/jhcnas3/Pathology/code/PrePath/temp #path to save feature
coors_dir=temp  # path where the coors files are saved
models="gpfm" # foundation models to be used
split_number=4  # 将数据集分为几个部分，并行处理
GPU_LIST="0 1 2 3 4 5 6 7" # 使用的GPU

batch_size=32
# python envs, define diffent envs for different machines
source scripts/extract_feature/python_envs/cpu5.sh
# --------------------------------------------
# GPU显存阈值 (单位: MiB)
declare -A MEMORY_THRESHOLD
MEMORY_THRESHOLD["resnet50"]=1600
MEMORY_THRESHOLD["gpfm"]=4000
MEMORY_THRESHOLD["phikon"]=2000
MEMORY_THRESHOLD["plip"]=2000
MEMORY_THRESHOLD["uni"]=2000
MEMORY_THRESHOLD["mstar"]=4000
MEMORY_THRESHOLD['chief']=1600
MEMORY_THRESHOLD['gigapath']=6200
MEMORY_THRESHOLD['virchow2']=6200
MEMORY_THRESHOLD["ctranspath"]=1600
MEMORY_THRESHOLD["conch"]=4000
# ---------------------------------------------


# ----DO NOT CHANGE THE FOLLOWING CODE----
csv_path=csv/$TASK_NAME
log_dir=scripts/extract_feature/logs
progress_log_file=scripts/extract_feature/logs/Progress_$TASK_NAME.log
export LD_LIBRARY_PATH=wsi_core/Aslide/kfb/lib:$LD_LIBRARY_PATH # kfb file support
export PYTHONPATH=.:$PYTHONPATH
# auto generate csv
echo "Automatic generating csv files: $split_number" >> $progress_log_file
python scripts/extract_feature/generate_csv.py --h5_dir $coors_dir/patches --num $split_number --root $csv_path
ls $csv_path >> $progress_log_file

# 0: 未启动 1: 运行中 2: 已完成
parts=($(seq 0 $((split_number - 1))))
declare -A tasks
for part in "${parts[@]}"; do
    for model in $models; do
        tasks["$part-$model"]=0
    done
done


check_and_run_tasks() {
    local part=$1
    local model=$2
    
    local selected_gpu=-1
    local max_free=0

    # 遍历所有GPU寻找最佳候选
    for gpu_index in $GPU_LIST; do
        local free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu_index | awk '{print $1}')
        local threshold=${MEMORY_THRESHOLD[$model]}
        
        if [ $free_memory -ge $threshold ] && [ $free_memory -gt $max_free ]; then
            selected_gpu=$gpu_index
            max_free=$free_memory
        fi
    done

    if [ $selected_gpu -ne -1 ]; then
        my_date=$(date +%c)
        echo ">> $my_date | Part:$part | Model:$model | GPU:$selected_gpu | 可用显存:${max_free}MiB" >> $progress_log_file
        
        # 设置GPU环境变量
        export CUDA_VISIBLE_DEVICES=$selected_gpu
        
        # 启动任务
        python_executable=${python_envs[$model]}
        nohup $python_executable extract_features_fp_fast.py \
            --model $model \
            --csv_path $csv_path/part_$part.csv \
            --data_coors_dir $coors_dir \
            --data_slide_dir $wsi_dir \
            --feat_dir $feat_dir \
            --ignore_partial yes \
            --batch_size $batch_size \
            --datatype auto \
            --slide_ext $slide_ext \
            --save_storage "yes" > $log_dir/${TASK_NAME}_${model}_${part}.log 2>&1 &
        
        # 记录任务状态
        tasks["$part-$model"]=1
        return 0
    else
        echo "  $my_date | 没有可用GPU满足${model}需求（需要${threshold}MiB）" >> $progress_log_file
        return 1
    fi
}

# 主任务循环
while true; do
    # 检查所有任务状态
    all_done=true
    for key in "${!tasks[@]}"; do
        if [ ${tasks[$key]} -ne 2 ]; then
            all_done=false
            break
        fi
    done

    if $all_done; then
        echo "== 所有任务已完成 ==" >> $progress_log_file
        break
    fi

    # 尝试启动新任务
    for part in "${parts[@]}"; do
        for model in $models; do
            if [ ${tasks["$part-$model"]} -eq 0 ]; then
                echo "try to start: $model part $part"
                check_and_run_tasks $part $model
                sleep 30  # 避免密集启动
            fi
        done
    done

    # 查运行中的任务状态
    for part in "${parts[@]}"; do
        for model in $models; do
            if [ ${tasks["$part-$model"]} -eq 1 ]; then
                # 通过日志判断是否完成
                log_file=$log_dir/${TASK_NAME}_${model}_${part}.log
                if [ -f $log_file ] && tail -n 1 $log_file | grep -q "Extracting end"; then
                    tasks["$part-$model"]=2
                    my_date=$(date +%c)
                    echo ">> 完成 $model part$part | $my_date" >> $progress_log_file
                # 检查进程是否存在
                elif ! pgrep -f "extract_features_fp_fast.py --model $model --csv_path.*part_$part.csv" > /dev/null; then
                    tasks["$part-$model"]=0
                    my_date=$(date +%c)
                    echo "!! 进程异常终止 $model part$part | $my_date" >> $progress_log_file
                fi
            fi
        done
    done

done