#!/bin/bash

# --- You Can Change Following Parameters ----
TASK_NAME=Test_Task   # Task name, change it as you want
wsi_dir=/jhcnas3/Pathology/code/PrePath/temp/svs  # The directory where the WSI files are stored
slide_ext=.svs  # The extension of the WSI files, remeber to keep the `.` in front
feat_dir=/jhcnas3/Pathology/code/PrePath/temp/patches #path to save feature
coors_dir=/jhcnas3/Pathology/code/PrePath/temp/patches  # path where the coors files are saved
models="gpfm" # foundation models to be used

split_number=1  # split the data into how many parts, for parallel computing
GPU_LIST="7" # GPU IDs you want to use, separated by space

batch_size=32
# python envs, define diffent envs for different machines
# PLEASE UPDATE THE PYTHON ENVIRONMENT PATHS, you can use `which python` to get the path
source scripts/extract_feature/python_envs/sal.sh
# --------------------------------------------
# GPU threhsold, the memory threshold for each model
# The memory threshold is the minimum free memory required to run the model
declare -A MEMORY_THRESHOLD
MEMORY_THRESHOLD["resnet50"]=1600
MEMORY_THRESHOLD["gpfm"]=4000
MEMORY_THRESHOLD["phikon"]=2000
MEMORY_THRESHOLD["phikon2"]=2000
MEMORY_THRESHOLD["plip"]=2000
MEMORY_THRESHOLD["uni"]=2000
MEMORY_THRESHOLD["uni2"]=2000
MEMORY_THRESHOLD["mstar"]=4000
MEMORY_THRESHOLD['chief']=1600
MEMORY_THRESHOLD['gigapath']=6200
MEMORY_THRESHOLD['virchow2']=6200
MEMORY_THRESHOLD['virchow']=6200
MEMORY_THRESHOLD["ctranspath"]=1600
MEMORY_THRESHOLD["conch"]=4000
MEMORY_THRESHOLD["conch15"]=4000
MEMORY_THRESHOLD["h-optimus-0"]=4000
MEMORY_THRESHOLD["h0-mini"]=2000
MEMORY_THRESHOLD["h-optimus-1"]=4000
MEMORY_THRESHOLD["lunit"]=4000
MEMORY_THRESHOLD["musk"]=4000
MEMORY_THRESHOLD["hibou-l"]=4000
# ---------------------------------------------


# ----DO NOT CHANGE THE FOLLOWING CODE----
csv_path=csv/$TASK_NAME
log_dir=scripts/extract_feature/logs
progress_log_file=scripts/extract_feature/logs/Progress_$TASK_NAME.log
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
        echo ">> $my_date | Part:$part | Model:$model | GPU:$selected_gpu | available memory:${max_free}MiB" >> $progress_log_file
        
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
        echo "  $my_date | No GPU availabel ${model}（need${threshold}MiB）" >> $progress_log_file
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
        echo "== ALL TASK DONE ==" >> $progress_log_file
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
                    echo ">> Done $model part$part | $my_date" >> $progress_log_file
                # 检查进程是否存在
                elif ! pgrep -f "extract_features_fp_fast.py --model $model --csv_path.*part_$part.csv" > /dev/null; then
                    tasks["$part-$model"]=0
                    my_date=$(date +%c)
                    echo "!! Process stoped abnormlly $model part$part | $my_date" >> $progress_log_file
                fi
            fi
        done
    done

done