import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===== 配置区域 =====
os.environ["LD_LIBRARY_PATH"] = "wsi_core/Aslide/sdpc/so:$LD_LIBRARY_PATH"
os.environ["LD_LIBRARY_PATH"] = "wsi_core/Aslide/kfb/lib:$LD_LIBRARY_PATH"
os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libffi.so.7"
# 基础路径配置
PYTHONPATH = "/data2/lbliao/Code/PrePATH"
PRESET_FILE = "/data2/lbliao/Code/PrePATH/presets/tcga.csv"
SCRIPT_PATH = "/data2/lbliao/Code/PrePATH/create_patches_fp.py"
LOG_NAME = "SAL.log"
WSI_DIR = "/NAS2/Data4/llb/中日友好医院结直肠癌数据/LS"
WSI_DIRS = []
for DIR in os.listdir(WSI_DIR):
    tmp = os.path.join(WSI_DIR, DIR)
    if os.path.isdir(tmp):
        WSI_DIRS.append(tmp)
# 基础保存路径
BASE_SAVE_DIR = "/NAS2/Data1/lbliao/Data/CRC/patches"

# 并发控制
MAX_WORKERS = 4  # 同时运行的最大线程数
# ===================

# 全局锁用于控制台输出同步
print_lock = threading.Lock()


def get_first_file_extension(directory):
    """获取指定目录下第一个文件的后缀名"""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            # 使用os.path.splitext获取后缀
            _, ext = os.path.splitext(filename)
            return ext.lower().lstrip('.')  # 返回不带点的后缀名
    return None


def process_wsi(wsi_dir):
    """处理单个WSI目录"""
    # 动态生成保存目录（BASE_SAVE_DIR + 最后一级目录名）
    last_dir_name = os.path.basename(wsi_dir.rstrip('/'))
    save_dir = os.path.join(BASE_SAVE_DIR, last_dir_name)

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 生成唯一日志文件名
    log_file = os.path.join(save_dir, f"{last_dir_name}_{LOG_NAME}")

    # 动态获取wsi_format（从第一个文件的后缀）
    wsi_format = get_first_file_extension(wsi_dir)
    if not wsi_format:
        error_msg = f"⚠️ 无法在目录中找到有效的文件: {wsi_dir}"
        return (False, wsi_dir, error_msg)

    # 构造命令参数
    cmd = [
        "python", SCRIPT_PATH,
        "--source", wsi_dir,
        "--save_dir", save_dir,
        "--preset", PRESET_FILE,
        "--patch_level", "0",
        "--wsi_format", wsi_format,
        "--seg",
        "--patch",
        "--stitch"
    ]

    # 执行命令并记录日志
    try:
        with print_lock:
            print(f"ℹ️ 开始处理: {last_dir_name} → 日志: {os.path.basename(log_file)}")
            print(f"  使用格式: {wsi_format}")

        with open(log_file, 'w') as log_f:
            # 使用Popen替代run，实现实时输出打印
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8'
            )

            # 实时读取输出并同时打印到控制台和日志文件
            while True:
                # 实时读取输出行
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break

                if line:
                    # 写入日志文件
                    log_f.write(line)
                    log_f.flush()  # 确保立即写入

                    # 打印到控制台（使用锁确保多线程安全）
                    with print_lock:
                        print(line, end='', flush=True)  # end=''避免重复换行

            # 等待进程结束并获取返回码
            returncode = proc.wait()

            # 检查返回码，模拟check=True的行为
            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, cmd)

        return (True, wsi_dir, f"✅ 处理成功 - 日志保存至: {log_file}")
    except subprocess.CalledProcessError as e:
        error_msg = f"❌ 命令执行失败(状态码:{e.returncode}): {e.cmd}\n详见日志: {log_file}"
        return (False, wsi_dir, error_msg)
    except subprocess.TimeoutExpired:
        error_msg = f"⏰ 命令执行超时: 已超过1小时\n详见日志: {log_file}"
        return (False, wsi_dir, error_msg)
    except Exception as e:
        error_msg = f"⚠️ 意外错误: {str(e)}"
        return (False, wsi_dir, error_msg)


if __name__ == "__main__":
    """主执行函数"""
    # 设置环境变量
    os.environ['PYTHONPATH'] = f"{PYTHONPATH}:{os.environ.get('PYTHONPATH', '')}"

    # 初始输出
    total = len(WSI_DIRS)
    print(f"🚀 开始处理 {total} 个WSI目录，并发数: {MAX_WORKERS}")
    print(f"基础保存目录: {BASE_SAVE_DIR}")
    print("=" * 60)

    success_count = 0
    failure_count = 0
    processed_count = 0

    # 使用线程池并发执行
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_wsi, wsi_dir): wsi_dir
                   for wsi_dir in WSI_DIRS}

        for future in as_completed(futures):
            processed_count += 1
            status, wsi_dir, message = future.result()

            with print_lock:
                print(f"[{processed_count}/{total}] {'✅' if status else '❌'} {os.path.basename(wsi_dir)}")
                print(f"  → {message}")
                print("-" * 60)

            if status:
                success_count += 1
            else:
                failure_count += 1

            # 实时显示进度
            with print_lock:
                print(f"📊 当前进度: 成功 {success_count} | 失败 {failure_count} | 剩余 {total - processed_count}")

    # 输出汇总报告
    print("\n" + "=" * 60)
    print(f"🎉 处理完成! 总计: {total} | 成功: {success_count} | 失败: {failure_count}")
    print("🔍 详细日志请查看各自保存目录下的日志文件")
    print("=" * 60)

    # 失败任务特别提示
    if failure_count > 0:
        print("\n⚠️ 注意: 有任务处理失败，请检查对应日志文件中的错误信息")
