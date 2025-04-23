import os
import sys
sys.path.append('/jhcnas3/Pathology/code/PrePath')
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from models import get_model, get_custom_transformer
import torch
import torch.nn as nn
from time import time
import warnings




def test_optimization(jit_model, torch_model, x ,
                     num_warmup=10, num_test=10, device="cuda"):
    # 基础配置
    results = {}
    
    # 原始模型基准结果
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time()
        ref_output = torch_model(x)
        torch.cuda.synchronize()

    test_models = {
        'jit': jit_model,
        'torch': torch_model
    }

    for name, optimized_model in test_models.items():
        # 初始化优化模型
        
        # 预热
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = optimized_model(x)
        
        # 时间测试
        torch.cuda.synchronize()
        start = time()
        for _ in range(num_test):
            with torch.no_grad():
                _ = optimized_model(x)
        avg_time = (time() - start) / num_test * 1000  # 毫秒/次
        torch.cuda.synchronize()
        
        # 精度测试
        with torch.no_grad():
            test_output = optimized_model(x)
            test_output = test_output.to(ref_output.dtype)  # 统一精度比较
            
        # 计算指标
        mean_abs = torch.mean(torch.abs(test_output - ref_output)).item()
        max_abs = torch.max(torch.abs(test_output - ref_output)).item()
        cos_sim = torch.cosine_similarity(
            test_output.flatten(), 
            ref_output.flatten(), 
            dim=0
        ).item()
        
        results[name] = {
            "time": avg_time,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "cos_sim": cos_sim
        }

    # 打印结果
    # 打印结果（新增加速比列）
    baseline_time = results["torch"]["time"]
    print(f"{'Method':<10} | {'Time(ms)':<10} | {'Speedup':<8} | {'Max ABS':<12} | {'Mean ABS':<12} | {'Cosine Sim.'}")
    print("-" * 65)
    
    
    # 打印优化结果
    for name, data in results.items():
        speedup = baseline_time / data["time"]
        print(f"{name:<10} | {data['time']:>8.3f}  | {speedup:>6.2f}x | {data['max_abs']:>10.4e} | {data['mean_abs']:>10.4e} | {data['cos_sim']:>6.4f}")


def read_data(model_name, device):
    import random
    from PIL import Image
    data_dir = 'test_data'
    file_names = os.listdir(data_dir)
    random.shuffle(file_names)
    # 读取前32个文件
    file_names = file_names[:32]
    images = []
    transform = get_custom_transformer(model_name)
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        image = Image.open(file_path).convert("RGB")
        image = transform(image)
        images.append(image)
    # 将图像堆叠成一个批次
    images = torch.stack(images)
    images = images.to(device)
    print("Image shape:", images.shape)
    return images
    

if __name__ == "__main__":
    # 创建示例模型（实际使用时替换为任意PyTorch模型）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = 'mstar'
    jit_model = get_model(model, device=device, gpu_num=1, jit=True)
    torch_model = get_model(model, device=device, gpu_num=1, jit=False)
    size = 224
    # 运行测试（建议在GPU环境下运行）
    data = read_data(model, device)
    if torch.cuda.is_available():
        test_optimization(jit_model, torch_model, data, device=device)
    else:
        print("CUDA not available, skipping acceleration test")