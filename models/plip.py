from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import torch
from torchvision import transforms
from PIL.Image import Resampling

def get_plip_trans():
    """
    获取PLIP的预处理transform，完全等价于CLIPProcessor
    """
    processor = CLIPProcessor.from_pretrained('vinid/plip')
    img_proc = processor.image_processor
    
    transform = transforms.Compose([
        # 1. Resize最短边到224，使用BICUBIC插值
        transforms.Resize(img_proc.size["shortest_edge"], interpolation=Resampling.BICUBIC),
        # 2. Center crop到224x224
        transforms.CenterCrop((img_proc.crop_size["height"], img_proc.crop_size["width"])),
        # 3. 转为tensor并自动除以255
        transforms.ToTensor(),
        # 4. 标准化
        transforms.Normalize(
            mean=img_proc.image_mean,
            std=img_proc.image_std
        )
    ])
    
    return transform

def print_data_info(inputs):
    tensor = inputs['pixel_values']
    print('Device:{}, shape:{}, max:{:.4f}, min: {:.4f}'.format(tensor.device,
            tensor.shape, tensor.max(), tensor.min()))


def plip(device, gpu_num):
    model = CLIPModel.from_pretrained('vinid/plip').to(device)

    pytorch_total_params = sum(p.numel() for p in model.vision_model.parameters())
    print(pytorch_total_params/1000/1000)

    model.eval()

    def func(image):
        # 检查输入类型，支持两种方式
        if isinstance(image, torch.Tensor):
            # 已经预处理过的tensor（来自DataLoader）
            if image.dim() == 3:  # (3, 224, 224)
                image = image.unsqueeze(0)  # 添加batch维度
            inputs = {"pixel_values": image.to(device, non_blocking=True)}
        else:
            # PIL图像（兼容原有方式，但效率较低）
            processor = CLIPProcessor.from_pretrained('vinid/plip')
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        
        # print_data_info(inputs)
        with torch.inference_mode():
            img_embed = model.get_image_features(**inputs)
        # img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
        return img_embed
    return func


def plip_transformers():
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    trnsfrms_val = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std)
        ]
    )
    return trnsfrms_val

if __name__ == '__main__':
    import numpy as np
    from PIL import Image
    import time
    
    def verify_plip_preprocessing():
        """验证两种预处理方式是否等价"""
        
        # 初始化processor
        processor = CLIPProcessor.from_pretrained('vinid/plip')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("🔍 PLIP Processor信息:")
        img_proc = processor.image_processor
        print(f"  Resize尺寸: {img_proc.size}")
        print(f"  Crop尺寸: {img_proc.crop_size}")
        print(f"  均值: {img_proc.image_mean}")
        print(f"  标准差: {img_proc.image_std}")
        from PIL.Image import Resampling
        print(f"  插值方法: {Resampling(img_proc.resample)} (值={img_proc.resample})")
        
        # 方法1：当前方式（processor预处理）
        def method1_current(pil_image):
            """当前方式：PIL -> processor -> tensor"""
            # 使用images参数明确指定只处理图像
            inputs = processor(images=pil_image, return_tensors="pt")
            return inputs["pixel_values"]
        
        # 方法2：修复方式（torchvision预处理）
        def method2_fixed(pil_image):
            """修复方式：PIL -> torchvision transforms -> tensor"""
            transform = get_plip_trans()
            tensor = transform(pil_image)
            return tensor.unsqueeze(0)  # 添加batch维度
        
        # 创建测试图像
        test_image = Image.new('RGB', (256, 256), color=(128, 64, 192))
        
        print(f"\n🧪 测试图像: {test_image.size}, mode: {test_image.mode}")
        
        # 测试预处理等价性
        tensor1 = method1_current(test_image)
        tensor2 = method2_fixed(test_image)
        
        print(f"\n📊 预处理结果对比:")
        print(f"  Processor方式: shape={tensor1.shape}, dtype={tensor1.dtype}")
        print(f"  Transform方式: shape={tensor2.shape}, dtype={tensor2.dtype}")
        print(f"  数值范围: [{tensor1.min():.6f}, {tensor1.max():.6f}] vs [{tensor2.min():.6f}, {tensor2.max():.6f}]")
        
        # 计算差异
        diff = torch.abs(tensor1 - tensor2)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"  最大差异: {max_diff:.10f}")
        print(f"  平均差异: {mean_diff:.10f}")
        
        # 验证等价性
        tolerance = 1e-6
        if max_diff < tolerance:
            print(f"  ✅ 预处理等价性验证通过！(差异 < {tolerance})")
        else:
            print(f"  ❌ 预处理等价性验证失败！差异过大: {max_diff}")
            
        # 性能测试
        print(f"\n⚡ 性能对比测试:")
        
        # 测试processor方式
        start_time = time.time()
        for _ in range(100):
            _ = method1_current(test_image)
        time1 = time.time() - start_time
        
        # 测试transform方式  
        start_time = time.time()
        for _ in range(100):
            _ = method2_fixed(test_image)
        time2 = time.time() - start_time
        
        print(f"  Processor方式: {time1:.4f}s (100次)")
        print(f"  Transform方式: {time2:.4f}s (100次)")
        print(f"  速度提升: {time1/time2:.2f}x")
        
        return max_diff < tolerance
    
    # 执行验证
    print("=" * 60)
    print("🧪 PLIP 预处理等价性验证")
    print("=" * 60)
    
    success = verify_plip_preprocessing()
    
    print(f"\n🎯 验证结果: {'✅ 通过' if success else '❌ 失败'}")
    if success:
        print("🚀 PLIP模型已优化，预期GPU利用率将显著提升！")