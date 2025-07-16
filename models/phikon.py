from PIL import Image
import torch
from transformers import AutoImageProcessor, ViTModel
from torchvision import transforms
from PIL.Image import Resampling

def get_phikon_trans():
    """
    获取Phikon的预处理transform，完全等价于processor
    """
    processor = AutoImageProcessor.from_pretrained("owkin/phikon")
    
    transform = transforms.Compose([
        # 1. Resize到224x224，使用BILINEAR插值（phikon使用值=2）
        transforms.Resize((processor.size["height"], processor.size["width"]), interpolation=Resampling.BILINEAR),
        # 2. 转为tensor并自动除以255
        transforms.ToTensor(),
        # 3. 标准化
        transforms.Normalize(
            mean=processor.image_mean,
            std=processor.image_std
        )
    ])
    
    return transform

def get_phikon(device, gpu_num):
    model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False).to(device)
    model.eval()
    # if gpu_num > 1:
    #     model = torch.nn.parallel.DataParallel(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params/1000/1000)

    def func(image):
        # 检查输入类型，支持两种方式
        if isinstance(image, torch.Tensor):
            # 已经预处理过的tensor（来自DataLoader）
            if image.dim() == 3:  # (3, 224, 224)
                image = image.unsqueeze(0)  # 添加batch维度
            inputs = {"pixel_values": image.to(device, non_blocking=True)}
        else:
            # PIL图像（兼容原有方式，但效率较低）
            processor = AutoImageProcessor.from_pretrained("owkin/phikon")
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        
        # get the features
        with torch.inference_mode():
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]  # (1, 768) shape
            return features
    return func

if __name__ == '__main__':
    import numpy as np
    from PIL import Image
    import time
    
    def verify_phikon_preprocessing():
        """验证两种预处理方式是否等价"""
        
        # 初始化processor
        processor = AutoImageProcessor.from_pretrained("owkin/phikon")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("🔍 Phikon Processor信息:")
        print(f"  Resize尺寸: {processor.size}")
        print(f"  均值: {processor.image_mean}")
        print(f"  标准差: {processor.image_std}")
        print(f"  Rescale因子: {processor.rescale_factor}")
        from PIL.Image import Resampling
        print(f"  插值方法: {Resampling(processor.resample)} (值={processor.resample})")
        
        # 方法1：当前方式（processor预处理）
        def method1_current(pil_image):
            """当前方式：PIL -> processor -> tensor"""
            inputs = processor(pil_image, return_tensors="pt")
            return inputs["pixel_values"]
        
        # 方法2：修复方式（torchvision预处理）
        def method2_fixed(pil_image):
            """修复方式：PIL -> torchvision transforms -> tensor"""
            transform = get_phikon_trans()
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
    print("🧪 Phikon 预处理等价性验证")
    print("=" * 60)
    
    success = verify_phikon_preprocessing()
    
    print(f"\n🎯 验证结果: {'✅ 通过' if success else '❌ 失败'}")
    if success:
        print("🚀 Phikon模型已优化，预期GPU利用率将显著提升！")