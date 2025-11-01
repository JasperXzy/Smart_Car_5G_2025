import cv2
import os
import glob
import uuid
import numpy as np
import random
from pathlib import Path
from datetime import datetime

def apply_data_augmentation(image):
    # 随机选择是否使用翻转，如果使用翻转就不再叠加其他效果
    use_flip = random.random() < 0.3  # 降低翻转概率到30%
    
    if use_flip:
        # 只进行翻转，不叠加其他效果
        result = cv2.flip(image, 1)
        return ('flip_only', result)
    else:
        # 不翻转，叠加其他增广效果
        result = image.copy().astype(np.float32)
        
        # 1. 亮度调整 (增亮)
        bright = cv2.convertScaleAbs(image, alpha=1.2, beta=30).astype(np.float32)
        result = cv2.addWeighted(result, 0.7, bright, 0.3, 0)
        
        # 2. 亮度调整 (变暗)
        dark = cv2.convertScaleAbs(image, alpha=0.8, beta=-30).astype(np.float32)
        result = cv2.addWeighted(result, 0.7, dark, 0.3, 0)
        
        # 3. 高斯模糊
        blurred = cv2.GaussianBlur(image, (3, 3), 0).astype(np.float32)
        result = cv2.addWeighted(result, 0.8, blurred, 0.2, 0)
        
        # 4. 添加轻微噪声
        noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
        noisy = cv2.add(image, noise).astype(np.float32)
        result = cv2.addWeighted(result, 0.8, noisy, 0.2, 0)
        
        # 转换回uint8
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return ('augmented_layered', result)

def augment_images_from_directory(input_dir, output_dir, use_uuid=True, keep_augmentation_info=False):
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 支持的图片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    # 收集所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
    
    print(f"找到 {len(image_files)} 张图片")
    
    total_augmented = 0
    augmentation_log = []
    
    for i, image_file in enumerate(image_files):
        print(f"处理图片 {i+1}/{len(image_files)}: {os.path.basename(image_file)}")
        
        # 读取图片
        image = cv2.imread(image_file)
        if image is None:
            print(f"  无法读取图片: {image_file}")
            continue
        
        # 进行数据增广
        aug_type, aug_image = apply_data_augmentation(image)
        
        # 生成唯一的UUID文件名
        if use_uuid:
            unique_id = str(uuid.uuid4())
            if keep_augmentation_info:
                output_filename = f"{unique_id}_{aug_type}.jpg"
            else:
                output_filename = f"{unique_id}.jpg"
        else:
            # 使用原始文件名 + 增广类型
            base_name = os.path.splitext(os.path.basename(image_file))[0]
            output_filename = f"{base_name}_{aug_type}_{uuid.uuid4().hex[:8]}.jpg"
        
        output_path = os.path.join(output_dir, output_filename)
        
        # 保存合成后的增广图片
        cv2.imwrite(output_path, aug_image)
        total_augmented += 1
        
        # 记录增广日志
        augmentation_log.append({
            'original_file': image_file,
            'output_file': output_path,
            'augmentation_type': aug_type
        })
        
        if (i + 1) % 10 == 0:
            print(f"  已处理 {i+1} 张原始图片，生成 {total_augmented} 张增广图片")
    
    # 保存增广日志
    log_file = os.path.join(output_dir, 'augmentation.log')
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"数据增广日志\n")
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"原始图片数量: {len(image_files)}\n")
        f.write(f"增广后图片数量: {total_augmented}\n")
        f.write(f"增广倍数: {total_augmented / len(image_files) if len(image_files) > 0 else 0:.1f}x\n\n")
        
        for log_entry in augmentation_log:
            f.write(f"原始: {log_entry['original_file']}\n")
            f.write(f"输出: {log_entry['output_file']}\n")
            f.write(f"类型: {log_entry['augmentation_type']}\n")
            f.write("-" * 50 + "\n")
    
    print(f"\n数据增广完成!")
    print(f"原始图片数量: {len(image_files)}")
    print(f"增广后图片数量: {total_augmented}")
    print(f"增广倍数: {total_augmented / len(image_files) if len(image_files) > 0 else 0:.1f}x")
    print(f"图片保存在: {output_dir}")
    print(f"增广日志保存在: {log_file}")

def main():
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    input_dir = os.path.join(parent_dir, "raw_image")
    output_dir = os.path.join(parent_dir, "image_argumented")
    
    print("数据增广工具")
    print("=" * 50)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print("=" * 50)
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    # 开始数据增广
    augment_images_from_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        use_uuid=True,
        keep_augmentation_info=False
    )

if __name__ == "__main__":
    main()
