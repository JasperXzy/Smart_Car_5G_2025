import cv2
import os
import glob
from pathlib import Path

def extract_frames_from_videos(video_dir, output_dir, fps=2):
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 支持的视频格式
    video_extensions = ['*.avi', '*.mp4', '*.mov', '*.mkv', '*.flv', '*.wmv']
    
    # 获取所有视频文件
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))
    
    print(f"找到 {len(video_files)} 个视频文件:")
    for video_file in video_files:
        print(f"  - {video_file}")
    
    total_frames_extracted = 0
    
    for video_file in video_files:
        print(f"\n处理视频: {os.path.basename(video_file)}")
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_file)
        
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_file}")
            continue
        
        # 获取视频信息
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        print(f"  视频FPS: {video_fps:.2f}")
        print(f"  总帧数: {total_frames}")
        print(f"  视频时长: {duration:.2f}秒")
        
        # 计算抽帧间隔
        if video_fps <= 0:
            print(f"  无法获取视频FPS，跳过该视频")
            cap.release()
            continue
            
        frame_interval = int(video_fps / fps)  # 每隔多少帧提取一帧
        print(f"  抽帧间隔: 每{frame_interval}帧提取1帧")
        
        # 创建该视频的输出子目录
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        video_output_dir = os.path.join(output_dir, video_name)
        Path(video_output_dir).mkdir(parents=True, exist_ok=True)
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 按间隔提取帧
            if frame_count % frame_interval == 0:
                # 计算时间戳
                timestamp = frame_count / video_fps
                
                # 生成输出文件名
                output_filename = f"{video_name}_frame_{extracted_count:06d}_time_{timestamp:.2f}s.jpg"
                output_path = os.path.join(video_output_dir, output_filename)
                
                # 保存帧
                cv2.imwrite(output_path, frame)
                extracted_count += 1
                
                if extracted_count % 100 == 0:
                    print(f"  已提取 {extracted_count} 帧...")
            
            frame_count += 1
        
        cap.release()
        print(f"  从视频 {video_name} 中提取了 {extracted_count} 帧")
        total_frames_extracted += extracted_count
    
    print(f"\n总共提取了 {total_frames_extracted} 帧图片")
    print(f"图片保存在: {output_dir}")

def main():
    """
    主函数
    """
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    video_dir = os.path.join(parent_dir, "raw_video")
    output_dir = os.path.join(parent_dir, "raw_image")
    
    # 检查视频目录是否存在
    if not os.path.exists(video_dir):
        print(f"错误: 视频目录不存在: {video_dir}")
        return
    
    # 开始提取帧
    extract_frames_from_videos(video_dir, output_dir, fps=2)

if __name__ == "__main__":
    main()