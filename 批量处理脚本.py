import os
import subprocess
from pathlib import Path

def process_folder(folder_path, output_base="results"):
    """处理文件夹中的所有图片"""
    folder_path = Path(folder_path)
    output_base = Path(output_base)
    
    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    for img_file in folder_path.iterdir():
        if img_file.suffix.lower() in image_extensions:
            print(f"\n处理文件: {img_file.name}")
            # 为每张图片创建独立的输出文件夹
            img_output_dir = output_base / img_file.stem
            
            # 运行主程序
            cmd = [
                "python", "main.py",
                "--image", str(img_file),
                "--output-dir", str(img_output_dir),
                "--model", "yolov8s.pt",
                "--conf", "0.5"
            ]
            
            subprocess.run(cmd)

if __name__ == "__main__":
    # 指定要处理的文件夹
    process_folder("sample")#改文件夹改这