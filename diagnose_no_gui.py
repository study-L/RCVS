import os
import sys

# 无GUI诊断脚本
def diagnose_no_gui():
    try:
        print(f"Python版本: {sys.version}")
        print(f"当前工作目录: {os.getcwd()}")
        
        # 尝试导入可能的依赖
        print("尝试导入依赖...")
        try:
            import cv2
            print(f"OpenCV已导入，版本: {cv2.__version__}")
        except ImportError:
            print("OpenCV未安装")
            
        try:
            import numpy
            print(f"NumPy已导入，版本: {numpy.__version__}")
        except ImportError:
            print("NumPy未安装")
        
        try:
            import torch
            print(f"PyTorch已导入，版本: {torch.__version__}")
        except ImportError:
            print("PyTorch未安装")
        
        try:
            from einops import rearrange
            print("einops已导入")
        except ImportError:
            print("einops未安装")
        
        # 检查文件路径
        print("\n检查文件路径...")
        ir_dir = r"D:\HDO_Raw_Data\ir"
        visible_dir = r"D:\HDO_Raw_Data\vi"
        
        print(f"红外目录: {ir_dir}")
        print(f"目录存在: {os.path.exists(ir_dir)}")
        if os.path.exists(ir_dir):
            files = os.listdir(ir_dir)
            print(f"目录文件数: {len(files)}")
            if files:
                print(f"前3个文件: {files[:3]}")
        
        print(f"\n可见光目录: {visible_dir}")
        print(f"目录存在: {os.path.exists(visible_dir)}")
        if os.path.exists(visible_dir):
            files = os.listdir(visible_dir)
            print(f"目录文件数: {len(files)}")
            if files:
                print(f"前3个文件: {files[:3]}")
        
        print("\n诊断完成！")
        
    except Exception as e:
        print(f"诊断过程中发生错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_no_gui()