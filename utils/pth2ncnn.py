import os
import sys
import argparse
import subprocess
import warnings
import torch

# 过滤 TorchScript 追踪警告
warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)

# 将 FastestDet 模型目录添加到路径以导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models', 'fastestdet'))

from module.detector import Detector


def export_torchscript(model, input_shape, output_path):
    print(f"Exporting TorchScript model to {output_path}...")
    model.eval()
    model.cpu()
    
    # 创建虚拟输入
    dummy_input = torch.rand(*input_shape)
    
    # 追踪模型
    traced_model = torch.jit.trace(model, dummy_input)
    
    # 保存模型
    traced_model.save(output_path)
    print(f"TorchScript model saved successfully!")
    
    return output_path


def run_pnnx(torchscript_path, input_shape, pnnx_path="pnnx", output_dir=None):
    print(f"\nRunning PNNX conversion...")

    input_shape_str = f"[{','.join(map(str, input_shape))}]"
    
    # 如果指定了输出目录，则处理文件路径
    if output_dir:
        import shutil
        os.makedirs(output_dir, exist_ok=True)
        original_dir = os.getcwd()
        
        # 复制 torchscript 文件到输出目录
        local_ts_path = os.path.join(output_dir, os.path.basename(torchscript_path))
        abs_torchscript_path = os.path.abspath(torchscript_path)
        abs_local_ts_path = os.path.abspath(local_ts_path)
        
        if abs_torchscript_path != abs_local_ts_path:
            shutil.copy(abs_torchscript_path, abs_local_ts_path)
            print(f"Copied model to: {abs_local_ts_path}")
        
        # 切换到输出目录
        os.chdir(output_dir)
        cmd = [pnnx_path, os.path.basename(torchscript_path), f"inputshape={input_shape_str}"]
    else:
        original_dir = None
        cmd = [pnnx_path, torchscript_path, f"inputshape={input_shape_str}"]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print("PNNX conversion successful!")
        
        # 列出生成的文件
        if output_dir:
            base_name = os.path.splitext(os.path.basename(torchscript_path))[0]
            ncnn_param = f"{base_name}.ncnn.param"
            ncnn_bin = f"{base_name}.ncnn.bin"
            
            if os.path.exists(ncnn_param) and os.path.exists(ncnn_bin):
                print(f"\nGenerated NCNN files:")
                print(f"  - {os.path.abspath(ncnn_param)}")
                print(f"  - {os.path.abspath(ncnn_bin)}")
        
        # 恢复原始目录
        if original_dir:
            os.chdir(original_dir)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running PNNX: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        if original_dir:
            os.chdir(original_dir)
        raise
    except FileNotFoundError as e:
        print(f"Error: PNNX executable not found at '{pnnx_path}'")
        print("Please install PNNX or specify the correct path with --pnnx-path")
        if original_dir:
            os.chdir(original_dir)
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        if original_dir:
            os.chdir(original_dir)
        raise


def main():
    parser = argparse.ArgumentParser(description='Convert FastestDet PTH to NCNN format using PNNX')
    parser.add_argument('--weight', type=str, required=True, 
                        help='Path to the .pth weight file')
    parser.add_argument('--nc', type=int, required=True,
                        help='Number of classes/categories')
    parser.add_argument('--input', type=str, default='352,352',
                        help='Input size as "width,height" (default: 352,352)')
    parser.add_argument('--output', type=str, default='./FastestDet.pt',
                        help='Path to save the torchscript model (default: ./FastestDet.pt)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save NCNN output files (default: same as torchscript)')
    parser.add_argument('--pnnx-path', type=str, default='pnnx',
                        help='Path to PNNX executable (default: pnnx in PATH)')
    parser.add_argument('--skip-pnnx', action='store_true',
                        help='Only export TorchScript, skip PNNX conversion')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    assert os.path.exists(args.weight), f"Weight file not found: {args.weight}"
    
    # 解析输入尺寸
    try:
        input_size = args.input.split(',')
        assert len(input_size) == 2, "Input size must be in format 'width,height'"
        input_width = int(input_size[0])
        input_height = int(input_size[1])
    except Exception as e:
        print(f"Error parsing input size: {e}")
        print("Please use format: --input width,height (e.g., --input 352,352)")
        return
    
    # 获取类别数
    category_num = args.nc
    
    print(f"\nModel Configuration:")
    print(f"  Categories: {category_num}")
    print(f"  Input size: {input_width}x{input_height}")
    
    # 加载模型
    print(f"\nLoading model from: {args.weight}")
    model = Detector(category_num, load_param=True)
    model.load_state_dict(torch.load(args.weight, map_location='cpu'))
    model.eval()
    model.cpu()
    
    # 导出为 TorchScript
    input_shape = (1, 3, input_height, input_width)
    torchscript_path = export_torchscript(model, input_shape, args.output)
    
    if not args.skip_pnnx:
        # 运行 PNNX 转换
        try:
            run_pnnx(torchscript_path, input_shape, args.pnnx_path, args.output_dir)
            print("\nConversion completed successfully!")
        except Exception as e:
            print(f"\nPNNX conversion failed: {e}")
            print(f"\nYou can manually run PNNX with:")
            print(f"  pnnx {torchscript_path} inputshape=[{','.join(map(str, input_shape))}]")
    else:
        print("\nTorchScript export completed!")
        print(f"\nTo convert to NCNN, run:")
        print(f"  pnnx {torchscript_path} inputshape=[{','.join(map(str, input_shape))}]")


if __name__ == '__main__':
    main()
