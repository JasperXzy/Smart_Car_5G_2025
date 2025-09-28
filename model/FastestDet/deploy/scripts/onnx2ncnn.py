import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path


class ONNX2NCNNConverter:
    """ONNX to NCNN model converter with optimization options"""
    
    def __init__(self, onnx_path, output_dir="./ncnn_models"):
        self.onnx_path = Path(onnx_path)
        self.output_dir = Path(output_dir)
        self.model_name = self.onnx_path.stem
        
        # Output file paths
        self.param_path = self.output_dir / f"{self.model_name}.param"
        self.bin_path = self.output_dir / f"{self.model_name}.bin"
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def check_dependencies(self):
        """Check if required tools are installed"""
        required_tools = ['onnx2ncnn']
        missing_tools = []
        
        for tool in required_tools:
            if not shutil.which(tool):
                missing_tools.append(tool)
        
        if missing_tools:
            print(f"ERROR: Missing required tools: {', '.join(missing_tools)}")
            print("\nTo install NCNN tools:")
            print("1. Follow the INSTALL_NCNN.md guide")
            print("2. Make sure onnx2ncnn is in your PATH")
            print("3. Typical location: /usr/local/bin/onnx2ncnn")
            return False
        
        return True
    
    def convert_basic(self):
        """Basic ONNX to NCNN conversion"""
        print(f"Converting {self.onnx_path} to NCNN format...")
        print(f"Output directory: {self.output_dir}")
        
        # Basic conversion command
        cmd = [
            'onnx2ncnn',
            str(self.onnx_path),
            str(self.param_path),
            str(self.bin_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("Basic conversion completed successfully!\n")
            
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
                
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Conversion failed with return code {e.returncode}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            return False
    
    def optimize_model(self, fp16=True, int8=False):
        """Optimize the converted NCNN model"""
        print("Optimizing NCNN model...")
        
        optimized_param = self.output_dir / f"{self.model_name}_opt.param"
        optimized_bin = self.output_dir / f"{self.model_name}_opt.bin"
        
        # ncnnoptimize usage: [inparam] [inbin] [outparam] [outbin] [flag] [cutstart] [cutend]
        # flag: 0=fp32, 1=fp16, 2=int8, 65536=no optimization
        flag = 0  # default fp32
        if fp16 and int8:
            flag = 2  # int8 has priority
            print("  - Enabling INT8 quantization")
        elif fp16:
            flag = 1  # fp16
            print("  - Enabling FP16 optimization")
        elif int8:
            flag = 2  # int8
            print("  - Enabling INT8 quantization")
        else:
            flag = 0  # fp32
            print("  - Using FP32 (no optimization)")
        
        cmd = [
            'ncnnoptimize',
            str(self.param_path),
            str(self.bin_path),
            str(optimized_param),
            str(optimized_bin),
            str(flag)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("Model optimization completed successfully!")
            
            # Replace original files with optimized versions
            shutil.move(str(optimized_param), str(self.param_path))
            shutil.move(str(optimized_bin), str(self.bin_path))
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"WARNING: Optimization failed: {e.stderr}")
            print("Using non-optimized model...")
            return False
        except FileNotFoundError:
            print("WARNING: ncnnoptimize not found, skipping optimization")
            return False
    
    def validate_model(self):
        """Validate the converted model"""
        print("\nValidating converted model...")
        
        if not self.param_path.exists():
            print(f"ERROR: Parameter file not found: {self.param_path}")
            return False
            
        if not self.bin_path.exists():
            print(f"ERROR: Binary file not found: {self.bin_path}")
            return False
        
        # Check file sizes
        param_size = self.param_path.stat().st_size
        bin_size = self.bin_path.stat().st_size
        
        print(f"Parameter file: {self.param_path} ({param_size:,} bytes)")
        print(f"Binary file: {self.bin_path} ({bin_size:,} bytes)")
        
        if param_size == 0 or bin_size == 0:
            print("ERROR: One or more output files are empty!")
            return False
        
        return True
    
    def print_model_info(self):
        """Print information about the converted model"""
        print(f"\n{'='*50}")
        print("CONVERSION SUMMARY")
        print(f"{'='*50}")
        print(f"Input ONNX: {self.onnx_path}")
        print(f"Output Param: {self.param_path}")
        print(f"Output Binary: {self.bin_path}")
        print(f"Model Name: {self.model_name}")
        
        # File sizes
        if self.param_path.exists() and self.bin_path.exists():
            param_size = self.param_path.stat().st_size / 1024 / 1024  # MB
            bin_size = self.bin_path.stat().st_size / 1024 / 1024      # MB
            print(f"Param Size: {param_size:.2f} MB")
            print(f"Binary Size: {bin_size:.2f} MB")
            print(f"Total Size: {param_size + bin_size:.2f} MB")
        
        print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX models to NCNN format for FastestDet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  ./onnx2ncnn.py model.onnx
  
  # Convert with custom output directory
  ./onnx2ncnn.py model.onnx --output ./models
  
  # Convert with optimizations
  ./onnx2ncnn.py model.onnx --fp16 --optimize
  
  # Convert without any optimizations
  ./onnx2ncnn.py model.onnx --no-optimize
        """
    )
    
    parser.add_argument('onnx_file', help='Input ONNX model file')
    parser.add_argument('-o', '--output', default='./ncnn_models', 
                       help='Output directory for NCNN models (default: ./ncnn_models)')
    parser.add_argument('--fp16', action='store_true', 
                       help='Enable FP16 optimization (default: enabled)')
    parser.add_argument('--int8', action='store_true', 
                       help='Enable INT8 quantization (experimental)')
    parser.add_argument('--optimize', action='store_true', default=True,
                       help='Enable model optimization (default: enabled)')
    parser.add_argument('--no-optimize', action='store_true',
                       help='Disable model optimization')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Handle optimization flags
    if args.no_optimize:
        args.optimize = False
    
    # Check input file
    if not Path(args.onnx_file).exists():
        print(f"ERROR: Input file not found: {args.onnx_file}")
        sys.exit(1)
    
    # Create converter
    converter = ONNX2NCNNConverter(args.onnx_file, args.output)
    
    # Check dependencies
    if not converter.check_dependencies():
        sys.exit(1)
    
    # Perform conversion
    success = True
    
    # Step 1: Basic conversion
    if not converter.convert_basic():
        success = False
    
    # Step 2: Optimization (if requested)
    if success and args.optimize:
        converter.optimize_model(fp16=args.fp16, int8=args.int8)
    
    # Step 3: Validation
    if success:
        success = converter.validate_model()
    
    # Step 4: Print summary
    if success:
        converter.print_model_info()
        print("\nConversion completed successfully!")
        sys.exit(0)
    else:
        print("\nConversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
