## ONNX 到 NCNN 转换 (onnx2ncnn.py)

### 前置条件

在使用脚本前，确保安装：

1. NCNN 工具
2. PATH 中包含所需工具
   - `onnx2ncnn`
   - `ncnnoptimize`

#### 快速 NCNN 工具检查

```bash
# 检查工具是否可用
which onnx2ncnn
which ncnnoptimize

# 它们应该位于
ls /usr/local/bin/onnx2ncnn
ls /usr/local/bin/ncnnoptimize
```

### 使用示例

```bash

# 基本转换
python3 ./onnx2ncnn.py ../models/FastestDet.onnx

# 使用自定义输出目录进行转换
python3 ./onnx2ncnn.py ../models/FastestDet.onnx --output ../models

# 带 FP16 优化的转换
python3 ./onnx2ncnn.py ../models/FastestDet.onnx --output ../models --fp16

# 带 INT8 量化的转换
python3 ./onnx2ncnn.py ../models/FastestDet.onnx --output ../models --int8

# 不带任何优化的转换
python3 ./onnx2ncnn.py ../models/FastestDet.onnx --output ../models --no-optimize

# 获取帮助
python3 ./onnx2ncnn.py --help
```

### 命令行选项

脚本支持以下选项：

- `--output` / `-o` - 转换模型的输出目录（默认：./ncnn_models）
- `--fp16` - 启用 FP16 优化（可减少模型大小约 50%）
- `--int8` - 启用 INT8 量化（实验性功能，最大压缩比）
- `--optimize` - 启用模型优化（默认：启用）
- `--no-optimize` - 禁用所有模型优化
- `--verbose` / `-v` - 启用详细输出

### 输出结构

```
ncnn_models/（或指定的输出目录）
├── FastestDet.param    # 模型结构定义文件
└── FastestDet.bin      # 模型权重二进制文件
```
