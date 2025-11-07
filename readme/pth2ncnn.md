
### 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--weight` | ✅ | - | .pth 权重文件路径 |
| `--nc` | ✅ | - | 类别数量 (number of classes) |
| `--input` | ❌ | `352,352` | 输入尺寸 "宽,高" |
| `--output` | ❌ | `./FastestDet.pt` | TorchScript 模型输出路径 |
| `--output-dir` | ❌ | 当前目录 | NCNN 输出文件目录 |
| `--pnnx-path` | ❌ | `pnnx` | PNNX 可执行文件路径 |
| `--skip-pnnx` | ❌ | `False` | 仅导出 TorchScript，跳过 PNNX 转换 |
