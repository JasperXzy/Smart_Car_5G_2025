
### 使用方法

```bash
./infer_video --video <path> --model <path> --output <path> --size <int> --conf <float> --nms <float> --fps <int> --display --no-save
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--video <path>` | 输入视频文件路径 | `test.avi` |
| `--model <path>` | 模型参数文件路径 | `FastestDet.ncnn.param` |
| `--output <path>` | 输出视频文件路径 | `output.avi` |
| `--size <int>` | 模型输入尺寸 | `352` |
| `--conf <float>` | 置信度阈值 | `0.65` |
| `--nms <float>` | NMS 阈值 | `0.45` |
| `--fps <int>` | 输出视频 FPS | 与输入相同 |
| `--display` | 实时显示推理结果 | `false` |
| `--no-save` | 不保存输出视频 | `false` |
| `-h, --help` | 显示帮助信息 | - |
