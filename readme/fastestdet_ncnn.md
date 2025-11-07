
### 使用方法

```
./infer --img <image> --model <model.param> --output <output.jpg> --conf 0.65 --nms 0.45 --size 352
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--img` | 输入图像路径 | `../test.jpg` |
| `--model` | 模型参数文件(.param) | `../FastestDet.ncnn.param` |
| `--output` | 输出图像路径 | `../output.jpg` |
| `--conf` | 置信度阈值 | 0.65 |
| `--nms` | NMS阈值 | 0.45 |
| `--size` | 输入尺寸 | 352 |
