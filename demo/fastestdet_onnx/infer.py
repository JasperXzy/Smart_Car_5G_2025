import os
import cv2
import time
import argparse
import numpy as np
import onnxruntime


def nms(bboxes, scores, threshold=0.5):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]
    
    return keep


def handle_preds(output, conf_thresh=0.25, nms_thresh=0.45):
    # output shape: [1, C, H, W]
    output = output[0]  # [C, H, W]
    C, H, W = output.shape
    
    # 前背景分类分支
    pobj = output[0:1, :, :]  # [1, H, W]
    # 检测框回归分支
    preg = output[1:5, :, :]  # [4, H, W]
    # 目标类别分类分支
    pcls = output[5:, :, :]   # [nc, H, W]
    
    # 生成网格坐标
    gy, gx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # 检测框置信度
    pobj_squeeze = pobj.squeeze(0)  # [H, W]
    pcls_max = pcls.max(axis=0)     # [H, W]
    pcls_argmax = pcls.argmax(axis=0)  # [H, W]
    
    conf = (pobj_squeeze ** 0.6) * (pcls_max ** 0.4)
    
    # 检测框的坐标（归一化）
    bw = 1 / (1 + np.exp(-preg[2, :, :]))  # sigmoid
    bh = 1 / (1 + np.exp(-preg[3, :, :]))  # sigmoid
    
    # tanh
    preg0_tanh = np.tanh(preg[0, :, :])
    preg1_tanh = np.tanh(preg[1, :, :])
    
    bcx = (preg0_tanh + gx) / W
    bcy = (preg1_tanh + gy) / H
    
    # cx,cy,w,h => x1,y1,x2,y2
    x1 = bcx - 0.5 * bw
    y1 = bcy - 0.5 * bh
    x2 = bcx + 0.5 * bw
    y2 = bcy + 0.5 * bh
    
    # 阈值筛选
    mask = conf > conf_thresh
    
    # 提取有效检测框
    x1_valid = x1[mask]
    y1_valid = y1[mask]
    x2_valid = x2[mask]
    y2_valid = y2[mask]
    conf_valid = conf[mask]
    cls_valid = pcls_argmax[mask]
    
    if len(x1_valid) == 0:
        return []
    
    # 对每个类别分别进行NMS
    output_bboxes = []
    unique_classes = np.unique(cls_valid)
    
    for cls in unique_classes:
        cls_mask = cls_valid == cls
        cls_bboxes = np.stack([x1_valid[cls_mask], 
                               y1_valid[cls_mask], 
                               x2_valid[cls_mask], 
                               y2_valid[cls_mask]], axis=1)
        cls_scores = conf_valid[cls_mask]
        
        # NMS
        keep_indices = nms(cls_bboxes, cls_scores, nms_thresh)
        
        for i in keep_indices:
            output_bboxes.append([
                cls_bboxes[i][0],  # x1
                cls_bboxes[i][1],  # y1
                cls_bboxes[i][2],  # x2
                cls_bboxes[i][3],  # y2
                cls_scores[i],     # conf
                cls                # class
            ])
    
    return output_bboxes


def main():
    parser = argparse.ArgumentParser(description='FastestDet ONNX Inference')
    parser.add_argument('--model', type=str, default='FastestDet.onnx', 
                        help='ONNX model path')
    parser.add_argument('--img', type=str, required=True, 
                        help='Input image path')
    parser.add_argument('--output', type=str, default='output.jpg', 
                        help='Output image path')
    parser.add_argument('--conf_thresh', type=float, default=0.65, 
                        help='Confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.45, 
                        help='NMS threshold')
    parser.add_argument('--input_size', type=int, default=352, 
                        help='Model input size')
    parser.add_argument('--names', type=str, default=None, 
                        help='Class names file path')
    
    args = parser.parse_args()
    
    # 检查文件
    assert os.path.exists(args.model), f"Model file not found: {args.model}"
    assert os.path.exists(args.img), f"Image file not found: {args.img}"
    
    # 加载类别名称
    LABEL_NAMES = []
    if args.names and os.path.exists(args.names):
        with open(args.names, 'r') as f:
            for line in f.readlines():
                LABEL_NAMES.append(line.strip())
    else:
        # 默认类别
        LABEL_NAMES = ['A', 'B']
    
    print(f"Class names: {LABEL_NAMES}")
    
    # 创建ONNX Runtime会话
    print(f"Loading model: {args.model}")
    session = onnxruntime.InferenceSession(args.model)
    
    # 获取输入输出信息
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Input name: {input_name}")
    print(f"Output name: {output_name}")
    
    # 读取并预处理图像
    print(f"Loading image: {args.img}")
    ori_img = cv2.imread(args.img)
    H, W, _ = ori_img.shape
    print(f"Original image size: {W}x{H}")
    
    # Resize图像
    res_img = cv2.resize(ori_img, (args.input_size, args.input_size), 
                         interpolation=cv2.INTER_LINEAR)
    
    # 转换为模型输入格式 [1, 3, H, W]
    img = res_img.transpose(2, 0, 1)  # HWC -> CHW
    img = img.reshape(1, 3, args.input_size, args.input_size)
    img = img.astype(np.float32) / 255.0
    
    # 推理
    print("Running inference...")
    start = time.perf_counter()
    output = session.run([output_name], {input_name: img})[0]
    end = time.perf_counter()
    inference_time = (end - start) * 1000.0
    print(f"Inference time: {inference_time:.2f}ms")
    
    # 后处理
    print("Post-processing...")
    detections = handle_preds(output, args.conf_thresh, args.nms_thresh)
    print(f"Detected {len(detections)} objects")
    
    # 绘制检测框
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        
        # 转换回原图坐标
        x1 = int(x1 * W)
        y1 = int(y1 * H)
        x2 = int(x2 * W)
        y2 = int(y2 * H)
        
        # 获取类别名称
        if cls < len(LABEL_NAMES):
            label = LABEL_NAMES[cls]
        else:
            label = f"Class_{cls}"
        
        print(f"  {label}: {conf:.3f} [{x1}, {y1}, {x2}, {y2}]")
        
        # 绘制边界框
        cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(ori_img, f'{conf:.2f}', (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(ori_img, label, (x1, y1 - 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 保存结果
    cv2.imwrite(args.output, ori_img)
    print(f"Result saved to: {args.output}")


if __name__ == '__main__':
    main()
