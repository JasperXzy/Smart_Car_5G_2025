#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace cv;

// 检测结果结构体
struct Detection {
    float x1, y1, x2, y2;  // 归一化坐标
    float confidence;
    int class_id;
};

// 配置参数
struct Config {
    string model_path = "../FastestDet.onnx";
    string image_path = "../test.jpg";
    string output_path = "../output.jpg";
    int input_size = 352;
    float conf_thresh = 0.65f;
    float nms_thresh = 0.45f;
    vector<string> class_names = {"A", "B"};
};

// Sigmoid函数
inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// NMS算法
vector<int> nms(const vector<Detection>& detections, float threshold) {
    vector<int> indices(detections.size());
    for (size_t i = 0; i < detections.size(); i++) {
        indices[i] = i;
    }
    
    // 按置信度排序
    sort(indices.begin(), indices.end(), [&](int a, int b) {
        return detections[a].confidence > detections[b].confidence;
    });
    
    vector<bool> suppressed(detections.size(), false);
    vector<int> keep;
    
    for (size_t i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        if (suppressed[idx]) continue;
        
        keep.push_back(idx);
        
        const Detection& det_a = detections[idx];
        float area_a = (det_a.x2 - det_a.x1) * (det_a.y2 - det_a.y1);
        
        for (size_t j = i + 1; j < indices.size(); j++) {
            int idx2 = indices[j];
            if (suppressed[idx2]) continue;
            
            const Detection& det_b = detections[idx2];
            
            // 计算IoU
            float xx1 = max(det_a.x1, det_b.x1);
            float yy1 = max(det_a.y1, det_b.y1);
            float xx2 = min(det_a.x2, det_b.x2);
            float yy2 = min(det_a.y2, det_b.y2);
            
            float w = max(0.0f, xx2 - xx1);
            float h = max(0.0f, yy2 - yy1);
            float inter = w * h;
            
            float area_b = (det_b.x2 - det_b.x1) * (det_b.y2 - det_b.y1);
            float iou = inter / (area_a + area_b - inter);
            
            if (iou > threshold) {
                suppressed[idx2] = true;
            }
        }
    }
    
    return keep;
}

// 后处理函数
vector<Detection> postprocess(float* output, int C, int H, int W, 
                             float conf_thresh, float nms_thresh) {
    vector<Detection> all_detections;
    
    // 生成网格坐标
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            int base_idx = h * W + w;
            
            // 获取各分支的值
            // pobj: channel 0
            float pobj = output[0 * H * W + base_idx];
            
            // preg: channels 1-4
            float preg0 = output[1 * H * W + base_idx];
            float preg1 = output[2 * H * W + base_idx];
            float preg2 = output[3 * H * W + base_idx];
            float preg3 = output[4 * H * W + base_idx];
            
            // pcls: channels 5+
            int num_classes = C - 5;
            float max_cls_score = -FLT_MAX;
            int max_cls_id = 0;
            
            for (int c = 0; c < num_classes; c++) {
                float cls_score = output[(5 + c) * H * W + base_idx];
                if (cls_score > max_cls_score) {
                    max_cls_score = cls_score;
                    max_cls_id = c;
                }
            }
            
            // 计算置信度: obj^0.6 * cls^0.4
            float conf = pow(pobj, 0.6f) * pow(max_cls_score, 0.4f);
            
            if (conf > conf_thresh) {
                // 计算边界框
                float bw = sigmoid(preg2);
                float bh = sigmoid(preg3);
                float bcx = (tanh(preg0) + w) / W;
                float bcy = (tanh(preg1) + h) / H;
                
                // 转换为x1,y1,x2,y2格式
                float x1 = bcx - 0.5f * bw;
                float y1 = bcy - 0.5f * bh;
                float x2 = bcx + 0.5f * bw;
                float y2 = bcy + 0.5f * bh;
                
                Detection det;
                det.x1 = x1;
                det.y1 = y1;
                det.x2 = x2;
                det.y2 = y2;
                det.confidence = conf;
                det.class_id = max_cls_id;
                
                all_detections.push_back(det);
            }
        }
    }
    
    if (all_detections.empty()) {
        return all_detections;
    }
    
    // 按类别分组进行NMS
    map<int, vector<Detection>> class_detections;
    for (const auto& det : all_detections) {
        class_detections[det.class_id].push_back(det);
    }
    
    vector<Detection> final_detections;
    for (auto& pair : class_detections) {
        vector<int> keep = nms(pair.second, nms_thresh);
        for (int idx : keep) {
            final_detections.push_back(pair.second[idx]);
        }
    }
    
    return final_detections;
}

// 主函数
int main(int argc, char* argv[]) {
    Config config;
    
    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            config.model_path = argv[++i];
        } else if (arg == "--img" && i + 1 < argc) {
            config.image_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_path = argv[++i];
        } else if (arg == "--conf" && i + 1 < argc) {
            config.conf_thresh = stof(argv[++i]);
        } else if (arg == "--nms" && i + 1 < argc) {
            config.nms_thresh = stof(argv[++i]);
        } else if (arg == "--size" && i + 1 < argc) {
            config.input_size = stoi(argv[++i]);
        }
    }

    cout << "Model: " << config.model_path << endl;
    cout << "Image: " << config.image_path << endl;
    cout << "Input size: " << config.input_size << endl;
    cout << "Conf thresh: " << config.conf_thresh << endl;
    cout << "NMS thresh: " << config.nms_thresh << endl;
    
    try {
        // 初始化ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "FastestDet");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // 加载模型
        cout << "\nLoading model..." << endl;
        Ort::Session session(env, config.model_path.c_str(), session_options);
        
        // 获取输入输出信息
        Ort::AllocatorWithDefaultOptions allocator;
        
        // 正确获取输入输出名称
        Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
        Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator);
        
        string input_name_str(input_name_ptr.get());
        string output_name_str(output_name_ptr.get());
        
        cout << "Input name: " << input_name_str << endl;
        cout << "Output name: " << output_name_str << endl;
        
        // 读取图像
        cout << "\nLoading image..." << endl;
        Mat ori_img = imread(config.image_path);
        if (ori_img.empty()) {
            cerr << "Error: Failed to load image: " << config.image_path << endl;
            return -1;
        }
        
        int ori_h = ori_img.rows;
        int ori_w = ori_img.cols;
        cout << "Image size: " << ori_w << "x" << ori_h << endl;
        
        // 预处理
        Mat img;
        resize(ori_img, img, Size(config.input_size, config.input_size), 0, 0, INTER_LINEAR);
        img.convertTo(img, CV_32FC3, 1.0 / 255.0);
        
        // 转换为CHW格式
        vector<float> input_tensor_values(config.input_size * config.input_size * 3);
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < config.input_size; h++) {
                for (int w = 0; w < config.input_size; w++) {
                    input_tensor_values[c * config.input_size * config.input_size + 
                                       h * config.input_size + w] = 
                        img.at<Vec3f>(h, w)[c];
                }
            }
        }
        
        // 创建输入tensor
        vector<int64_t> input_shape = {1, 3, config.input_size, config.input_size};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size());
        
        // 推理
        cout << "\nRunning inference..." << endl;
        auto start = chrono::high_resolution_clock::now();
        
        vector<const char*> input_names = {input_name_str.c_str()};
        vector<const char*> output_names = {output_name_str.c_str()};
        
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(), &input_tensor, 1,
            output_names.data(), 1);
        
        auto end = chrono::high_resolution_clock::now();
        float inference_time = chrono::duration<float, milli>(end - start).count();
        cout << "Inference time: " << inference_time << " ms" << endl;
        
        // 获取输出
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        int C = output_shape[1];
        int H = output_shape[2];
        int W = output_shape[3];
        
        cout << "Output shape: [1, " << C << ", " << H << ", " << W << "]" << endl;
        
        // 后处理
        cout << "\nPost-processing..." << endl;
        vector<Detection> detections = postprocess(
            output_data, C, H, W, config.conf_thresh, config.nms_thresh);
        
        cout << "Detected " << detections.size() << " objects" << endl;
        
        // 可视化
        for (const auto& det : detections) {
            int x1 = static_cast<int>(det.x1 * ori_w);
            int y1 = static_cast<int>(det.y1 * ori_h);
            int x2 = static_cast<int>(det.x2 * ori_w);
            int y2 = static_cast<int>(det.y2 * ori_h);
            
            string label = static_cast<size_t>(det.class_id) < config.class_names.size() 
                ? config.class_names[det.class_id] 
                : "class_" + to_string(det.class_id);
            
            cout << "  " << label << ": " << det.confidence 
                 << " at [" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << "]" << endl;
            
            // 绘制边界框
            rectangle(ori_img, Point(x1, y1), Point(x2, y2), Scalar(255, 255, 0), 2);
            
            // 绘制置信度
            char conf_text[32];
            sprintf(conf_text, "%.2f", det.confidence);
            putText(ori_img, conf_text, Point(x1, y1 - 5),
                   FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
            
            // 绘制类别名称
            putText(ori_img, label, Point(x1, y1 - 25),
                   FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
        }
        
        // 保存结果
        imwrite(config.output_path, ori_img);
        cout << "\nResult saved to: " << config.output_path << endl;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return -1;
    }
    
    return 0;
}
