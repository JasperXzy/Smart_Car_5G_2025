/**
 * @file AIDetector.cpp
 * @brief AI检测模块
 */

#include "AIDetector.hpp"
#include "Config.hpp"
#include <net.h>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <float.h>

namespace smartcar {

// ==================== AIDetector ====================

AIDetector::AIDetector()
    : loaded_(false)
    , input_size_(352)
    , conf_threshold_(0.5f)
    , nms_threshold_(0.45f)
    , num_threads_(4)
    , inference_time_ms_(0.0f) {
    net_ = std::make_unique<ncnn::Net>();
}

AIDetector::~AIDetector() {
    if (net_) {
        net_->clear();
    }
}

bool AIDetector::loadModel(const std::string& param_path,
                           const std::string& bin_path,
                           int input_size,
                           float conf_thresh,
                           float nms_thresh,
                           int num_threads) {
    input_size_ = input_size;
    conf_threshold_ = conf_thresh;
    nms_threshold_ = nms_thresh;
    num_threads_ = num_threads;
    
    // 设置线程数
    net_->opt.num_threads = num_threads_;
    net_->opt.use_vulkan_compute = false;
    
    // 加载模型
    if (net_->load_param(param_path.c_str()) != 0) {
        std::cerr << "[AIDetector] Failed to load param: " << param_path << std::endl;
        return false;
    }
    
    if (net_->load_model(bin_path.c_str()) != 0) {
        std::cerr << "[AIDetector] Failed to load model: " << bin_path << std::endl;
        return false;
    }
    
    loaded_ = true;
    std::cout << "[AIDetector] Model loaded: " << param_path << std::endl;
    
    return true;
}

bool AIDetector::detect(const cv::Mat& img, std::vector<DetectionResult>& results) {
    if (!loaded_) {
        std::cerr << "[AIDetector] Model not loaded" << std::endl;
        return false;
    }
    
    results.clear();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // 预处理
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        img.data, ncnn::Mat::PIXEL_BGR, 
        img.cols, img.rows, 
        input_size_, input_size_);
    
    // 归一化
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);
    
    // 推理
    ncnn::Extractor ex = net_->create_extractor();
    ex.input("in0", in);
    
    ncnn::Mat out;
    ex.extract("out0", out);
    
    // 后处理
    postprocess(out, img, results);
    
    // NMS
    nms(results);
    
    auto end = std::chrono::high_resolution_clock::now();
    inference_time_ms_ = std::chrono::duration<float, std::milli>(end - start).count();
    
    return true;
}

void AIDetector::postprocess(const ncnn::Mat& out,
                             const cv::Mat& img,
                             std::vector<DetectionResult>& results) {
    
    const int channels = out.c;
    const int height = out.h;
    const int width = out.w;
    
    const int num_classes = channels - 5; // 类别数量
    
    int img_w = img.cols;
    int img_h = img.rows;
    
    // 遍历特征图的每个位置
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int base_idx = i * width + j;
            
            // 从通道0获取前背景分数
            const float* pobj_ptr = out.channel(0);
            float obj_score = pobj_ptr[base_idx];
            
            // 提前退出
            if (obj_score < conf_threshold_) {
                continue;
            }
            
            // 从通道1-4获取边界框回归值
            const float* preg_ptr0 = out.channel(1);
            const float* preg_ptr1 = out.channel(2);
            const float* preg_ptr2 = out.channel(3);
            const float* preg_ptr3 = out.channel(4);
            
            float tx = preg_ptr0[base_idx];
            float ty = preg_ptr1[base_idx];
            float tw = preg_ptr2[base_idx];
            float th = preg_ptr3[base_idx];
            
            // 从通道5+找到最大类别分数和对应索引
            int class_index = 0;
            float class_score = -FLT_MAX;
            for (int k = 0; k < num_classes; k++) {
                const float* pcls_ptr = out.channel(5 + k);
                float score = pcls_ptr[base_idx];
                if (score > class_score) {
                    class_score = score;
                    class_index = k;
                }
            }
            
            // 计算置信度: obj^0.6 * cls^0.4
            float confidence = pow(obj_score, 0.6f) * pow(class_score, 0.4f);
            
            if (confidence < conf_threshold_) {
                continue;
            }
            
            // 解码边界框
            float bcx = (tanh(tx) + j) / width;
            float bcy = (tanh(ty) + i) / height;
            float bw = 1.0f / (1.0f + exp(-tw));  // sigmoid
            float bh = 1.0f / (1.0f + exp(-th));  // sigmoid
            
            // 转换为 x1, y1, x2, y2
            float x1 = bcx - 0.5f * bw;
            float y1 = bcy - 0.5f * bh;
            float x2 = bcx + 0.5f * bw;
            float y2 = bcy + 0.5f * bh;
            
            // 裁剪到 [0, 1]
            x1 = std::max(0.0f, std::min(1.0f, x1));
            y1 = std::max(0.0f, std::min(1.0f, y1));
            x2 = std::max(0.0f, std::min(1.0f, x2));
            y2 = std::max(0.0f, std::min(1.0f, y2));
            
            // 转换为像素坐标
            x1 *= img_w;
            y1 *= img_h;
            x2 *= img_w;
            y2 *= img_h;
            
            // 转换为整数坐标
            int ix1 = static_cast<int>(x1);
            int iy1 = static_cast<int>(y1);
            int ix2 = static_cast<int>(x2);
            int iy2 = static_cast<int>(y2);
            
            cv::Rect box(ix1, iy1, ix2 - ix1, iy2 - iy1);
            results.emplace_back(class_index, confidence, box);
        }
    }
}

void AIDetector::nms(std::vector<DetectionResult>& detections) {
    // 快速排序
    if (detections.size() > 1) {
        qsort_descent_inplace(detections, 0, detections.size() - 1);
    }
    
    std::vector<bool> suppressed(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;
        
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;
            
            // 类别内 NMS
            if (detections[i].label != detections[j].label)
                continue;
            
            // 计算 IoU
            cv::Rect& box1 = detections[i].box;
            cv::Rect& box2 = detections[j].box;
            
            int x1 = std::max(box1.x, box2.x);
            int y1 = std::max(box1.y, box2.y);
            int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
            int y2 = std::min(box1.y + box1.height, box2.y + box2.height);
            
            int inter_area = std::max(0, x2 - x1) * std::max(0, y2 - y1);
            int area1 = box1.width * box1.height;
            int area2 = box2.width * box2.height;
            float iou = static_cast<float>(inter_area) / (area1 + area2 - inter_area);
            
            if (iou > nms_threshold_) {
                suppressed[j] = true;
            }
        }
    }
    
    // 移除被抑制的检测
    std::vector<DetectionResult> filtered;
    for (size_t i = 0; i < detections.size(); i++) {
        if (!suppressed[i]) {
            filtered.push_back(detections[i]);
        }
    }
    
    detections = filtered;
}

void AIDetector::qsort_descent_inplace(std::vector<DetectionResult>& detections, int left, int right) {
    int i = left;
    int j = right;
    float p = detections[(left + right) / 2].confidence;

    while (i <= j) {
        while (detections[i].confidence > p)
            i++;

        while (detections[j].confidence < p)
            j--;

        if (i <= j) {
            std::swap(detections[i], detections[j]);
            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(detections, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(detections, i, right);
        }
    }
}

void AIDetector::drawResults(cv::Mat& img, const std::vector<DetectionResult>& results) {
    for (const auto& result : results) {
        // 绘制边界框
        cv::rectangle(img, result.box, cv::Scalar(0, 255, 0), 2);
        
        // 绘制标签
        std::string label = "Class" + std::to_string(result.label) + 
                           ": " + std::to_string(static_cast<int>(result.confidence * 100)) + "%";
        
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        
        cv::rectangle(img, 
                     cv::Point(result.box.x, result.box.y - text_size.height - 5),
                     cv::Point(result.box.x + text_size.width, result.box.y),
                     cv::Scalar(0, 255, 0), -1);
        
        cv::putText(img, label, 
                   cv::Point(result.box.x, result.box.y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
    
    // 显示 FPS
    std::string fps_text = "Inference: " + std::to_string(static_cast<int>(inference_time_ms_)) + " ms";
    cv::putText(img, fps_text, cv::Point(10, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
}

// ==================== MultiDetector ====================

MultiDetector::MultiDetector()
    : initialized_(false) {
}

MultiDetector& MultiDetector::getInstance() {
    static MultiDetector instance;
    return instance;
}

bool MultiDetector::init() {
    if (initialized_) {
        std::cout << "[MultiDetector] Already initialized" << std::endl;
        return true;
    }
    
    std::cout << "[MultiDetector] Initializing..." << std::endl;
    
    auto& config = Config::getInstance();
    if (!config.isLoaded()) {
        std::cerr << "[MultiDetector] Config not loaded" << std::endl;
        return false;
    }
    
    // 加载障碍物检测模型
    if (!obstacle_detector_.loadModel(
        config.models.obstacle.param_path,
        config.models.obstacle.bin_path,
        config.models.obstacle.input_size,
        config.models.obstacle.conf_threshold,
        config.models.obstacle.nms_threshold,
        config.models.obstacle.num_threads)) {
        std::cerr << "[MultiDetector] Failed to load obstacle detector" << std::endl;
        return false;
    }
    
    // 加载方向检测模型
    if (!direction_detector_.loadModel(
        config.models.direction.param_path,
        config.models.direction.bin_path,
        config.models.direction.input_size,
        config.models.direction.conf_threshold,
        config.models.direction.nms_threshold,
        config.models.direction.num_threads)) {
        std::cerr << "[MultiDetector] Failed to load direction detector" << std::endl;
        return false;
    }
    
    // 加载停车位检测模型
    if (!parking_detector_.loadModel(
        config.models.parking.param_path,
        config.models.parking.bin_path,
        config.models.parking.input_size,
        config.models.parking.conf_threshold,
        config.models.parking.nms_threshold,
        config.models.parking.num_threads)) {
        std::cerr << "[MultiDetector] Failed to load parking detector" << std::endl;
        return false;
    }
    
    initialized_ = true;
    std::cout << "[MultiDetector] Initialization complete" << std::endl;
    
    return true;
}

} // namespace smartcar
