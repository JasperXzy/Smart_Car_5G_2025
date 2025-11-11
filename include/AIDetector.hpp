/**
 * @file AIDetector.hpp
 * @brief AI检测模块
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include <net.h>

namespace smartcar {

/**
 * @brief 检测结果结构体
 */
struct DetectionResult {
    int label;              // 类别标签
    float confidence;       // 置信度
    cv::Rect box;           // 边界框
    
    DetectionResult() : label(0), confidence(0.0f) {}
    DetectionResult(int l, float conf, const cv::Rect& b)
        : label(l), confidence(conf), box(b) {}
};

/**
 * @brief AI检测器类
 */
class AIDetector {
public:
    AIDetector();
    ~AIDetector();
    
    /**
     * @brief 加载模型
     * @param param_path .param文件路径
     * @param bin_path .bin文件路径
     * @param input_size 输入尺寸
     * @param conf_thresh 置信度阈值
     * @param nms_thresh NMS阈值
     * @param num_threads 线程数
     * @return true=成功, false=失败
     */
    bool loadModel(const std::string& param_path,
                   const std::string& bin_path,
                   int input_size = 352,
                   float conf_thresh = 0.5f,
                   float nms_thresh = 0.45f,
                   int num_threads = 4);
    
    /**
     * @brief 执行检测
     * @param img 输入图像
     * @param results 检测结果
     * @return true=成功, false=失败
     */
    bool detect(const cv::Mat& img, std::vector<DetectionResult>& results);
    
    /**
     * @brief 绘制检测结果
     * @param img 图像
     * @param results 检测结果
     */
    void drawResults(cv::Mat& img, const std::vector<DetectionResult>& results);
    
    /**
     * @brief 检查是否已加载模型
     */
    bool isLoaded() const { return loaded_; }
    
    /**
     * @brief 获取推理时间（ms）
     */
    float getInferenceTime() const { return inference_time_ms_; }
    
private:
    std::unique_ptr<ncnn::Net> net_;
    bool loaded_;
    
    int input_size_;
    float conf_threshold_;
    float nms_threshold_;
    int num_threads_;
    
    float inference_time_ms_;
    
    // NMS处理
    void nms(std::vector<DetectionResult>& detections);
    
    // 快速排序
    void qsort_descent_inplace(std::vector<DetectionResult>& detections, int left, int right);
    
    // 后处理
    void postprocess(const ncnn::Mat& out, 
                     const cv::Mat& img,
                     std::vector<DetectionResult>& results);
};

/**
 * @brief 多模型检测器管理类
 */
class MultiDetector {
public:
    static MultiDetector& getInstance();
    
    // 禁止拷贝和赋值
    MultiDetector(const MultiDetector&) = delete;
    MultiDetector& operator=(const MultiDetector&) = delete;
    
    /**
     * @brief 初始化所有检测器
     * @return true=成功, false=失败
     */
    bool init();
    
    /**
     * @brief 障碍物检测
     */
    AIDetector& getObstacleDetector() { return obstacle_detector_; }
    
    /**
     * @brief 方向检测
     */
    AIDetector& getDirectionDetector() { return direction_detector_; }
    
    /**
     * @brief 停车位检测
     */
    AIDetector& getParkingDetector() { return parking_detector_; }
    
    /**
     * @brief 检查是否已初始化
     */
    bool isInitialized() const { return initialized_; }
    
private:
    MultiDetector();
    ~MultiDetector() = default;
    
    bool initialized_;
    AIDetector obstacle_detector_;
    AIDetector direction_detector_;
    AIDetector parking_detector_;
};

} // namespace smartcar
