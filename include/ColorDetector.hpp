/**
 * @file ColorDetector.hpp
 * @brief 颜色检测模块
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace smartcar {

/**
 * @brief 颜色检测结果
 */
struct ColorDetection {
    cv::Rect box;           // 边界框
    int area;               // 面积
    cv::Point center;       // 中心点
    
    ColorDetection() : area(0) {}
    ColorDetection(const cv::Rect& b, int a, const cv::Point& c)
        : box(b), area(a), center(c) {}
};

/**
 * @brief 颜色检测器类
 */
class ColorDetector {
public:
    ColorDetector();
    ~ColorDetector() = default;
    
    /**
     * @brief 初始化颜色检测器
     * @return true=成功, false=失败
     */
    bool init();
    
    /**
     * @brief 检测蓝色区域
     * @param src 源图像
     * @param results 检测结果
     * @param min_area 最小面积阈值
     * @return 检测到的数量
     */
    int detectBlue(const cv::Mat& src,
                   std::vector<ColorDetection>& results,
                   int min_area = 500);
    
    /**
     * @brief 检测红色区域
     * @param src 源图像
     * @param results 检测结果
     * @param min_area 最小面积阈值
     * @return 检测到的数量
     */
    int detectRed(const cv::Mat& src,
                  std::vector<ColorDetection>& results,
                  int min_area = 1000);
    
    /**
     * @brief 通用颜色检测
     * @param src 源图像
     * @param hsv_lower HSV 下界
     * @param hsv_upper HSV 上界
     * @param results 检测结果
     * @param min_area 最小面积阈值
     * @param roi 感兴趣区域
     * @return 检测到的数量
     */
    int detectColor(const cv::Mat& src,
                    const cv::Scalar& hsv_lower,
                    const cv::Scalar& hsv_upper,
                    std::vector<ColorDetection>& results,
                    int min_area = 100,
                    const cv::Rect& roi = cv::Rect());
    
    /**
     * @brief 绘制检测结果
     * @param img 图像
     * @param results 检测结果
     * @param color 绘制颜色
     */
    void drawResults(cv::Mat& img,
                    const std::vector<ColorDetection>& results,
                    const cv::Scalar& color = cv::Scalar(0, 255, 0));
    
    /**
     * @brief 获取掩码图像
     */
    const cv::Mat& getMask() const { return mask_; }
    
private:
    bool initialized_;
    
    // 蓝色 HSV 范围
    cv::Scalar blue_lower_;
    cv::Scalar blue_upper_;
    
    // 红色 HSV 范围
    cv::Scalar red_lower1_;
    cv::Scalar red_upper1_;
    cv::Scalar red_lower2_;
    cv::Scalar red_upper2_;
    
    // 临时存储
    cv::Mat hsv_;
    cv::Mat mask_;
    cv::Mat mask1_;
    cv::Mat mask2_;
    
    // 形态学核
    cv::Mat morph_kernel_;
};

} // namespace smartcar
