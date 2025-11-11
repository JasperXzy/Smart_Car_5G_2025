/**
 * @file ColorDetector.cpp
 * @brief 颜色检测模块
 */

#include "ColorDetector.hpp"
#include "Config.hpp"
#include <iostream>

namespace smartcar {

ColorDetector::ColorDetector()
    : initialized_(false) {
}

bool ColorDetector::init() {
    auto& config = Config::getInstance();
    if (!config.isLoaded()) {
        std::cerr << "[ColorDetector] Config not loaded" << std::endl;
        return false;
    }
    
    // 设置蓝色 HSV 范围
    blue_lower_ = cv::Scalar(
        config.blue_card.hsv_range.h_min,
        config.blue_card.hsv_range.s_min,
        config.blue_card.hsv_range.v_min
    );
    blue_upper_ = cv::Scalar(
        config.blue_card.hsv_range.h_max,
        config.blue_card.hsv_range.s_max,
        config.blue_card.hsv_range.v_max
    );
    
    // 设置红色 HSV 范围
    red_lower1_ = cv::Scalar(0, 43, 46);
    red_upper1_ = cv::Scalar(10, 255, 255);
    red_lower2_ = cv::Scalar(156, 43, 46);
    red_upper2_ = cv::Scalar(180, 255, 255);
    
    // 创建形态学核
    morph_kernel_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    
    initialized_ = true;
    std::cout << "[ColorDetector] Initialized" << std::endl;
    
    return true;
}

int ColorDetector::detectBlue(const cv::Mat& src,
                               std::vector<ColorDetection>& results,
                               int min_area) {
    return detectColor(src, blue_lower_, blue_upper_, results, min_area);
}

int ColorDetector::detectRed(const cv::Mat& src,
                             std::vector<ColorDetection>& results,
                             int min_area) {
    results.clear();
    
    // 转换到 HSV
    cv::cvtColor(src, hsv_, cv::COLOR_BGR2HSV);
    
    // 红色在 HSV 中分为两个区间
    cv::inRange(hsv_, red_lower1_, red_upper1_, mask1_);
    cv::inRange(hsv_, red_lower2_, red_upper2_, mask2_);
    
    // 合并两个掩码
    cv::bitwise_or(mask1_, mask2_, mask_);
    
    // 形态学操作去噪
    cv::morphologyEx(mask_, mask_, cv::MORPH_OPEN, morph_kernel_);
    cv::morphologyEx(mask_, mask_, cv::MORPH_CLOSE, morph_kernel_);
    
    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // 筛选符合条件的轮廓
    for (const auto& contour : contours) {
        int area = static_cast<int>(cv::contourArea(contour));
        if (area < min_area) continue;
        
        cv::Rect box = cv::boundingRect(contour);
        cv::Point center(box.x + box.width / 2, box.y + box.height / 2);
        
        results.emplace_back(box, area, center);
    }
    
    return static_cast<int>(results.size());
}

int ColorDetector::detectColor(const cv::Mat& src,
                                const cv::Scalar& hsv_lower,
                                const cv::Scalar& hsv_upper,
                                std::vector<ColorDetection>& results,
                                int min_area,
                                const cv::Rect& roi) {
    results.clear();
    
    // 转换到 HSV
    cv::cvtColor(src, hsv_, cv::COLOR_BGR2HSV);
    
    // 如果指定了 ROI，只处理 ROI 区域
    cv::Mat hsv_to_process = hsv_;
    cv::Point offset(0, 0);
    
    if (roi.width > 0 && roi.height > 0) {
        hsv_to_process = hsv_(roi);
        offset = cv::Point(roi.x, roi.y);
    }
    
    // 颜色阈值分割
    cv::inRange(hsv_to_process, hsv_lower, hsv_upper, mask_);
    
    // 形态学操作去噪
    cv::morphologyEx(mask_, mask_, cv::MORPH_OPEN, morph_kernel_);
    cv::morphologyEx(mask_, mask_, cv::MORPH_CLOSE, morph_kernel_);
    
    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // 筛选符合条件的轮廓
    for (const auto& contour : contours) {
        int area = static_cast<int>(cv::contourArea(contour));
        if (area < min_area) continue;
        
        cv::Rect box = cv::boundingRect(contour);
        
        // 调整坐标
        box.x += offset.x;
        box.y += offset.y;
        
        cv::Point center(box.x + box.width / 2, box.y + box.height / 2);
        
        results.emplace_back(box, area, center);
    }
    
    // 按面积从大到小排序
    std::sort(results.begin(), results.end(),
              [](const ColorDetection& a, const ColorDetection& b) {
                  return a.area > b.area;
              });
    
    return static_cast<int>(results.size());
}

void ColorDetector::drawResults(cv::Mat& img,
                                const std::vector<ColorDetection>& results,
                                const cv::Scalar& color) {
    for (const auto& result : results) {
        // 绘制边界框
        cv::rectangle(img, result.box, color, 2);
        
        // 绘制中心点
        cv::circle(img, result.center, 5, color, -1);
        
        // 绘制面积标签
        std::string label = "Area: " + std::to_string(result.area);
        cv::putText(img, label,
                   cv::Point(result.box.x, result.box.y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
}

} // namespace smartcar
