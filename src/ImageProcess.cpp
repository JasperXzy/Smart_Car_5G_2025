/**
 * @file ImageProcess.cpp
 * @brief 图像处理模块
 */

#include "ImageProcess.hpp"
#include <iostream>
#include <algorithm>

namespace smartcar {

ImageProcess::ImageProcess()
    : width_(0)
    , height_(0)
    , initialized_(false) {
}

bool ImageProcess::init(int width, int height) {
    width_ = width;
    height_ = height;
    
    // 预分配缓存矩阵
    temp_gray_ = cv::Mat(height, width, CV_8UC1);
    temp_binary_ = cv::Mat(height, width, CV_8UC1);
    temp_edges_ = cv::Mat(height, width, CV_8UC1);
    
    // 预留巡线数据空间
    mid_line_.reserve(50);
    left_line_.reserve(50);
    right_line_.reserve(50);
    last_mid_line_.reserve(50);
    
    initialized_ = true;
    std::cout << "[ImageProcess] Initialized (" << width << "x" << height << ")" << std::endl;
    
    return true;
}

void ImageProcess::sobelEdge(const cv::Mat& src, cv::Mat& dst, int ksize) {
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;
    
    // Sobel X 方向
    cv::Sobel(src, grad_x, CV_16S, 1, 0, ksize);
    cv::convertScaleAbs(grad_x, abs_grad_x);
    
    // Sobel Y方向
    cv::Sobel(src, grad_y, CV_16S, 0, 1, ksize);
    cv::convertScaleAbs(grad_y, abs_grad_y);
    
    // 合并
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
}

// 逐行巡线追踪
bool ImageProcess::tracking(const cv::Mat& binary_image, int start_row, int end_row) {
    // 参数检查
    if (binary_image.empty() || binary_image.type() != CV_8U) {
        std::cerr << "[ImageProcess] tracking: Input image invalid!" << std::endl;
        return false;
    }
    
    if (start_row < end_row || end_row < 0 || start_row >= binary_image.rows) {
        std::cerr << "[ImageProcess] tracking: Invalid row parameters!" << std::endl;
        return false;
    }
    
    // 确定搜索起始点
    int begin = binary_image.cols / 2; // 默认从图像中心开始
    
    // 如果上一帧有有效数据，使用上一帧中线的平均值作为起点
    if (!last_mid_line_.empty() && last_mid_line_.size() >= 20) {
        int sum_x = 0;
        int sample_count = std::min(20, static_cast<int>(last_mid_line_.size()));
        for (int i = 0; i < sample_count; ++i) {
            sum_x += last_mid_line_[i].x;
        }
        begin = sum_x / sample_count;
    }
    
    // 清空当前帧数据
    left_line_.clear();
    right_line_.clear();
    mid_line_.clear();
    
    // 逐行搜索，从 start_row 到 end_row
    for (int i = start_row; i >= end_row; --i) {
        int left = begin;
        int right = begin;
        bool left_found = false;
        bool right_found = false;
        
        const uchar* row = binary_image.ptr<uchar>(i);
        
        // 向左搜索左边界
        while (left > 1) {
            if (row[left] == 255 && row[left + 1] == 255) {
                left_found = true;
                left_line_.emplace_back(left, i);
                break;
            }
            --left;
        }
        if (!left_found) {
            left_line_.emplace_back(1, i); // 左边界未找到，记录边缘
        }
        
        // 向右搜索右边界
        while (right < binary_image.cols - 1) {
            if (row[right] == 255 && right >= 2 && row[right - 2] == 255) {
                right_found = true;
                right_line_.emplace_back(right, i);
                break;
            }
            ++right;
        }
        if (!right_found) {
            right_line_.emplace_back(binary_image.cols - 2, i); // 右边界未找到
        }
        
        // 计算中点
        const cv::Point& left_point = left_line_.back();
        const cv::Point& right_point = right_line_.back();
        int mid_x = (left_point.x + right_point.x) / 2;
        mid_line_.emplace_back(mid_x, i);
        
        // 更新下一行的搜索起点
        begin = mid_x;
    }
    
    // 保存当前中线供下一帧使用
    last_mid_line_ = mid_line_;
    
    return !mid_line_.empty();
}

// 避障模式的巡线追踪
bool ImageProcess::trackingObstacle(const cv::Mat& binary_image, int obstacle_top_y) {
    if (binary_image.empty() || binary_image.type() != CV_8U) {
        std::cerr << "[ImageProcess] trackingObstacle: 输入图像无效！" << std::endl;
        return false;
    }
    
    // 确定搜索下界
    int lower_bound = obstacle_top_y;
    if (lower_bound < 0 || lower_bound > 153) {
        lower_bound = 110; // 回落到安全搜索下限
    }
    
    // 使用常规 tracking，但限制搜索范围
    return tracking(binary_image, 153, lower_bound);
}

// 获取巡线中心点
int ImageProcess::getTrackingCenter(int sample_count) const {
    if (mid_line_.empty()) {
        return -1;
    }
    
    // 取前几行的平均值
    int valid_count = std::min(sample_count, static_cast<int>(mid_line_.size()));
    if (valid_count == 0) {
        return -1;
    }
    
    int sum_x = 0;
    for (int i = 0; i < valid_count; ++i) {
        sum_x += mid_line_[i].x;
    }
    
    return sum_x / valid_count;
}

// 在二值图上补线
cv::Mat ImageProcess::drawWhiteLine(const cv::Mat& binary_image, 
                                    const cv::Point& start, 
                                    const cv::Point& end, 
                                    int line_width) {
    cv::Mat result = binary_image.clone();
    
    int x1 = start.x, y1 = start.y;
    int x2 = end.x, y2 = end.y;
    
    if (x1 == x2) { // 垂直线
        for (int y = std::min(y1, y2); y <= std::max(y1, y2); ++y) {
            for (int i = -line_width / 2; i <= line_width / 2; ++i) {
                int new_x = x1 + i;
                if (new_x >= 0 && new_x < result.cols && y >= 0 && y < result.rows) {
                    result.at<uchar>(y, new_x) = 255;
                }
            }
        }
    } else { // 斜线
        double slope = static_cast<double>(y2 - y1) / (x2 - x1);
        double intercept = y1 - slope * x1;
        
        for (int x = std::min(x1, x2); x <= std::max(x1, x2); ++x) {
            int y = static_cast<int>(slope * x + intercept);
            for (int i = -line_width / 2; i <= line_width / 2; ++i) {
                int new_y = y + i;
                if (x >= 0 && x < result.cols && new_y >= 0 && new_y < result.rows) {
                    result.at<uchar>(new_y, x) = 255;
                }
            }
        }
    }
    
    return result;
}

int ImageProcess::detectZebraCrossing(const cv::Mat& src,
                                       const cv::Scalar& hsv_lower,
                                       const cv::Scalar& hsv_upper,
                                       const cv::Rect& roi) {
    // 转换到 HSV
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    
    // 提取 ROI
    cv::Mat roi_hsv = hsv(roi);
    
    // 颜色阈值分割
    cv::Mat mask;
    cv::inRange(roi_hsv, hsv_lower, hsv_upper, mask);
    
    // 形态学操作去噪
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    
    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // 统计符合条件的条纹
    int stripe_count = 0;
    for (const auto& contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);
        
        // 检查是否符合条纹特征（横向的矩形）
        if (rect.width > 10 && rect.height > 10 &&
            rect.width < 50 && rect.height < 50) {
            stripe_count++;
        }
    }
    
    return stripe_count;
}

void ImageProcess::morphologyOpen(const cv::Mat& src, cv::Mat& dst, int kernel_size) {
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, 
                                               cv::Size(kernel_size, kernel_size));
    cv::morphologyEx(src, dst, cv::MORPH_OPEN, kernel);
}

void ImageProcess::morphologyClose(const cv::Mat& src, cv::Mat& dst, int kernel_size) {
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                               cv::Size(kernel_size, kernel_size));
    cv::morphologyEx(src, dst, cv::MORPH_CLOSE, kernel);
}

void ImageProcess::gaussianBlur(const cv::Mat& src, cv::Mat& dst, int kernel_size) {
    // 确保 kernel_size 是奇数
    if (kernel_size % 2 == 0) kernel_size++;
    cv::GaussianBlur(src, dst, cv::Size(kernel_size, kernel_size), 0);
}

// 绘制巡线追踪结果
void ImageProcess::drawTrackingLines(cv::Mat& img, bool draw_mid, 
                                     bool draw_left, bool draw_right) {
    // 绘制左边界（蓝色）
    if (draw_left && !left_line_.empty()) {
        for (size_t i = 0; i < left_line_.size(); ++i) {
            cv::circle(img, left_line_[i], 2, cv::Scalar(255, 0, 0), -1);
        }
    }
    
    // 绘制右边界（红色）
    if (draw_right && !right_line_.empty()) {
        for (size_t i = 0; i < right_line_.size(); ++i) {
            cv::circle(img, right_line_[i], 2, cv::Scalar(0, 0, 255), -1);
        }
    }
    
    // 绘制中线（绿色）
    if (draw_mid && !mid_line_.empty()) {
        for (size_t i = 0; i < mid_line_.size() - 1; ++i) {
            cv::line(img, mid_line_[i], mid_line_[i + 1], 
                    cv::Scalar(0, 255, 0), 2);
        }
        
        // 绘制 PID 控制点（加粗）
        int pid_center = getTrackingCenter(3);
        if (pid_center > 0 && !mid_line_.empty()) {
            cv::circle(img, cv::Point(pid_center, mid_line_[0].y), 
                      8, cv::Scalar(0, 255, 255), -1);
        }
    }
}

void ImageProcess::undistort(const cv::Mat& src, cv::Mat& dst,
                             const cv::Mat& camera_matrix,
                             const cv::Mat& dist_coeffs) {
    cv::undistort(src, dst, camera_matrix, dist_coeffs);
}

} // namespace smartcar
