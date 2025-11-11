/**
 * @file ImageProcess.cpp
 * @brief 图像处理模块
 */

#include "ImageProcess.hpp"
#include <iostream>

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

int ImageProcess::findLineCenter(const cv::Mat& gray, int search_y, int search_start_x) {
    if (search_y < 0 || search_y >= gray.rows) {
        return -1;
    }
    
    // 从指定位置向左右搜索黑线边缘
    int left_edge = -1;
    int right_edge = -1;
    
    const uchar* row = gray.ptr<uchar>(search_y);
    
    // 向左搜索左边缘
    for (int x = search_start_x; x >= 0; x--) {
        if (row[x] < 100) {  // 黑色阈值
            left_edge = x;
            break;
        }
    }
    
    // 向右搜索右边缘
    for (int x = search_start_x; x < gray.cols; x++) {
        if (row[x] < 100) {  // 黑色阈值
            right_edge = x;
            break;
        }
    }
    
    // 如果找到两条边，计算中心
    if (left_edge > 0 && right_edge > 0 && right_edge > left_edge) {
        return (left_edge + right_edge) / 2;
    }
    
    // 只找到一条边
    if (left_edge > 0) return left_edge;
    if (right_edge > 0) return right_edge;
    
    return -1;  // 未找到
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

void ImageProcess::drawLineCenter(cv::Mat& img, int center_x, int search_y) {
    if (center_x > 0) {
        // 绘制中心点
        cv::circle(img, cv::Point(center_x, search_y), 5, cv::Scalar(0, 255, 0), -1);
        
        // 绘制十字线
        cv::line(img, cv::Point(center_x, 0), cv::Point(center_x, img.rows),
                cv::Scalar(0, 255, 0), 1);
    }
    
    // 绘制搜索行
    cv::line(img, cv::Point(0, search_y), cv::Point(img.cols, search_y),
            cv::Scalar(255, 0, 0), 1);
}

void ImageProcess::undistort(const cv::Mat& src, cv::Mat& dst,
                             const cv::Mat& camera_matrix,
                             const cv::Mat& dist_coeffs) {
    cv::undistort(src, dst, camera_matrix, dist_coeffs);
}

} // namespace smartcar
