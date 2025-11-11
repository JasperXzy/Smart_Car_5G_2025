/**
 * @file ImageProcess.hpp
 * @brief 图像处理模块
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace smartcar {

/**
 * @brief 图像处理类
 */
class ImageProcess {
public:
    ImageProcess();
    ~ImageProcess() = default;
    
    /**
     * @brief 初始化图像处理器
     * @param width 图像宽度
     * @param height 图像高度
     * @return true=成功, false=失败
     */
    bool init(int width, int height);
    
    /**
     * @brief Sobel 边缘检测与车道线提取
     * @param frame 源图像
     * @param debug_overlay 调试覆盖图
     * @return 二值化的车道线图像
     */
    cv::Mat imageSobel(const cv::Mat& frame, cv::Mat* debug_overlay = nullptr);
    
    /**
     * @brief 逐行巡线追踪
     * @param binary_image 二值化图像（白色为车道线）
     * @param start_row 起始行号
     * @param end_row 结束行号
     * @return true=成功, false=失败
     */
    bool tracking(const cv::Mat& binary_image, int start_row = 153, int end_row = 110);
    
    /**
     * @brief 避障模式的巡线追踪
     * @param binary_image 二值化图像
     * @param obstacle_top_y 障碍物顶部 Y 坐标
     * @return true=成功, false=失败
     */
    bool trackingObstacle(const cv::Mat& binary_image, int obstacle_top_y);
    
    /**
     * @brief 获取中线点集
     * @return 中线点集合
     */
    const std::vector<cv::Point>& getMidLine() const { return mid_line_; }
    
    /**
     * @brief 获取左边界点集
     * @return 左边界点集合
     */
    const std::vector<cv::Point>& getLeftLine() const { return left_line_; }
    
    /**
     * @brief 获取右边界点集
     * @return 右边界点集合
     */
    const std::vector<cv::Point>& getRightLine() const { return right_line_; }
    
    /**
     * @brief 获取巡线中心点
     * @param sample_count 采样点数
     * @return 中心点 X 坐标，-1表示无效
     */
    int getTrackingCenter(int sample_count = 3) const;
    
    /**
     * @brief 在二值图上补线
     * @param binary_image 二值图
     * @param start 起点
     * @param end 终点
     * @param line_width 线宽
     * @return 补线后的图像
     */
    cv::Mat drawWhiteLine(const cv::Mat& binary_image, 
                         const cv::Point& start, 
                         const cv::Point& end, 
                         int line_width = 8);
    
    /**
     * @brief 斑马线检测
     * @param src 源图像
     * @param hsv_lower HSV 下界
     * @param hsv_upper HSV 上界
     * @param roi 感兴趣区域 (x, y, width, height)
     * @return 检测到的白色条纹数量
     */
    int detectZebraCrossing(const cv::Mat& src, 
                           const cv::Scalar& hsv_lower,
                           const cv::Scalar& hsv_upper,
                           const cv::Rect& roi);
    
    /**
     * @brief 开运算
     * @param src 源图像
     * @param dst 输出图像
     * @param kernel_size 核大小
     */
    void morphologyOpen(const cv::Mat& src, cv::Mat& dst, int kernel_size = 5);
    
    /**
     * @brief 闭运算
     * @param src 源图像
     * @param dst 输出图像
     * @param kernel_size 核大小
     */
    void morphologyClose(const cv::Mat& src, cv::Mat& dst, int kernel_size = 5);
    
    /**
     * @brief 高斯模糊
     * @param src 源图像
     * @param dst 输出图像
     * @param kernel_size 核大小
     */
    void gaussianBlur(const cv::Mat& src, cv::Mat& dst, int kernel_size = 5);
    
    /**
     * @brief 绘制巡线追踪结果
     * @param img 图像
     * @param draw_mid 是否绘制中线
     * @param draw_left 是否绘制左线
     * @param draw_right 是否绘制右线
     */
    void drawTrackingLines(cv::Mat& img, bool draw_mid = true, 
                          bool draw_left = true, bool draw_right = true);
    
    /**
     * @brief 图像去畸变
     * @param src 源图像
     * @param dst 输出图像
     * @param camera_matrix 相机内参矩阵
     * @param dist_coeffs 畸变系数
     */
    void undistort(const cv::Mat& src, cv::Mat& dst,
                   const cv::Mat& camera_matrix,
                   const cv::Mat& dist_coeffs);
    
private:
    int width_;
    int height_;
    bool initialized_;
    
    // 缓存用的矩阵
    cv::Mat temp_gray_;
    cv::Mat temp_binary_;
    cv::Mat temp_edges_;
    
    // 巡线数据
    std::vector<cv::Point> mid_line_;       // 中线点集
    std::vector<cv::Point> left_line_;      // 左边界点集
    std::vector<cv::Point> right_line_;     // 右边界点集
    std::vector<cv::Point> last_mid_line_;  // 上一帧中线
    
    // ImageSobel 相关常量
    static constexpr int MIN_COMPONENT_AREA = 400;  // 连通域最小面积
};

} // namespace smartcar
