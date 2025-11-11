/**
 * @file RaceController.hpp
 * @brief 主控制器模块
 */

#pragma once

#include "Hardware.hpp"
#include "ImageProcess.hpp"
#include "AIDetector.hpp"
#include "ColorDetector.hpp"
#include "Navigation.hpp"
#include <opencv2/opencv.hpp>
#include <string>

namespace smartcar {

/**
 * @brief 比赛状态枚举
 */
enum class RaceState {
    WAIT_START,          // 等待开始
    LINE_FOLLOW,         // 巡线
    ZEBRA_STOP,          // 斑马线停车
    LANE_CHANGE,         // 变道
    OBSTACLE_AVOID,      // 避障
    PARKING_SEARCH,      // 寻找停车位
    PARKING_ENTER,       // 进入停车位
    FINISHED             // 完成
};

/**
 * @brief 比赛控制器类
 */
class RaceController {
public:
    static RaceController& getInstance();
    
    // 禁止拷贝和赋值
    RaceController(const RaceController&) = delete;
    RaceController& operator=(const RaceController&) = delete;
    
    /**
     * @brief 初始化控制器
     * @return true=成功, false=失败
     */
    bool init();
    
    /**
     * @brief 主循环
     * @return 退出码（0=正常，其他=错误）
     */
    int run();
    
    /**
     * @brief 处理一帧图像
     * @param frame 输入帧
     */
    void processFrame(cv::Mat& frame);
    
    /**
     * @brief 停止运行
     */
    void stop();
    
    /**
     * @brief 检查是否正在运行
     */
    bool isRunning() const { return running_; }
    
    /**
     * @brief 获取当前状态
     */
    RaceState getCurrentState() const { return current_state_; }
    
    /**
     * @brief 获取状态名称
     */
    std::string getStateName() const;
    
private:
    RaceController();
    ~RaceController() = default;
    
    bool initialized_;
    bool running_;
    
    // 当前状态
    RaceState current_state_;
    int frame_count_;
    
    // 各状态计数器
    int zebra_count_;
    int obstacle_count_;
    int parking_count_;
    int lane_change_direction_;  // 0=左，1=右
    
    // 摄像头
    cv::VideoCapture camera_;
    int camera_width_;
    int camera_height_;
    
    // 各模块引用
    Hardware* hardware_;
    ImageProcess image_proc_;
    MultiDetector* detector_;
    ColorDetector color_detector_;
    Navigation* navigation_;
    
    // 状态处理函数
    void stateWaitStart(cv::Mat& frame);
    void stateLineFollow(cv::Mat& frame);
    void stateZebraStop(cv::Mat& frame);
    void stateLaneChange(cv::Mat& frame);
    void stateObstacleAvoid(cv::Mat& frame);
    void stateParkingSearch(cv::Mat& frame);
    void stateParkingEnter(cv::Mat& frame);
    void stateFinished(cv::Mat& frame);
    
    // 辅助函数
    bool initCamera();
    void changeState(RaceState new_state);
    void drawStatus(cv::Mat& frame);
};

} // namespace smartcar
