/**
 * @file Navigation.hpp
 * @brief 导航控制模块
 */

#pragma once

#include "Hardware.hpp"
#include <opencv2/opencv.hpp>

namespace smartcar {

/**
 * @brief PID 控制器
 */
class PIDController {
public:
    PIDController();
    
    /**
     * @brief 设置 PID 参数
     * @param kp 比例系数
     * @param kd 微分系数
     * @param ki 积分系数
     */
    void setParams(float kp, float kd, float ki);
    
    /**
     * @brief 计算控制量
     * @param error 当前误差
     * @return 控制量
     */
    float compute(float error);
    
    /**
     * @brief 重置 PID 状态
     */
    void reset();
    
private:
    float kp_;
    float kd_;
    float ki_;
    
    float prev_error_;
    float integral_;
    bool first_run_;
};

/**
 * @brief 导航控制器
 */
class Navigation {
public:
    static Navigation& getInstance();
    
    // 禁止拷贝和赋值
    Navigation(const Navigation&) = delete;
    Navigation& operator=(const Navigation&) = delete;
    
    /**
     * @brief 初始化导航控制器
     * @return true=成功, false=失败
     */
    bool init();
    
    /**
     * @brief 巡线控制
     * @param line_center 检测到的线中心位置
     * @param image_width 图像宽度
     * @param speed 速度
     */
    void lineFollowing(int line_center, int image_width, int speed);
    
    /**
     * @brief 避障控制
     * @param obstacle_box 障碍物边界框
     * @param image_width 图像宽度
     */
    void obstacleAvoidance(const cv::Rect& obstacle_box, int image_width);
    
    /**
     * @brief 停车控制
     * @param parking_center 停车位中心
     * @param image_width 图像宽度
     */
    void parkingAlign(int parking_center, int image_width);
    
    /**
     * @brief 变道控制
     * @param direction 方向（0=左，1=右）
     */
    void laneChange(int direction);
    
    /**
     * @brief 直行
     * @param speed 速度
     * @param duration_ms 持续时间（ms），0表示持续
     */
    void goStraight(int speed, int duration_ms = 0);
    
    /**
     * @brief 停车
     */
    void stop();
    
    /**
     * @brief 设置舵机角度
     * @param angle 角度值
     */
    void setServo(float angle);
    
    /**
     * @brief 设置电机速度
     * @param speed 速度值
     */
    void setMotor(float speed);
    
    /**
     * @brief 获取当前舵机角度
     */
    float getCurrentServo() const { return current_servo_; }
    
    /**
     * @brief 获取当前电机速度
     */
    float getCurrentSpeed() const { return current_speed_; }
    
private:
    Navigation();
    ~Navigation() = default;
    
    bool initialized_;
    
    // PID 控制器
    PIDController pid_normal_;
    PIDController pid_obstacle_;
    PIDController pid_parking_;
    
    // 硬件引用
    Hardware* hardware_;
    
    // 当前状态
    float current_servo_;
    float current_speed_;
    
    // 配置参数
    float servo_mid_;
    float servo_min_;
    float servo_max_;
};

} // namespace smartcar
