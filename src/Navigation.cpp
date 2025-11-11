/**
 * @file Navigation.cpp
 * @brief 导航控制模块
 */

#include "Navigation.hpp"
#include "Config.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <cmath>

namespace smartcar {

// ==================== PIDController ====================

PIDController::PIDController()
    : kp_(0.0f)
    , kd_(0.0f)
    , ki_(0.0f)
    , prev_error_(0.0f)
    , integral_(0.0f)
    , first_run_(true) {
}

void PIDController::setParams(float kp, float kd, float ki) {
    kp_ = kp;
    kd_ = kd;
    ki_ = ki;
}

float PIDController::compute(float error) {
    if (first_run_) {
        prev_error_ = error;
        first_run_ = false;
    }
    
    // 比例项
    float p_term = kp_ * error;
    
    // 微分项
    float d_term = kd_ * (error - prev_error_);
    
    // 积分项
    integral_ += error;
    float i_term = ki_ * integral_;
    
    // 积分限幅
    const float integral_max = 1000.0f;
    if (i_term > integral_max) i_term = integral_max;
    if (i_term < -integral_max) i_term = -integral_max;
    
    prev_error_ = error;
    
    return p_term + d_term + i_term;
}

void PIDController::reset() {
    prev_error_ = 0.0f;
    integral_ = 0.0f;
    first_run_ = true;
}

// ==================== Navigation ====================

Navigation::Navigation()
    : initialized_(false)
    , hardware_(nullptr)
    , current_servo_(0.0f)
    , current_speed_(0.0f)
    , servo_mid_(690.0f)
    , servo_min_(580.0f)
    , servo_max_(1000.0f) {
}

Navigation& Navigation::getInstance() {
    static Navigation instance;
    return instance;
}

bool Navigation::init() {
    if (initialized_) {
        std::cout << "[Navigation] Already initialized" << std::endl;
        return true;
    }
    
    std::cout << "[Navigation] Initializing..." << std::endl;
    
    // 获取硬件引用
    hardware_ = &Hardware::getInstance();
    if (!hardware_->isInitialized()) {
        std::cerr << "[Navigation] Hardware not initialized" << std::endl;
        return false;
    }
    
    // 获取配置
    auto& config = Config::getInstance();
    if (!config.isLoaded()) {
        std::cerr << "[Navigation] Config not loaded" << std::endl;
        return false;
    }
    
    // 设置舵机参数
    servo_mid_ = config.hardware.servo.mid_value;
    servo_min_ = config.hardware.servo.min_value;
    servo_max_ = config.hardware.servo.max_value;
    
    // 设置 PID 参数
    pid_normal_.setParams(
        config.line_tracking.pid_normal.kp,
        config.line_tracking.pid_normal.kd,
        config.line_tracking.pid_normal.ki
    );
    
    pid_obstacle_.setParams(
        config.obstacle_avoidance.turn_angle_base / 100.0f,  // 简化的 PID 参数
        config.obstacle_avoidance.turn_angle_base / 100.0f,
        0.0f
    );
    
    pid_parking_.setParams(3.0f, 3.0f, 0.0f);  // 停车用的 PID
    
    // 初始化状态：舵机回中，电机停止
    current_servo_ = servo_mid_;
    current_speed_ = 0.0f;
    
    initialized_ = true;
    std::cout << "[Navigation] Initialization complete" << std::endl;
    
    return true;
}

void Navigation::lineFollowing(int line_center, int image_width, int speed) {
    if (!initialized_) {
        std::cerr << "[Navigation] Not initialized" << std::endl;
        return;
    }
    
    // 计算误差
    int target_center = image_width / 2;
    float error = static_cast<float>(line_center - target_center);
    
    // PID计算
    float control = pid_normal_.compute(error);
    
    // 转换为舵机角度
    float servo_value = servo_mid_ - control;
    
    // 限制范围
    if (servo_value < servo_min_) servo_value = servo_min_;
    if (servo_value > servo_max_) servo_value = servo_max_;
    
    // 应用控制
    setServo(servo_value);
    setMotor(static_cast<float>(speed));
}

void Navigation::obstacleAvoidance(const cv::Rect& obstacle_box, int image_width) {
    if (!initialized_) {
        std::cerr << "[Navigation] Not initialized" << std::endl;
        return;
    }
    
    auto& config = Config::getInstance();
    
    // 计算障碍物中心
    int obstacle_center = obstacle_box.x + obstacle_box.width / 2;
    int target_center = image_width / 2;
    
    // 计算误差
    float error = static_cast<float>(obstacle_center - target_center);
    
    // PID 计算
    float control = pid_obstacle_.compute(error);
    
    // 根据障碍物位置决定避让方向
    float servo_value;
    if (obstacle_center < target_center) {
        // 障碍物在左边，向右转
        servo_value = servo_mid_ + config.obstacle_avoidance.turn_angle_base;
    } else {
        // 障碍物在右边，向左转
        servo_value = servo_mid_ - config.obstacle_avoidance.turn_angle_base;
    }
    
    // 限制范围
    if (servo_value < servo_min_) servo_value = servo_min_;
    if (servo_value > servo_max_) servo_value = servo_max_;
    
    // 应用控制
    setServo(servo_value);
    setMotor(static_cast<float>(config.obstacle_avoidance.avoid_speed));
}

void Navigation::parkingAlign(int parking_center, int image_width) {
    if (!initialized_) {
        std::cerr << "[Navigation] Not initialized" << std::endl;
        return;
    }
    
    auto& config = Config::getInstance();
    
    // 计算误差
    int target_center = image_width / 2;
    float error = static_cast<float>(parking_center - target_center);
    
    // PID 计算
    float control = pid_parking_.compute(error);
    
    // 转换为舵机角度
    float servo_value = servo_mid_ - control;
    
    // 限制范围
    if (servo_value < servo_min_) servo_value = servo_min_;
    if (servo_value > servo_max_) servo_value = servo_max_;
    
    // 应用控制
    setServo(servo_value);
    setMotor(static_cast<float>(config.parking.approach_speed));
}

void Navigation::laneChange(int direction) {
    if (!initialized_) {
        std::cerr << "[Navigation] Not initialized" << std::endl;
        return;
    }
    
    auto& config = Config::getInstance();
    
    std::cout << "[Navigation] Lane change: " << (direction == 0 ? "LEFT" : "RIGHT") << std::endl;
    
    if (direction == 0) {
        // 左转
        setServo(config.lane_change.turn_angle_left);
        setMotor(static_cast<float>(config.lane_change.change_speed));
        std::this_thread::sleep_for(std::chrono::milliseconds(config.lane_change.turn_duration_ms));
        
        // 回正
        setServo(servo_mid_);
    } else {
        // 右转
        setServo(config.lane_change.turn_angle_right);
        setMotor(static_cast<float>(config.lane_change.change_speed));
        std::this_thread::sleep_for(std::chrono::milliseconds(config.lane_change.turn_duration_ms));
        
        // 回正
        setServo(servo_mid_);
    }
}

void Navigation::goStraight(int speed, int duration_ms) {
    if (!initialized_) {
        std::cerr << "[Navigation] Not initialized" << std::endl;
        return;
    }
    
    setServo(servo_mid_);
    setMotor(static_cast<float>(speed));
    
    if (duration_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
    }
}

void Navigation::stop() {
    if (!initialized_) {
        std::cerr << "[Navigation] Not initialized" << std::endl;
        return;
    }
    
    setServo(servo_mid_);
    setMotor(0.0f);
    
    // 重置 PID 状态
    pid_normal_.reset();
    pid_obstacle_.reset();
    pid_parking_.reset();
}

void Navigation::setServo(float angle) {
    if (!hardware_) return;
    
    current_servo_ = angle;
    hardware_->getServo().setValue(angle);
}

void Navigation::setMotor(float speed) {
    if (!hardware_) return;
    
    current_speed_ = speed;
    hardware_->getMotor().setSpeed(speed);
}

} // namespace smartcar
