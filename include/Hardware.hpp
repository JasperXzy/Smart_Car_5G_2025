/**
 * @file Hardware.hpp
 * @brief 硬件抽象层模块
 */

#pragma once

#include <string>

namespace smartcar {

/**
 * @brief 舵机控制类
 */
class Servo {
public:
    Servo();
    ~Servo();
    
    /**
     * @brief 初始化舵机
     * @param pin GPIO 引脚号
     * @param pwm_range PWM 范围
     * @param frequency PWM 频率
     * @param mid_value 中间值
     * @param min_value 最小值
     * @param max_value 最大值
     * @return true=成功, false=失败
     */
    bool init(int pin, float pwm_range, float frequency,
              float mid_value, float min_value, float max_value);
    
    /**
     * @brief 使用 PWM 值设置舵机角度
     * @param value PWM 值
     */
    void setValue(float value);
    
    /**
     * @brief 回到中间位置
     */
    void setMiddle();
    
    /**
     * @brief 检查是否已初始化
     */
    bool isInitialized() const { return initialized_; }
    
    /**
     * @brief 获取当前值
     */
    float getCurrentValue() const { return current_value_; }
    
private:
    int pin_;
    float pwm_range_;
    float frequency_;
    float mid_value_;
    float min_value_;
    float max_value_;
    float current_value_;
    bool initialized_;
};

/**
 * @brief 电机控制类
 */
class Motor {
public:
    Motor();
    ~Motor();
    
    /**
     * @brief 初始化电机
     * @param pin GPIO 引脚号
     * @param pwm_range PWM 范围
     * @param frequency PWM 频率
     * @param base_speed 基础速度
     * @return true=成功, false=失败
     */
    bool init(int pin, float pwm_range, float frequency, float base_speed);
    
    /**
     * @brief 设置电机速度
     * @param speed 速度值
     */
    void setSpeed(float speed);
    
    /**
     * @brief 停止电机
     */
    void stop();
    
    /**
     * @brief 检查是否已初始化
     */
    bool isInitialized() const { return initialized_; }
    
    /**
     * @brief 获取当前速度
     */
    float getCurrentSpeed() const { return current_speed_; }
    
private:
    int pin_;
    float pwm_range_;
    float frequency_;
    float base_speed_;
    float current_speed_;
    bool initialized_;
};

/**
 * @brief 硬件管理类
 * 
 * 统一管理所有硬件设备的初始化和控制
 */
class Hardware {
public:
    static Hardware& getInstance();
    
    // 禁止拷贝和赋值
    Hardware(const Hardware&) = delete;
    Hardware& operator=(const Hardware&) = delete;
    
    /**
     * @brief 初始化硬件
     * @return true=成功, false=失败
     */
    bool init();
    
    /**
     * @brief 清理硬件资源
     */
    void cleanup();
    
    /**
     * @brief 检查是否已初始化
     */
    bool isInitialized() const { return initialized_; }
    
    // 获取硬件设备
    Servo& getServo() { return servo_; }
    Motor& getMotor() { return motor_; }
    
private:
    Hardware();
    ~Hardware();
    
    bool initialized_;
    Servo servo_;
    Motor motor_;
};

} // namespace smartcar
