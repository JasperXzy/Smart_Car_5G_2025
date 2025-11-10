/**
 * @file Hardware.cpp
 * @brief 硬件抽象层模块
 */

#include "Hardware.hpp"
#include "Config.hpp"
#include <pigpio.h>
#include <iostream>

namespace smartcar {

// ==================== Servo 实现 ====================

Servo::Servo() 
    : pin_(0)
    , pwm_range_(0.0f)
    , frequency_(0.0f)
    , mid_value_(0.0f)
    , min_value_(0.0f)
    , max_value_(0.0f)
    , current_value_(0.0f)
    , initialized_(false) {
}

Servo::~Servo() {
    if (initialized_) {
        setMiddle();
    }
}

bool Servo::init(int pin, float pwm_range, float frequency,
                 float mid_value, float min_value, float max_value) {
    pin_ = pin;
    pwm_range_ = pwm_range;
    frequency_ = frequency;
    mid_value_ = mid_value;
    min_value_ = min_value;
    max_value_ = max_value;
    current_value_ = mid_value;
    
    // 设置 PWM 频率
    gpioSetPWMfrequency(pin_, static_cast<unsigned>(frequency_));
    gpioSetPWMrange(pin_, static_cast<unsigned>(pwm_range_));
    
    // 设置到中间位置
    setMiddle();
    
    initialized_ = true;
    std::cout << "[Servo] Initialized on pin " << pin_ << std::endl;
    
    return true;
}

void Servo::setValue(float value) {
    if (!initialized_) {
        std::cerr << "[Servo] Not initialized" << std::endl;
        return;
    }
    
    // 限制范围
    if (value < min_value_) value = min_value_;
    if (value > max_value_) value = max_value_;
    
    current_value_ = value;
    gpioPWM(pin_, static_cast<unsigned>(value));
}

void Servo::setMiddle() {
    setValue(mid_value_);
}

// ==================== Motor 实现 ====================

Motor::Motor()
    : pin_(0)
    , pwm_range_(0.0f)
    , frequency_(0.0f)
    , base_speed_(0.0f)
    , current_speed_(0.0f)
    , initialized_(false) {
}

Motor::~Motor() {
    if (initialized_) {
        stop();
    }
}

bool Motor::init(int pin, float pwm_range, float frequency, float base_speed) {
    pin_ = pin;
    pwm_range_ = pwm_range;
    frequency_ = frequency;
    base_speed_ = base_speed;
    current_speed_ = 0.0f;
    
    // 设置 PWM 频率
    gpioSetPWMfrequency(pin_, static_cast<unsigned>(frequency_));
    gpioSetPWMrange(pin_, static_cast<unsigned>(pwm_range_));
    
    // 初始化为停止状态
    stop();
    
    initialized_ = true;
    std::cout << "[Motor] Initialized on pin " << pin_ << std::endl;
    
    return true;
}

void Motor::setSpeed(float speed) {
    if (!initialized_) {
        std::cerr << "[Motor] Not initialized" << std::endl;
        return;
    }
    
    current_speed_ = speed;
    gpioPWM(pin_, static_cast<unsigned>(speed));
}

void Motor::stop() {
    setSpeed(0.0f);
}

// ==================== Hardware 实现 ====================

Hardware::Hardware()
    : initialized_(false) {
}

Hardware::~Hardware() {
    cleanup();
}

Hardware& Hardware::getInstance() {
    static Hardware instance;
    return instance;
}

bool Hardware::init() {
    if (initialized_) {
        std::cout << "[Hardware] Already initialized" << std::endl;
        return true;
    }
    
    std::cout << "[Hardware] Initializing..." << std::endl;
    
    // 初始化 pigpio
    if (gpioInitialise() < 0) {
        std::cerr << "[Hardware] Failed to initialize pigpio" << std::endl;
        return false;
    }
    
    std::cout << "[Hardware] pigpio initialized" << std::endl;
    
    // 获取配置
    auto& config = Config::getInstance();
    if (!config.isLoaded()) {
        std::cerr << "[Hardware] Config not loaded" << std::endl;
        gpioTerminate();
        return false;
    }
    
    // 初始化舵机
    if (!servo_.init(
        config.hardware.servo.pin,
        config.hardware.servo.pwm_range,
        config.hardware.servo.frequency,
        config.hardware.servo.mid_value,
        config.hardware.servo.min_value,
        config.hardware.servo.max_value)) {
        std::cerr << "[Hardware] Failed to initialize servo" << std::endl;
        gpioTerminate();
        return false;
    }
    
    // 初始化电机
    if (!motor_.init(
        config.hardware.motor.pin,
        config.hardware.motor.pwm_range,
        config.hardware.motor.frequency,
        config.hardware.motor.base_speed)) {
        std::cerr << "[Hardware] Failed to initialize motor" << std::endl;
        gpioTerminate();
        return false;
    }
    
    initialized_ = true;
    std::cout << "[Hardware] Initialization complete" << std::endl;
    
    return true;
}

void Hardware::cleanup() {
    if (!initialized_) {
        return;
    }
    
    std::cout << "[Hardware] Cleaning up..." << std::endl;
    
    // 停止电机
    motor_.stop();
    
    // 舵机回中
    servo_.setMiddle();
    
    // 终止 pigpio
    gpioTerminate();
    
    initialized_ = false;
    std::cout << "[Hardware] Cleanup complete" << std::endl;
}

} // namespace smartcar
