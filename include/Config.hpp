/**
 * @file Config.hpp
 * @brief 配置管理模块
 */

#pragma once

#include <string>
#include <vector>

namespace smartcar {

/**
 * @brief 配置管理类
 */
class Config {
public:
    // 获取单例实例
    static Config& getInstance();
    
    // 禁止拷贝和赋值
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;
    
    // 加载配置文件
    bool load(const std::string& yaml_file);
    
    // 验证配置
    bool validate() const;
    
    // 打印配置
    void print() const;
    
    // 保存配置
    bool save(const std::string& yaml_file) const;
    
    // 检查是否已加载
    bool isLoaded() const { return loaded_; }
    
    // ==================== 配置结构体 ====================
    
    // 硬件配置
    struct HardwareConfig {
        struct ServoConfig {
            int pin = 12;
            float pwm_range = 10000.0f;
            float frequency = 50.0f;
            float mid_value = 690.0f;
            float min_value = 580.0f;
            float max_value = 1000.0f;
        } servo;
        
        struct MotorConfig {
            int pin = 13;
            float pwm_range = 40000.0f;
            float frequency = 200.0f;
            float base_speed = 11400.0f;
        } motor;
    } hardware;
    
    // 摄像头配置
    struct CameraConfig {
        int device_id = 0;
        int width = 320;
        int height = 240;
        int fps = 30;
        std::string fourcc = "MJPG";
        
        struct CalibrationConfig {
            double k1 = 0.0;
            double k2 = 0.0;
            double k3 = 0.0;
            double p1 = 0.0;
            double p2 = 0.0;
            double fx = 176.842;
            double fy = 176.991;
            double cx = 159.706;
            double cy = 120.558;
        } calibration;
    } camera;
    
    // AI 模型配置
    struct ModelConfig {
        std::string param_path;
        std::string bin_path;
        int input_size = 352;
        int num_threads = 4;
        float conf_threshold = 0.5f;
        float nms_threshold = 0.45f;
    };
    
    struct ModelsConfig {
        ModelConfig obstacle;
        ModelConfig direction;
        ModelConfig parking;
    } models;
    
    // 巡线配置
    struct LineTrackingConfig {
        struct PIDConfig {
            float kp = 1.0f;
            float kd = 2.0f;
            float ki = 0.0f;
        } pid_normal;
        
        struct SpeedConfig {
            int straight = 5000;
            int curve = 1400;
            int start = 1500;
        } speed;
        
        int curve_frame_threshold = 550;
        int search_start_x = 160;
    } line_tracking;
    
    // 避障配置
    struct ObstacleAvoidanceConfig {
        int height_threshold = 170;
        int max_obstacle_count = 3;
        float turn_angle_base = 100.0f;
        int avoid_speed = 1300;
    } obstacle_avoidance;
    
    // 斑马线配置
    struct ZebraCrossingConfig {
        int detect_threshold = 4;
        int stop_duration_ms = 3000;
    } zebra_crossing;
    
    // 变道配置
    struct LaneChangeConfig {
        float turn_angle_left = 825.0f;
        float turn_angle_right = 620.0f;
        int turn_duration_ms = 1200;
        int change_speed = 1300;
    } lane_change;
    
    // 停车配置
    struct ParkingConfig {
        int detect_threshold = 5;
        int approach_speed = 1700;
        float entry_angle = 90.0f;
    } parking;
    
    // 蓝色挡板配置
    struct BlueCardConfig {
        struct HSVRange {
            int h_min = 100;
            int h_max = 124;
            int s_min = 43;
            int s_max = 255;
            int v_min = 46;
            int v_max = 255;
        } hsv_range;
        
        int min_area = 500;
    } blue_card;
    
    // 其他配置
    struct MiscConfig {
        bool debug_mode = false;
        bool enable_display = false;
        std::string log_level = "INFO";
    } misc;
    
private:
    Config() = default;
    
    bool loaded_ = false;
    std::string config_file_;
};

} // namespace smartcar
