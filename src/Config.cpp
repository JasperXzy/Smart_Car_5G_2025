/**
 * @file Config.cpp
 * @brief 配置管理模块
 */

#include "Config.hpp"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>

namespace smartcar {

// ==================== 单例实现 ====================

Config& Config::getInstance() {
    static Config instance;
    return instance;
}

// ==================== 加载配置 ====================

bool Config::load(const std::string& yaml_file) {
    try {
        std::cout << "[Config] Loading from: " << yaml_file << std::endl;
        
        YAML::Node config = YAML::LoadFile(yaml_file);
        
        // 解析 Hardware 配置
        if (config["hardware"]) {
            auto hw = config["hardware"];
            
            if (hw["servo"]) {
                auto servo = hw["servo"];
                hardware.servo.pin = servo["pin"].as<int>(hardware.servo.pin);
                hardware.servo.pwm_range = servo["pwm_range"].as<float>(hardware.servo.pwm_range);
                hardware.servo.frequency = servo["frequency"].as<float>(hardware.servo.frequency);
                hardware.servo.mid_value = servo["mid_value"].as<float>(hardware.servo.mid_value);
                hardware.servo.min_value = servo["min_value"].as<float>(hardware.servo.min_value);
                hardware.servo.max_value = servo["max_value"].as<float>(hardware.servo.max_value);
            }
            
            if (hw["motor"]) {
                auto motor = hw["motor"];
                hardware.motor.pin = motor["pin"].as<int>(hardware.motor.pin);
                hardware.motor.pwm_range = motor["pwm_range"].as<float>(hardware.motor.pwm_range);
                hardware.motor.frequency = motor["frequency"].as<float>(hardware.motor.frequency);
                hardware.motor.base_speed = motor["base_speed"].as<float>(hardware.motor.base_speed);
            }
        }
        
        // 解析 Camera 配置
        if (config["camera"]) {
            auto cam = config["camera"];
            camera.device_id = cam["device_id"].as<int>(camera.device_id);
            camera.width = cam["width"].as<int>(camera.width);
            camera.height = cam["height"].as<int>(camera.height);
            camera.fps = cam["fps"].as<int>(camera.fps);
            camera.fourcc = cam["fourcc"].as<std::string>(camera.fourcc);
            
            if (cam["calibration"]) {
                auto calib = cam["calibration"];
                camera.calibration.k1 = calib["k1"].as<double>(camera.calibration.k1);
                camera.calibration.k2 = calib["k2"].as<double>(camera.calibration.k2);
                camera.calibration.k3 = calib["k3"].as<double>(camera.calibration.k3);
                camera.calibration.p1 = calib["p1"].as<double>(camera.calibration.p1);
                camera.calibration.p2 = calib["p2"].as<double>(camera.calibration.p2);
                camera.calibration.fx = calib["fx"].as<double>(camera.calibration.fx);
                camera.calibration.fy = calib["fy"].as<double>(camera.calibration.fy);
                camera.calibration.cx = calib["cx"].as<double>(camera.calibration.cx);
                camera.calibration.cy = calib["cy"].as<double>(camera.calibration.cy);
            }
        }
        
        // 解析 Models 配置
        if (config["models"]) {
            auto mdl = config["models"];
            
            if (mdl["obstacle"]) {
                auto obs = mdl["obstacle"];
                models.obstacle.param_path = obs["param_path"].as<std::string>(models.obstacle.param_path);
                models.obstacle.bin_path = obs["bin_path"].as<std::string>(models.obstacle.bin_path);
                models.obstacle.input_size = obs["input_size"].as<int>(models.obstacle.input_size);
                models.obstacle.num_threads = obs["num_threads"].as<int>(models.obstacle.num_threads);
                models.obstacle.conf_threshold = obs["conf_threshold"].as<float>(models.obstacle.conf_threshold);
                models.obstacle.nms_threshold = obs["nms_threshold"].as<float>(models.obstacle.nms_threshold);
            }
            
            if (mdl["direction"]) {
                auto dir = mdl["direction"];
                models.direction.param_path = dir["param_path"].as<std::string>(models.direction.param_path);
                models.direction.bin_path = dir["bin_path"].as<std::string>(models.direction.bin_path);
                models.direction.input_size = dir["input_size"].as<int>(models.direction.input_size);
                models.direction.num_threads = dir["num_threads"].as<int>(models.direction.num_threads);
                models.direction.conf_threshold = dir["conf_threshold"].as<float>(models.direction.conf_threshold);
                models.direction.nms_threshold = dir["nms_threshold"].as<float>(models.direction.nms_threshold);
            }
            
            if (mdl["parking"]) {
                auto park = mdl["parking"];
                models.parking.param_path = park["param_path"].as<std::string>(models.parking.param_path);
                models.parking.bin_path = park["bin_path"].as<std::string>(models.parking.bin_path);
                models.parking.input_size = park["input_size"].as<int>(models.parking.input_size);
                models.parking.num_threads = park["num_threads"].as<int>(models.parking.num_threads);
                models.parking.conf_threshold = park["conf_threshold"].as<float>(models.parking.conf_threshold);
                models.parking.nms_threshold = park["nms_threshold"].as<float>(models.parking.nms_threshold);
            }
        }
        
        // 解析 Line Tracking 配置
        if (config["line_tracking"]) {
            auto lt = config["line_tracking"];
            
            if (lt["pid_normal"]) {
                auto pid = lt["pid_normal"];
                line_tracking.pid_normal.kp = pid["kp"].as<float>(line_tracking.pid_normal.kp);
                line_tracking.pid_normal.kd = pid["kd"].as<float>(line_tracking.pid_normal.kd);
                line_tracking.pid_normal.ki = pid["ki"].as<float>(line_tracking.pid_normal.ki);
            }
            
            if (lt["speed"]) {
                auto speed = lt["speed"];
                line_tracking.speed.straight = speed["straight"].as<int>(line_tracking.speed.straight);
                line_tracking.speed.curve = speed["curve"].as<int>(line_tracking.speed.curve);
                line_tracking.speed.start = speed["start"].as<int>(line_tracking.speed.start);
            }
            
            line_tracking.curve_frame_threshold = lt["curve_frame_threshold"].as<int>(line_tracking.curve_frame_threshold);
            line_tracking.search_start_x = lt["search_start_x"].as<int>(line_tracking.search_start_x);
        }
        
        // 解析 Obstacle Avoidance 配置
        if (config["obstacle_avoidance"]) {
            auto oa = config["obstacle_avoidance"];
            obstacle_avoidance.height_threshold = oa["height_threshold"].as<int>(obstacle_avoidance.height_threshold);
            obstacle_avoidance.max_obstacle_count = oa["max_obstacle_count"].as<int>(obstacle_avoidance.max_obstacle_count);
            obstacle_avoidance.turn_angle_base = oa["turn_angle_base"].as<float>(obstacle_avoidance.turn_angle_base);
            obstacle_avoidance.avoid_speed = oa["avoid_speed"].as<int>(obstacle_avoidance.avoid_speed);
        }
        
        // 解析 Zebra Crossing 配置
        if (config["zebra_crossing"]) {
            auto zc = config["zebra_crossing"];
            zebra_crossing.detect_threshold = zc["detect_threshold"].as<int>(zebra_crossing.detect_threshold);
            zebra_crossing.stop_duration_ms = zc["stop_duration_ms"].as<int>(zebra_crossing.stop_duration_ms);
        }
        
        // 解析 Lane Change 配置
        if (config["lane_change"]) {
            auto lc = config["lane_change"];
            lane_change.turn_angle_left = lc["turn_angle_left"].as<float>(lane_change.turn_angle_left);
            lane_change.turn_angle_right = lc["turn_angle_right"].as<float>(lane_change.turn_angle_right);
            lane_change.turn_duration_ms = lc["turn_duration_ms"].as<int>(lane_change.turn_duration_ms);
            lane_change.change_speed = lc["change_speed"].as<int>(lane_change.change_speed);
        }
        
        // 解析 Parking 配置
        if (config["parking"]) {
            auto park = config["parking"];
            parking.detect_threshold = park["detect_threshold"].as<int>(parking.detect_threshold);
            parking.approach_speed = park["approach_speed"].as<int>(parking.approach_speed);
            parking.entry_angle = park["entry_angle"].as<float>(parking.entry_angle);
        }
        
        // 解析 Blue Card 配置
        if (config["blue_card"]) {
            auto bc = config["blue_card"];
            
            if (bc["hsv_range"]) {
                auto hsv = bc["hsv_range"];
                blue_card.hsv_range.h_min = hsv["h_min"].as<int>(blue_card.hsv_range.h_min);
                blue_card.hsv_range.h_max = hsv["h_max"].as<int>(blue_card.hsv_range.h_max);
                blue_card.hsv_range.s_min = hsv["s_min"].as<int>(blue_card.hsv_range.s_min);
                blue_card.hsv_range.s_max = hsv["s_max"].as<int>(blue_card.hsv_range.s_max);
                blue_card.hsv_range.v_min = hsv["v_min"].as<int>(blue_card.hsv_range.v_min);
                blue_card.hsv_range.v_max = hsv["v_max"].as<int>(blue_card.hsv_range.v_max);
            }
            
            blue_card.min_area = bc["min_area"].as<int>(blue_card.min_area);
        }
        
        // 解析 Misc 配置
        if (config["misc"]) {
            auto misc_node = config["misc"];
            misc.debug_mode = misc_node["debug_mode"].as<bool>(misc.debug_mode);
            misc.enable_display = misc_node["enable_display"].as<bool>(misc.enable_display);
            misc.log_level = misc_node["log_level"].as<std::string>(misc.log_level);
        }
        
        config_file_ = yaml_file;
        loaded_ = true;
        
        std::cout << "[Config] Loaded successfully" << std::endl;
        return true;
        
    } catch (const YAML::Exception& e) {
        std::cerr << "[Config] YAML parsing error: " << e.what() << std::endl;
        std::cerr << "[Config] Using default values" << std::endl;
        loaded_ = true;  // 使用默认值
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[Config] Error: " << e.what() << std::endl;
        return false;
    }
}

// ==================== 验证配置 ====================

bool Config::validate() const {
    if (!loaded_) {
        std::cerr << "[Config] Not loaded yet" << std::endl;
        return false;
    }
    
    bool valid = true;
    
    // 验证硬件参数
    if (hardware.servo.pin < 0 || hardware.servo.pin > 40) {
        std::cerr << "[Config] Invalid servo pin: " << hardware.servo.pin << std::endl;
        valid = false;
    }
    
    if (hardware.motor.pin < 0 || hardware.motor.pin > 40) {
        std::cerr << "[Config] Invalid motor pin: " << hardware.motor.pin << std::endl;
        valid = false;
    }
    
    // 验证摄像头参数
    if (camera.width <= 0 || camera.height <= 0) {
        std::cerr << "[Config] Invalid camera resolution: " 
                  << camera.width << "x" << camera.height << std::endl;
        valid = false;
    }
    
    // 验证 PID 参数
    if (line_tracking.pid_normal.kp < 0 || line_tracking.pid_normal.kd < 0) {
        std::cerr << "[Config] Invalid PID parameters" << std::endl;
        valid = false;
    }
    
    // 验证模型路径
    if (models.obstacle.param_path.empty()) {
        std::cerr << "[Config] Warning: Obstacle model path not configured" << std::endl;
    }
    
    if (models.direction.param_path.empty()) {
        std::cerr << "[Config] Warning: Direction model path not configured" << std::endl;
    }
    
    if (models.parking.param_path.empty()) {
        std::cerr << "[Config] Warning: Parking model path not configured" << std::endl;
    }
    
    return valid;
}

// ==================== 打印配置 ====================

void Config::print() const {
    std::cout << "\n========== Smart Car Configuration ==========\n";
    
    std::cout << "\n[Hardware]" << std::endl;
    std::cout << "  Servo: Pin=" << hardware.servo.pin 
              << ", Mid=" << hardware.servo.mid_value << std::endl;
    std::cout << "  Motor: Pin=" << hardware.motor.pin 
              << ", Base=" << hardware.motor.base_speed << std::endl;
    
    std::cout << "\n[Camera]" << std::endl;
    std::cout << "  Resolution: " << camera.width << "x" << camera.height << std::endl;
    std::cout << "  FPS: " << camera.fps << ", FourCC: " << camera.fourcc << std::endl;
    
    std::cout << "\n[Models]" << std::endl;
    std::cout << "  Obstacle: " << models.obstacle.param_path << std::endl;
    std::cout << "  Direction: " << models.direction.param_path << std::endl;
    std::cout << "  Parking: " << models.parking.param_path << std::endl;
    
    std::cout << "\n[Line Tracking]" << std::endl;
    std::cout << "  PID (Kp, Kd, Ki): " << line_tracking.pid_normal.kp 
              << ", " << line_tracking.pid_normal.kd 
              << ", " << line_tracking.pid_normal.ki << std::endl;
    std::cout << "  Speed (Straight, Curve, Start): " 
              << line_tracking.speed.straight << ", " 
              << line_tracking.speed.curve << ", " 
              << line_tracking.speed.start << std::endl;
    
    std::cout << "\n[Obstacle Avoidance]" << std::endl;
    std::cout << "  Height Threshold: " << obstacle_avoidance.height_threshold << std::endl;
    std::cout << "  Max Count: " << obstacle_avoidance.max_obstacle_count << std::endl;
    std::cout << "  Avoid Speed: " << obstacle_avoidance.avoid_speed << std::endl;
    
    std::cout << "\n[Lane Change]" << std::endl;
    std::cout << "  Turn Angle (Left, Right): " 
              << lane_change.turn_angle_left << ", " 
              << lane_change.turn_angle_right << std::endl;
    std::cout << "  Duration: " << lane_change.turn_duration_ms << " ms" << std::endl;
    
    std::cout << "\n[Parking]" << std::endl;
    std::cout << "  Approach Speed: " << parking.approach_speed << std::endl;
    std::cout << "  Entry Angle: " << parking.entry_angle << std::endl;
    
    std::cout << "\n[Blue Card]" << std::endl;
    std::cout << "  HSV Range: H[" << blue_card.hsv_range.h_min << "-" << blue_card.hsv_range.h_max << "]"
              << " S[" << blue_card.hsv_range.s_min << "-" << blue_card.hsv_range.s_max << "]"
              << " V[" << blue_card.hsv_range.v_min << "-" << blue_card.hsv_range.v_max << "]" << std::endl;
    std::cout << "  Min Area: " << blue_card.min_area << std::endl;
    
    std::cout << "\n[Misc]" << std::endl;
    std::cout << "  Debug Mode: " << (misc.debug_mode ? "ON" : "OFF") << std::endl;
    std::cout << "  Display: " << (misc.enable_display ? "ON" : "OFF") << std::endl;
    std::cout << "  Log Level: " << misc.log_level << std::endl;
    
    std::cout << "\n=============================================\n" << std::endl;
}

// ==================== 保存配置 ====================

bool Config::save(const std::string& yaml_file) const {
    try {
        YAML::Emitter out;
        
        out << YAML::BeginMap;
        
        // Hardware
        out << YAML::Key << "hardware" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "servo" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "pin" << YAML::Value << hardware.servo.pin;
        out << YAML::Key << "pwm_range" << YAML::Value << hardware.servo.pwm_range;
        out << YAML::Key << "frequency" << YAML::Value << hardware.servo.frequency;
        out << YAML::Key << "mid_value" << YAML::Value << hardware.servo.mid_value;
        out << YAML::Key << "min_value" << YAML::Value << hardware.servo.min_value;
        out << YAML::Key << "max_value" << YAML::Value << hardware.servo.max_value;
        out << YAML::EndMap;
        
        out << YAML::Key << "motor" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "pin" << YAML::Value << hardware.motor.pin;
        out << YAML::Key << "pwm_range" << YAML::Value << hardware.motor.pwm_range;
        out << YAML::Key << "frequency" << YAML::Value << hardware.motor.frequency;
        out << YAML::Key << "base_speed" << YAML::Value << hardware.motor.base_speed;
        out << YAML::EndMap;
        out << YAML::EndMap;
        
        // Camera
        out << YAML::Key << "camera" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "device_id" << YAML::Value << camera.device_id;
        out << YAML::Key << "width" << YAML::Value << camera.width;
        out << YAML::Key << "height" << YAML::Value << camera.height;
        out << YAML::Key << "fps" << YAML::Value << camera.fps;
        out << YAML::Key << "fourcc" << YAML::Value << camera.fourcc;
        
        out << YAML::Key << "calibration" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "k1" << YAML::Value << camera.calibration.k1;
        out << YAML::Key << "k2" << YAML::Value << camera.calibration.k2;
        out << YAML::Key << "k3" << YAML::Value << camera.calibration.k3;
        out << YAML::Key << "p1" << YAML::Value << camera.calibration.p1;
        out << YAML::Key << "p2" << YAML::Value << camera.calibration.p2;
        out << YAML::Key << "fx" << YAML::Value << camera.calibration.fx;
        out << YAML::Key << "fy" << YAML::Value << camera.calibration.fy;
        out << YAML::Key << "cx" << YAML::Value << camera.calibration.cx;
        out << YAML::Key << "cy" << YAML::Value << camera.calibration.cy;
        out << YAML::EndMap;
        out << YAML::EndMap;
        
        // Models
        out << YAML::Key << "models" << YAML::Value << YAML::BeginMap;
        
        out << YAML::Key << "obstacle" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "param_path" << YAML::Value << models.obstacle.param_path;
        out << YAML::Key << "bin_path" << YAML::Value << models.obstacle.bin_path;
        out << YAML::Key << "input_size" << YAML::Value << models.obstacle.input_size;
        out << YAML::Key << "num_threads" << YAML::Value << models.obstacle.num_threads;
        out << YAML::Key << "conf_threshold" << YAML::Value << models.obstacle.conf_threshold;
        out << YAML::Key << "nms_threshold" << YAML::Value << models.obstacle.nms_threshold;
        out << YAML::EndMap;
        
        out << YAML::Key << "direction" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "param_path" << YAML::Value << models.direction.param_path;
        out << YAML::Key << "bin_path" << YAML::Value << models.direction.bin_path;
        out << YAML::Key << "input_size" << YAML::Value << models.direction.input_size;
        out << YAML::Key << "num_threads" << YAML::Value << models.direction.num_threads;
        out << YAML::Key << "conf_threshold" << YAML::Value << models.direction.conf_threshold;
        out << YAML::Key << "nms_threshold" << YAML::Value << models.direction.nms_threshold;
        out << YAML::EndMap;
        
        out << YAML::Key << "parking" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "param_path" << YAML::Value << models.parking.param_path;
        out << YAML::Key << "bin_path" << YAML::Value << models.parking.bin_path;
        out << YAML::Key << "input_size" << YAML::Value << models.parking.input_size;
        out << YAML::Key << "num_threads" << YAML::Value << models.parking.num_threads;
        out << YAML::Key << "conf_threshold" << YAML::Value << models.parking.conf_threshold;
        out << YAML::Key << "nms_threshold" << YAML::Value << models.parking.nms_threshold;
        out << YAML::EndMap;
        
        out << YAML::EndMap;
        
        // Line Tracking
        out << YAML::Key << "line_tracking" << YAML::Value << YAML::BeginMap;
        
        out << YAML::Key << "pid_normal" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "kp" << YAML::Value << line_tracking.pid_normal.kp;
        out << YAML::Key << "kd" << YAML::Value << line_tracking.pid_normal.kd;
        out << YAML::Key << "ki" << YAML::Value << line_tracking.pid_normal.ki;
        out << YAML::EndMap;
        
        out << YAML::Key << "speed" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "straight" << YAML::Value << line_tracking.speed.straight;
        out << YAML::Key << "curve" << YAML::Value << line_tracking.speed.curve;
        out << YAML::Key << "start" << YAML::Value << line_tracking.speed.start;
        out << YAML::EndMap;
        
        out << YAML::Key << "curve_frame_threshold" << YAML::Value << line_tracking.curve_frame_threshold;
        out << YAML::Key << "search_start_x" << YAML::Value << line_tracking.search_start_x;
        out << YAML::EndMap;
        
        // Obstacle Avoidance
        out << YAML::Key << "obstacle_avoidance" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "height_threshold" << YAML::Value << obstacle_avoidance.height_threshold;
        out << YAML::Key << "max_obstacle_count" << YAML::Value << obstacle_avoidance.max_obstacle_count;
        out << YAML::Key << "turn_angle_base" << YAML::Value << obstacle_avoidance.turn_angle_base;
        out << YAML::Key << "avoid_speed" << YAML::Value << obstacle_avoidance.avoid_speed;
        out << YAML::EndMap;
        
        // Zebra Crossing
        out << YAML::Key << "zebra_crossing" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "detect_threshold" << YAML::Value << zebra_crossing.detect_threshold;
        out << YAML::Key << "stop_duration_ms" << YAML::Value << zebra_crossing.stop_duration_ms;
        out << YAML::EndMap;
        
        // Lane Change
        out << YAML::Key << "lane_change" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "turn_angle_left" << YAML::Value << lane_change.turn_angle_left;
        out << YAML::Key << "turn_angle_right" << YAML::Value << lane_change.turn_angle_right;
        out << YAML::Key << "turn_duration_ms" << YAML::Value << lane_change.turn_duration_ms;
        out << YAML::Key << "change_speed" << YAML::Value << lane_change.change_speed;
        out << YAML::EndMap;
        
        // Parking
        out << YAML::Key << "parking" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "detect_threshold" << YAML::Value << parking.detect_threshold;
        out << YAML::Key << "approach_speed" << YAML::Value << parking.approach_speed;
        out << YAML::Key << "entry_angle" << YAML::Value << parking.entry_angle;
        out << YAML::EndMap;
        
        // Blue Card
        out << YAML::Key << "blue_card" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "hsv_range" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "h_min" << YAML::Value << blue_card.hsv_range.h_min;
        out << YAML::Key << "h_max" << YAML::Value << blue_card.hsv_range.h_max;
        out << YAML::Key << "s_min" << YAML::Value << blue_card.hsv_range.s_min;
        out << YAML::Key << "s_max" << YAML::Value << blue_card.hsv_range.s_max;
        out << YAML::Key << "v_min" << YAML::Value << blue_card.hsv_range.v_min;
        out << YAML::Key << "v_max" << YAML::Value << blue_card.hsv_range.v_max;
        out << YAML::EndMap;
        out << YAML::Key << "min_area" << YAML::Value << blue_card.min_area;
        out << YAML::EndMap;
        
        // Misc
        out << YAML::Key << "misc" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "debug_mode" << YAML::Value << misc.debug_mode;
        out << YAML::Key << "enable_display" << YAML::Value << misc.enable_display;
        out << YAML::Key << "log_level" << YAML::Value << misc.log_level;
        out << YAML::EndMap;
        
        out << YAML::EndMap;
        
        std::ofstream fout(yaml_file);
        if (!fout.is_open()) {
            std::cerr << "[Config] Failed to open file for writing: " << yaml_file << std::endl;
            return false;
        }
        
        fout << out.c_str();
        fout.close();
        
        std::cout << "[Config] Saved to: " << yaml_file << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[Config] Save error: " << e.what() << std::endl;
        return false;
    }
}

} // namespace smartcar
