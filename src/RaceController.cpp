/**
 * @file RaceController.cpp
 * @brief 主控制器模块
 */

#include "RaceController.hpp"
#include "Config.hpp"
#include <iostream>
#include <thread>
#include <chrono>

namespace smartcar {

RaceController::RaceController()
    : initialized_(false)
    , running_(false)
    , current_state_(RaceState::WAIT_START)
    , frame_count_(0)
    , zebra_count_(0)
    , obstacle_count_(0)
    , parking_count_(0)
    , lane_change_direction_(0)
    , camera_width_(320)
    , camera_height_(240)
    , hardware_(nullptr)
    , detector_(nullptr)
    , navigation_(nullptr) {
}

RaceController& RaceController::getInstance() {
    static RaceController instance;
    return instance;
}

bool RaceController::init() {
    if (initialized_) {
        std::cout << "[RaceController] Already initialized" << std::endl;
        return true;
    }
    
    std::cout << "[RaceController] Initializing..." << std::endl;
    
    // 获取配置
    auto& config = Config::getInstance();
    if (!config.isLoaded()) {
        std::cerr << "[RaceController] Config not loaded" << std::endl;
        return false;
    }
    
    // 初始化硬件
    hardware_ = &Hardware::getInstance();
    if (!hardware_->init()) {
        std::cerr << "[RaceController] Failed to initialize hardware" << std::endl;
        return false;
    }
    
    // 初始化摄像头
    if (!initCamera()) {
        std::cerr << "[RaceController] Failed to initialize camera" << std::endl;
        return false;
    }
    
    // 初始化图像处理
    if (!image_proc_.init(camera_width_, camera_height_)) {
        std::cerr << "[RaceController] Failed to initialize image process" << std::endl;
        return false;
    }
    
    // 初始化 AI 检测器
    detector_ = &MultiDetector::getInstance();
    if (!detector_->init()) {
        std::cerr << "[RaceController] Failed to initialize detector" << std::endl;
        return false;
    }
    
    // 初始化颜色检测器
    if (!color_detector_.init()) {
        std::cerr << "[RaceController] Failed to initialize color detector" << std::endl;
        return false;
    }
    
    // 初始化导航
    navigation_ = &Navigation::getInstance();
    if (!navigation_->init()) {
        std::cerr << "[RaceController] Failed to initialize navigation" << std::endl;
        return false;
    }
    
    initialized_ = true;
    std::cout << "[RaceController] Initialization complete" << std::endl;
    
    return true;
}

bool RaceController::initCamera() {
    auto& config = Config::getInstance();
    
    camera_width_ = config.camera.width;
    camera_height_ = config.camera.height;
    
    std::cout << "[RaceController] Opening camera " << config.camera.device_id << std::endl;
    
    camera_.open(config.camera.device_id);
    if (!camera_.isOpened()) {
        std::cerr << "[RaceController] Failed to open camera" << std::endl;
        return false;
    }
    
    // 设置摄像头参数
    camera_.set(cv::CAP_PROP_FRAME_WIDTH, camera_width_);
    camera_.set(cv::CAP_PROP_FRAME_HEIGHT, camera_height_);
    camera_.set(cv::CAP_PROP_FPS, config.camera.fps);
    
    std::cout << "[RaceController] Camera opened: " 
              << camera_width_ << "x" << camera_height_ 
              << " @" << config.camera.fps << "fps" << std::endl;
    
    return true;
}

int RaceController::run() {
    if (!initialized_) {
        std::cerr << "[RaceController] Not initialized" << std::endl;
        return -1;
    }
    
    running_ = true;
    frame_count_ = 0;
    
    std::cout << "[RaceController] Starting main loop..." << std::endl;
    
    cv::Mat frame;
    
    while (running_) {
        // 读取帧
        if (!camera_.read(frame)) {
            std::cerr << "[RaceController] Failed to read frame" << std::endl;
            break;
        }
        
        frame_count_++;
        
        // 处理帧
        processFrame(frame);
        
        // 显示
        auto& config = Config::getInstance();
        if (config.misc.enable_display) {
            drawStatus(frame);
            cv::imshow("Smart Car", frame);
            
            int key = cv::waitKey(1);
            if (key == 27 || key == 'q') {  // ESC 或 'q' 退出
                break;
            }
        }
        
        // 检查是否完成
        if (current_state_ == RaceState::FINISHED) {
            std::cout << "[RaceController] Race finished!" << std::endl;
            break;
        }
    }
    
    // 停止
    navigation_->stop();
    camera_.release();
    cv::destroyAllWindows();
    
    std::cout << "[RaceController] Stopped. Total frames: " << frame_count_ << std::endl;
    
    return 0;
}

void RaceController::processFrame(cv::Mat& frame) {
    // 调用对应的处理函数
    switch (current_state_) {
        case RaceState::WAIT_START:
            stateWaitStart(frame);
            break;
        case RaceState::LINE_FOLLOW:
            stateLineFollow(frame);
            break;
        case RaceState::ZEBRA_STOP:
            stateZebraStop(frame);
            break;
        case RaceState::LANE_CHANGE:
            stateLaneChange(frame);
            break;
        case RaceState::OBSTACLE_AVOID:
            stateObstacleAvoid(frame);
            break;
        case RaceState::PARKING_SEARCH:
            stateParkingSearch(frame);
            break;
        case RaceState::PARKING_ENTER:
            stateParkingEnter(frame);
            break;
        case RaceState::FINISHED:
            stateFinished(frame);
            break;
    }
}

void RaceController::stateWaitStart(cv::Mat& frame) {
    // 等待蓝色挡板检测开始
    std::vector<ColorDetection> blue_detections;
    int count = color_detector_.detectBlue(frame, blue_detections);
    
    if (count > 0) {
        std::cout << "[RaceController] Blue card detected! Starting race..." << std::endl;
        changeState(RaceState::LINE_FOLLOW);
    }
}

void RaceController::stateLineFollow(cv::Mat& frame) {
    auto& config = Config::getInstance();
    
    // 图像转灰度
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    
    // 边缘检测生成图
    cv::Mat binary;
    cv::threshold(gray, binary, 128, 255, cv::THRESH_BINARY);
    
    // 逐行精细追踪
    bool track_ok = image_proc_.tracking(binary, 153, 110);
    
    if (track_ok) {
        // 获取巡线中心点
        int line_center = image_proc_.getTrackingCenter(3);
        
        if (line_center > 0) {
            // 巡线控制
            navigation_->lineFollowing(
                line_center, 
                camera_width_,
                config.line_tracking.speed.straight
            );
            
            // 绘制巡线结果
            image_proc_.drawTrackingLines(frame, true, true, true);
        }
    } else {
        std::cerr << "[RaceController] Tracking failed, stopping" << std::endl;
        navigation_->stop();
    }
    
    // 检测斑马线
    cv::Rect zebra_roi(0, 100, camera_width_, 100);
    int stripe_count = image_proc_.detectZebraCrossing(
        frame,
        cv::Scalar(0, 0, 200),
        cv::Scalar(180, 30, 255),
        zebra_roi
    );
    
    if (stripe_count >= config.zebra_crossing.detect_threshold) {
        std::cout << "[RaceController] Zebra crossing detected!" << std::endl;
        changeState(RaceState::ZEBRA_STOP);
    }
    
    // 检测障碍物
    std::vector<DetectionResult> obstacles;
    detector_->getObstacleDetector().detect(frame, obstacles);
    
    if (!obstacles.empty() && obstacles[0].box.height > config.obstacle_avoidance.height_threshold) {
        std::cout << "[RaceController] Obstacle detected!" << std::endl;
        changeState(RaceState::OBSTACLE_AVOID);
    }
}

void RaceController::stateZebraStop(cv::Mat& frame) {
    auto& config = Config::getInstance();
    
    // 停车
    navigation_->stop();
    
    // 等待
    std::this_thread::sleep_for(std::chrono::milliseconds(config.zebra_crossing.stop_duration_ms));
    
    std::cout << "[RaceController] Zebra crossing passed" << std::endl;
    changeState(RaceState::LINE_FOLLOW);
}

void RaceController::stateLaneChange(cv::Mat& frame) {
    // 执行变道
    navigation_->laneChange(lane_change_direction_);
    
    std::cout << "[RaceController] Lane change complete" << std::endl;
    changeState(RaceState::LINE_FOLLOW);
}

void RaceController::stateObstacleAvoid(cv::Mat& frame) {
    auto& config = Config::getInstance();
    
    // 检测障碍物
    std::vector<DetectionResult> obstacles;
    detector_->getObstacleDetector().detect(frame, obstacles);
    
    if (!obstacles.empty()) {
        // 获取障碍物信息
        const auto& obs = obstacles[0];
        int obs_center_x = obs.box.x + obs.box.width / 2;
        int obs_top_y = obs.box.y;
        int obs_bottom_y = obs.box.y + obs.box.height;
        
        std::cout << "[RaceController] Obstacle detected - CenterX: " << obs_center_x 
                  << ", TopY: " << obs_top_y << ", BottomY: " << obs_bottom_y << std::endl;
        
        // 图像预处理
        cv::Mat gray, binary;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, binary, 128, 255, cv::THRESH_BINARY);
        
        // 根据障碍物位置绘制连接线
        cv::Mat binary_with_line = binary.clone();
        
        // 先进行一次常规追踪获取左右边界
        if (image_proc_.tracking(binary, 153, 110)) {
            const auto& left_line = image_proc_.getLeftLine();
            const auto& right_line = image_proc_.getRightLine();
            
            if (!left_line.empty() && !right_line.empty()) {
                // 计算左右边界前3个点的平均位置
                int left_avg = 0, right_avg = 0;
                for (int i = 0; i < std::min(3, (int)left_line.size()); i++) {
                    left_avg += left_line[i].x;
                    right_avg += right_line[i].x;
                }
                left_avg /= std::min(3, (int)left_line.size());
                right_avg /= std::min(3, (int)right_line.size());
                
                // 根据障碍物位置决定补线方向
                if (obs_center_x < camera_width_ / 2) {
                    // 障碍物在左边，补线到右边
                    std::cout << "[RaceController] Obstacle on left, draw line to right" << std::endl;
                    binary_with_line = image_proc_.drawWhiteLine(
                        binary_with_line,
                        cv::Point(obs_center_x, obs_bottom_y),
                        cv::Point(right_avg, 155),
                        8
                    );
                } else {
                    // 障碍物在右边，补线到左边
                    std::cout << "[RaceController] Obstacle on right, draw line to left" << std::endl;
                    binary_with_line = image_proc_.drawWhiteLine(
                        binary_with_line,
                        cv::Point(obs_center_x, obs_bottom_y),
                        cv::Point(left_avg, 155),
                        8
                    );
                }
            }
        }
        
        // 使用避障模式巡线
        if (image_proc_.trackingObstacle(binary_with_line, obs_top_y)) {
            int line_center = image_proc_.getTrackingCenter(3);
            
            if (line_center > 0) {
                // 使用避障速度进行巡线
                navigation_->lineFollowing(
                    line_center,
                    camera_width_,
                    config.obstacle_avoidance.avoid_speed
                );
                
                // 绘制避障巡线结果
                image_proc_.drawTrackingLines(frame, true, true, true);
            }
        } else {
            // 直接使用避障控制
            navigation_->obstacleAvoidance(obs.box, camera_width_);
        }
        
        obstacle_count_++;
        
        // 绘制检测结果
        detector_->getObstacleDetector().drawResults(frame, obstacles);
        
        // 如果避过了足够多的障碍物，继续巡线
        if (obstacle_count_ >= config.obstacle_avoidance.max_obstacle_count) {
            std::cout << "[RaceController] All obstacles avoided" << std::endl;
            obstacle_count_ = 0;
            changeState(RaceState::LINE_FOLLOW);
        }
    } else {
        // 未检测到障碍物，继续巡线
        std::cout << "[RaceController] No obstacle detected, return to line following mode" << std::endl;
        changeState(RaceState::LINE_FOLLOW);
    }
}

void RaceController::stateParkingSearch(cv::Mat& frame) {
    auto& config = Config::getInstance();
    
    // 检测停车位
    std::vector<ColorDetection> parking_spots;
    color_detector_.detectRed(frame, parking_spots, config.parking.detect_threshold);
    
    if (!parking_spots.empty()) {
        parking_count_++;
        
        // 绘制检测结果
        color_detector_.drawResults(frame, parking_spots, cv::Scalar(0, 0, 255));
        
        if (parking_count_ >= config.parking.detect_threshold) {
            std::cout << "[RaceController] Parking spot found!" << std::endl;
            changeState(RaceState::PARKING_ENTER);
        }
    } else {
        parking_count_ = 0;
        
        // 继续巡线寻找
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        int line_center = image_proc_.findLineCenter(gray, camera_height_ / 2, camera_width_ / 2);
        
        if (line_center > 0) {
            navigation_->lineFollowing(line_center, camera_width_, config.parking.approach_speed);
        }
    }
}

void RaceController::stateParkingEnter(cv::Mat& frame) {
    auto& config = Config::getInstance();
    
    // 检测停车位
    std::vector<ColorDetection> parking_spots;
    color_detector_.detectRed(frame, parking_spots, config.parking.detect_threshold);
    
    if (!parking_spots.empty()) {
        // 对准停车位中心
        navigation_->parkingAlign(parking_spots[0].center.x, camera_width_);
        
        // 如果已经对准，进入停车位
        int error = std::abs(parking_spots[0].center.x - camera_width_ / 2);
        if (error < 20) {
            std::cout << "[RaceController] Entering parking spot..." << std::endl;
            navigation_->goStraight(config.parking.approach_speed / 2, 2000);
            navigation_->stop();
            
            changeState(RaceState::FINISHED);
        }
    } else {
        // 失去目标，返回搜索
        changeState(RaceState::PARKING_SEARCH);
    }
}

void RaceController::stateFinished(cv::Mat& frame) {
    // 停止
    navigation_->stop();
    running_ = false;
}

void RaceController::stop() {
    running_ = false;
}

void RaceController::changeState(RaceState new_state) {
    std::cout << "[RaceController] State: " << getStateName() 
              << " -> " ;
    
    current_state_ = new_state;
    
    std::cout << getStateName() << std::endl;
}

std::string RaceController::getStateName() const {
    switch (current_state_) {
        case RaceState::WAIT_START:     return "WAIT_START";
        case RaceState::LINE_FOLLOW:    return "LINE_FOLLOW";
        case RaceState::ZEBRA_STOP:     return "ZEBRA_STOP";
        case RaceState::LANE_CHANGE:    return "LANE_CHANGE";
        case RaceState::OBSTACLE_AVOID: return "OBSTACLE_AVOID";
        case RaceState::PARKING_SEARCH: return "PARKING_SEARCH";
        case RaceState::PARKING_ENTER:  return "PARKING_ENTER";
        case RaceState::FINISHED:       return "FINISHED";
        default:                        return "UNKNOWN";
    }
}

void RaceController::drawStatus(cv::Mat& frame) {
    // 绘制状态信息
    std::string status = "State: " + getStateName();
    cv::putText(frame, status, cv::Point(10, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    
    // 绘制帧数
    std::string frame_info = "Frame: " + std::to_string(frame_count_);
    cv::putText(frame, frame_info, cv::Point(10, 60),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    
    // 绘制舵机和速度信息
    std::string servo_info = "Servo: " + std::to_string(static_cast<int>(navigation_->getCurrentServo()));
    cv::putText(frame, servo_info, cv::Point(10, 90),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    
    std::string speed_info = "Speed: " + std::to_string(static_cast<int>(navigation_->getCurrentSpeed()));
    cv::putText(frame, speed_info, cv::Point(10, 120),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
}

} // namespace smartcar
