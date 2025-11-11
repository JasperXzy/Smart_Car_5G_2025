/**
 * @file main.cpp
 * @brief 主程序
 */

#include "Config.hpp"
#include "Hardware.hpp"
#include "ImageProcess.hpp"
#include "AIDetector.hpp"
#include "ColorDetector.hpp"
#include "Navigation.hpp"
#include "RaceController.hpp"

#include <iostream>
#include <csignal>
#include <cstdlib>

using namespace smartcar;

// 全局控制器引用
RaceController* g_controller = nullptr;

/**
 * @brief 信号处理函数
 */
void signalHandler(int signum) {
    std::cout << "\n[Main] Interrupt signal (" << signum << ") received." << std::endl;
    if (g_controller) {
        g_controller->stop();
    }
    exit(signum);
}

/**
 * @brief 主函数
 */
int main(int argc, char* argv[]) {
    // 注册信号处理
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    // 1. 加载配置文件
    std::string config_file = "config/config.yaml";
    if (argc > 1) {
        config_file = argv[1];
    }
    
    std::cout << "\n[Main] Loading config: " << config_file << std::endl;
    Config& config = Config::getInstance();
    if (!config.load(config_file)) {
        std::cerr << "[Main] ERROR: Failed to load config file!" << std::endl;
        std::cerr << "[Main] Please ensure " << config_file << " exists" << std::endl;
        return -1;
    }
    
    if (!config.validate()) {
        std::cerr << "[Main] ERROR: Config validation failed!" << std::endl;
        return -1;
    }
    
    std::cout << "[Main] Config loaded successfully" << std::endl;
    
    // 2. 初始化控制器
    std::cout << "\n[Main] Initializing race controller..." << std::endl;
    RaceController& controller = RaceController::getInstance();
    g_controller = &controller;
    
    if (!controller.init()) {
        std::cerr << "[Main] ERROR: Failed to initialize race controller!" << std::endl;
        return -1;
    }
    
    std::cout << "[Main] Race controller initialized successfully" << std::endl;
    
    // 3. 运行主循环
    std::cout << "\n======================================" << std::endl;
    std::cout << "[Main] Starting race..." << std::endl;
    std::cout << "[Main] Press 'q' or ESC to quit" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    int ret = controller.run();
    
    // 4. 清理
    std::cout << "\n======================================" << std::endl;
    std::cout << "[Main] Shutting down..." << std::endl;
    g_controller = nullptr;
    
    std::cout << "[Main] Program exited with code: " << ret << std::endl;
    std::cout << "========================================" << std::endl;
    
    return ret;
}
