/**
 * @file fastestdet_example.cpp
 * @brief FastestDet 使用示例
 */

#include "fastestdet.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

int main(int argc, char** argv)
{
    // 设置模型路径和参数
    std::string param_path = "../FastestDet.ncnn.param";
    std::string bin_path = "../FastestDet.ncnn.bin";
    std::string image_path = "../test.jpg";
    
    // 设置类别信息
    int num_classes = 2;
    std::vector<std::string> class_names = {"A", "B"};
    
    // 设置检测参数
    int input_size = 352;
    float conf_threshold = 0.65f;
    float nms_threshold = 0.45f;

    // 命令行参数解析
    if (argc >= 2)
    {
        image_path = argv[1];
    }
    if (argc >= 3)
    {
        param_path = argv[2];
        bin_path = param_path;
        size_t pos = bin_path.find(".param");
        if (pos != std::string::npos)
        {
            bin_path.replace(pos, 6, ".bin");
        }
    }

    std::cout << "========================================" << std::endl;
    std::cout << "FastestDet Example" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Model: " << param_path << std::endl;
    std::cout << "Image: " << image_path << std::endl;
    std::cout << "Input size: " << input_size << std::endl;
    std::cout << "Conf threshold: " << conf_threshold << std::endl;
    std::cout << "NMS threshold: " << nms_threshold << std::endl;
    std::cout << "========================================\n" << std::endl;

    // 初始化FastestDet
    FastestDet detector(param_path, 
                       bin_path,
                       num_classes,
                       class_names,
                       input_size,
                       conf_threshold,
                       nms_threshold,
                       4,      // 线程数
                       false); // 不使用GPU

    // 读取图像
    cv::Mat frame = cv::imread(image_path);
    if (frame.empty())
    {
        std::cerr << "Error: Failed to load image: " << image_path << std::endl;
        return -1;
    }

    std::cout << "Image loaded: " << frame.cols << "x" << frame.rows << std::endl;
    std::cout << "\nDetecting...\n" << std::endl;

    // 执行检测
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<DetectObject> objects = detector.detect(frame);
    auto end = std::chrono::high_resolution_clock::now();
    
    float inference_time = std::chrono::duration<float, std::milli>(end - start).count();

    // 输出检测结果
    std::cout << "========================================" << std::endl;
    std::cout << "Detection Results" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Detected objects: " << objects.size() << std::endl;
    std::cout << "Inference time: " << inference_time << " ms" << std::endl;
    std::cout << "FPS: " << (1000.0f / inference_time) << std::endl;
    std::cout << "========================================\n" << std::endl;

    // 打印每个检测到的对象信息
    for (size_t i = 0; i < objects.size(); i++)
    {
        const DetectObject& obj = objects[i];
        int center_x = static_cast<int>(obj.rect.x + obj.rect.width / 2);
        int center_y = static_cast<int>(obj.rect.y + obj.rect.height / 2);
        
        std::cout << "[" << i << "] "
                  << "Class: " << detector.get_class_name(obj.label) 
                  << " | Score: " << obj.prob
                  << " | Center: (" << center_x << ", " << center_y << ")"
                  << " | BBox: [" << static_cast<int>(obj.rect.x) << ", "
                  << static_cast<int>(obj.rect.y) << ", "
                  << static_cast<int>(obj.rect.x + obj.rect.width) << ", "
                  << static_cast<int>(obj.rect.y + obj.rect.height) << "]"
                  << std::endl;
    }

    // 在图像上绘制检测结果
    detector.draw_objects(frame, objects, true, true);

    // 保存结果图像
    std::string output_path = "output.jpg";
    cv::imwrite(output_path, frame);
    std::cout << "\nResult saved to: " << output_path << std::endl;

    // 显示结果
    // cv::imshow("Detection Result", frame);
    // cv::waitKey(0);

    return 0;
}
