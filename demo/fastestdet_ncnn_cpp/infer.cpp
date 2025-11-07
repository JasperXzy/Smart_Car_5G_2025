#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <float.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <chrono>

using namespace std;

// 检测对象结构体
struct Object
{
    cv::Rect_<float> rect;      // 边界框
    int label;                  // 类别ID
    float prob;                 // 置信度
};

// 计算两个边界框的交集面积
static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

// 快速排序
static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            std::swap(objects[i], objects[j]);
            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

// NMS
static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            // 类别内NMS
            if (a.label != b.label)
                continue;

            // 计算IoU
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

static inline float fast_tanh(float x)
{
    return tanh(x);
}

// 生成检测候选框
// FastestDet输出格式: [1, C, H, W]
// C = 1 + 4 + num_class
// 通道0: 前背景分类分数
// 通道1-4: 边界框回归
// 通道5+: 类别分类分数
static void generate_proposals(const ncnn::Mat& out, int img_w, int img_h, 
                               float prob_threshold, std::vector<Object>& objects)
{
    const int channels = out.c;
    const int height = out.h;
    const int width = out.w;
    
    const int num_class = channels - 5; // 类别数量
    
    // 遍历特征图的每个位置
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int base_idx = i * width + j;
            
            // 从通道0获取前背景分数
            const float* pobj_ptr = out.channel(0);
            float obj_score = pobj_ptr[base_idx];
            
            // 从通道1-4获取边界框回归值
            const float* preg_ptr0 = out.channel(1);
            const float* preg_ptr1 = out.channel(2);
            const float* preg_ptr2 = out.channel(3);
            const float* preg_ptr3 = out.channel(4);
            
            float tx = preg_ptr0[base_idx];
            float ty = preg_ptr1[base_idx];
            float tw = preg_ptr2[base_idx];
            float th = preg_ptr3[base_idx];
            
            // 从通道5+找到最大类别分数和对应索引
            int class_index = 0;
            float class_score = -FLT_MAX;
            for (int k = 0; k < num_class; k++)
            {
                const float* pcls_ptr = out.channel(5 + k);
                float score = pcls_ptr[base_idx];
                if (score > class_score)
                {
                    class_score = score;
                    class_index = k;
                }
            }
            
            // 计算置信度: obj^0.6 * cls^0.4
            float confidence = pow(obj_score, 0.6f) * pow(class_score, 0.4f);
            
            if (confidence < prob_threshold)
                continue;
            
            // 解码边界框
            // bcx = (tanh(tx) + gx) / W
            // bcy = (tanh(ty) + gy) / H
            // bw = sigmoid(tw)
            // bh = sigmoid(th)
            
            float bcx = (fast_tanh(tx) + j) / width;
            float bcy = (fast_tanh(ty) + i) / height;
            float bw = sigmoid(tw);
            float bh = sigmoid(th);
            
            // 转换为 x1, y1, x2, y2
            float x1 = bcx - 0.5f * bw;
            float y1 = bcy - 0.5f * bh;
            float x2 = bcx + 0.5f * bw;
            float y2 = bcy + 0.5f * bh;
            
            // 裁剪到 [0, 1]
            x1 = std::max(0.0f, std::min(1.0f, x1));
            y1 = std::max(0.0f, std::min(1.0f, y1));
            x2 = std::max(0.0f, std::min(1.0f, x2));
            y2 = std::max(0.0f, std::min(1.0f, y2));
            
            // 转换为像素坐标
            x1 *= img_w;
            y1 *= img_h;
            x2 *= img_w;
            y2 *= img_h;
            
            Object obj;
            obj.rect.x = x1;
            obj.rect.y = y1;
            obj.rect.width = x2 - x1;
            obj.rect.height = y2 - y1;
            obj.label = class_index;
            obj.prob = confidence;
            
            objects.push_back(obj);
        }
    }
}

static int detect_fastestdet(const cv::Mat& bgr, std::vector<Object>& objects,
                            const char* param_path, const char* bin_path,
                            int target_size = 352, float prob_threshold = 0.65f, 
                            float nms_threshold = 0.45f)
{
    ncnn::Net fastestdet;

    // 设置NCNN选项
    fastestdet.opt.use_vulkan_compute = false;      // 不使用GPU加速
    fastestdet.opt.num_threads = 4;                 // 使用4线程

    // 加载模型参数
    if (fastestdet.load_param(param_path))
    {
        fprintf(stderr, "Failed to load param: %s\n", param_path);
        return -1;
    }
    if (fastestdet.load_model(bin_path))
    {
        fprintf(stderr, "Failed to load model: %s\n", bin_path);
        return -1;
    }

    cout << "Input name: in0" << endl;
    cout << "Output name: out0" << endl;

    int img_w = bgr.cols;
    int img_h = bgr.rows;
    
    cout << "\nLoading image..." << endl;
    cout << "Image size: " << img_w << "x" << img_h << endl;

    // 图像预处理
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, 
                                                  img_w, img_h, target_size, target_size);

    // 归一化到 [0, 1]
    const float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    in.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = fastestdet.create_extractor();

    ex.input("in0", in);

    // 执行推理并计时
    cout << "\nRunning inference..." << endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    ncnn::Mat out;
    ex.extract("out0", out);
    
    auto end = std::chrono::high_resolution_clock::now();
    float inference_time = std::chrono::duration<float, std::milli>(end - start).count();
    cout << "Inference time: " << inference_time << " ms" << endl;
    
    cout << "Output shape: [1, " << out.c << ", " << out.h << ", " << out.w << "]" << endl;

    std::vector<Object> proposals;

    // 后处理
    cout << "\nPost-processing..." << endl;
    generate_proposals(out, img_w, img_h, prob_threshold, proposals);

    // 按置信度排序
    qsort_descent_inplace(proposals);

    // 应用NMS
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];
    }

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, 
                        const std::vector<std::string>& class_names,
                        const std::string& output_path)
{
    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
        
        int x1 = static_cast<int>(obj.rect.x);
        int y1 = static_cast<int>(obj.rect.y);
        int x2 = static_cast<int>(obj.rect.x + obj.rect.width);
        int y2 = static_cast<int>(obj.rect.y + obj.rect.height);
        
        std::string label;
        if (obj.label < (int)class_names.size())
        {
            label = class_names[obj.label];
        }
        else
        {
            label = "Class_" + std::to_string(obj.label);
        }
        
        // 输出检测结果
        cout << "  " << label << ": " << obj.prob 
             << " at [" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << "]" << endl;

        // 绘制黄色边界框
        cv::Scalar color = cv::Scalar(255, 255, 0);
        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        // 准备置信度文本
        char text[256];
        sprintf(text, "%.2f", obj.prob);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseLine);

        int x = x1;
        int y = y1 - 5;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        // 置信度
        cv::putText(image, text, cv::Point(x, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        // 类别名称
        cv::putText(image, label, cv::Point(x1, y1 - 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }

    // 保存结果图像
    cv::imwrite(output_path, image);
}

int main(int argc, char** argv)
{
    std::string model_path = "../FastestDet.ncnn.param";
    std::string bin_path = "../FastestDet.ncnn.bin";
    std::string image_path = "../test.jpg";
    std::string output_path = "../output.jpg";
    int target_size = 352;
    float prob_threshold = 0.65f;
    float nms_threshold = 0.45f;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc)
        {
            model_path = argv[++i];
            bin_path = model_path;
            size_t pos = bin_path.find(".param");
            if (pos != std::string::npos)
            {
                bin_path.replace(pos, 6, ".bin");
            }
        }
        else if (arg == "--img" && i + 1 < argc)
        {
            image_path = argv[++i];
        }
        else if (arg == "--output" && i + 1 < argc)
        {
            output_path = argv[++i];
        }
        else if (arg == "--conf" && i + 1 < argc)
        {
            prob_threshold = std::stof(argv[++i]);
        }
        else if (arg == "--nms" && i + 1 < argc)
        {
            nms_threshold = std::stof(argv[++i]);
        }
        else if (arg == "--size" && i + 1 < argc)
        {
            target_size = std::stoi(argv[++i]);
        }
        else if (i == 1)
        {
            image_path = argv[i];
        }
        else if (i == 2)
        {
            std::string modeldir = argv[i];
            model_path = modeldir + "/FastestDet.ncnn.param";
            bin_path = modeldir + "/FastestDet.ncnn.bin";
        }
    }

    // 打印配置信息
    cout << "Model: " << model_path << endl;
    cout << "Image: " << image_path << endl;
    cout << "Input size: " << target_size << endl;
    cout << "Conf thresh: " << prob_threshold << endl;
    cout << "NMS thresh: " << nms_threshold << endl;

    // 读取图像
    cv::Mat m = cv::imread(image_path, cv::IMREAD_COLOR);
    if (m.empty())
    {
        fprintf(stderr, "\nError: Failed to load image: %s\n", image_path.c_str());
        return -1;
    }

    cout << "\nLoading model..." << endl;

    // 类别名称
    std::vector<std::string> class_names = {"A", "B"};

    // 检测
    std::vector<Object> objects;
    int ret = detect_fastestdet(m, objects, model_path.c_str(), bin_path.c_str(),
                                target_size, prob_threshold, nms_threshold);
    
    if (ret != 0)
    {
        fprintf(stderr, "Detection failed\n");
        return -1;
    }

    cout << "Detected " << objects.size() << " objects" << endl;
    
    draw_objects(m, objects, class_names, output_path);
    
    cout << "\nResult saved to: " << output_path << endl;

    return 0;
}
