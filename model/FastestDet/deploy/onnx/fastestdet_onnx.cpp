#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#ifdef RASPBERRY_PI
#include <arm_neon.h>
#endif

// Activation functions
inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

inline float tanh_activate(float x) {
    return tanhf(x);
}

// Detection box structure
struct Detection {
    cv::Rect2f bbox;
    float confidence;
    int class_id;
};

// COCO class names
static const std::vector<std::string> COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

class FastestDetONNX {
private:
    std::unique_ptr<Ort::Session> session_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::vector<std::string> input_names_str_;
    std::vector<std::string> output_names_str_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
    
    int input_width_ = 352;
    int input_height_ = 352;
    float conf_threshold_ = 0.6f;
    float nms_threshold_ = 0.45f;
    int num_classes_ = 80;

public:
    FastestDetONNX() : env_(ORT_LOGGING_LEVEL_WARNING, "FastestDet-ONNX") {
        session_options_.SetIntraOpNumThreads(0); // Use all available cores
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
#ifdef RASPBERRY_PI
        std::cout << "Raspberry Pi optimization mode enabled" << std::endl;
        session_options_.SetIntraOpNumThreads(4); // Raspberry Pi 4B has 4 cores
#endif
    }
    
    bool LoadModel(const std::string& model_path) {
        try {
            std::cout << "Loading ONNX model: " << model_path << std::endl;
            
#ifdef _WIN32
            std::wstring wide_model_path(model_path.begin(), model_path.end());
            session_ = std::make_unique<Ort::Session>(env_, wide_model_path.c_str(), session_options_);
#else
            session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
#endif
            
            // Get input and output information
            Ort::AllocatorWithDefaultOptions allocator;
            
            // Input information
            size_t num_input_nodes = session_->GetInputCount();
            input_names_str_.clear();
            input_names_.clear();
            for (size_t i = 0; i < num_input_nodes; i++) {
                auto input_name = session_->GetInputNameAllocated(i, allocator);
                input_names_str_.push_back(std::string(input_name.get()));
                input_names_.push_back(input_names_str_.back().c_str());
                
                Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(i);
                auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
                auto input_dims = input_tensor_info.GetShape();
                input_shapes_.push_back(input_dims);
                
                std::cout << "Input[" << i << "]: " << input_names_str_.back();
                std::cout << " shape: [";
                for (size_t j = 0; j < input_dims.size(); j++) {
                    std::cout << input_dims[j];
                    if (j < input_dims.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }
            
            // Output information
            size_t num_output_nodes = session_->GetOutputCount();
            output_names_str_.clear();
            output_names_.clear();
            for (size_t i = 0; i < num_output_nodes; i++) {
                auto output_name = session_->GetOutputNameAllocated(i, allocator);
                output_names_str_.push_back(std::string(output_name.get()));
                output_names_.push_back(output_names_str_.back().c_str());
                
                Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(i);
                auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
                auto output_dims = output_tensor_info.GetShape();
                output_shapes_.push_back(output_dims);
                
                std::cout << "Output[" << i << "]: " << output_names_str_.back();
                std::cout << " shape: [";
                for (size_t j = 0; j < output_dims.size(); j++) {
                    std::cout << output_dims[j];
                    if (j < output_dims.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }
            
            std::cout << "Model loaded successfully!" << std::endl;
            return true;
            
        } catch (const Ort::Exception& e) {
            std::cerr << "Failed to load ONNX model: " << e.what() << std::endl;
            return false;
        }
    }
    
    cv::Mat Preprocess(const cv::Mat& image) {
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(input_width_, input_height_));
        
        // Convert to RGB and normalize
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32F, 1.0f / 255.0f);
        
        return rgb;
    }
    
    std::vector<Detection> Postprocess(const std::vector<Ort::Value>& outputs, 
                                      const cv::Size& original_size) {
        const float* output_data = outputs[0].GetTensorData<float>();
        auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        
        int feat_h = static_cast<int>(output_shape[2]);
        int feat_w = static_cast<int>(output_shape[3]);
        int channels = static_cast<int>(output_shape[1]);
        
        std::vector<Detection> detections;
        
        for (int h = 0; h < feat_h; h++) {
            for (int w = 0; w < feat_w; w++) {
                int base_idx = h * feat_w + w;
                
                // Foreground probability
                float obj_score = sigmoid(output_data[base_idx]);
                
                // Find highest scoring class
                float max_cls_score = 0.0f;
                int max_cls_id = 0;
                for (int c = 0; c < num_classes_; c++) {
                    int cls_idx = (5 + c) * feat_h * feat_w + base_idx;
                    float cls_score = sigmoid(output_data[cls_idx]);
                    if (cls_score > max_cls_score) {
                        max_cls_score = cls_score;
                        max_cls_id = c;
                    }
                }
                
                // Calculate final score
                float final_score = pow(obj_score, 0.6f) * pow(max_cls_score, 0.4f);
                
                if (final_score > conf_threshold_) {
                    // Parse coordinates
                    float x_offset = tanh_activate(output_data[feat_h * feat_w + base_idx]);
                    float y_offset = tanh_activate(output_data[2 * feat_h * feat_w + base_idx]);
                    float box_w = sigmoid(output_data[3 * feat_h * feat_w + base_idx]);
                    float box_h = sigmoid(output_data[4 * feat_h * feat_w + base_idx]);
                    
                    float cx = (w + x_offset) / feat_w;
                    float cy = (h + y_offset) / feat_h;
                    
                    float x1 = (cx - box_w * 0.5f) * original_size.width;
                    float y1 = (cy - box_h * 0.5f) * original_size.height;
                    float x2 = (cx + box_w * 0.5f) * original_size.width;
                    float y2 = (cy + box_h * 0.5f) * original_size.height;
                    
                    Detection detection;
                    detection.bbox = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
                    detection.confidence = final_score;
                    detection.class_id = max_cls_id;
                    detections.push_back(detection);
                }
            }
        }
        
        // NMS
        return ApplyNMS(detections);
    }
    
    std::vector<Detection> ApplyNMS(std::vector<Detection>& detections) {
        std::sort(detections.begin(), detections.end(), 
                 [](const Detection& a, const Detection& b) {
                     return a.confidence > b.confidence;
                 });
        
        std::vector<Detection> result;
        std::vector<bool> suppressed(detections.size(), false);
        
        for (size_t i = 0; i < detections.size(); ++i) {
            if (suppressed[i]) continue;
            
            result.push_back(detections[i]);
            
            for (size_t j = i + 1; j < detections.size(); ++j) {
                if (suppressed[j] || detections[i].class_id != detections[j].class_id) continue;
                
                float iou = CalculateIoU(detections[i].bbox, detections[j].bbox);
                if (iou > nms_threshold_) {
                    suppressed[j] = true;
                }
            }
        }
        
        return result;
    }
    
    float CalculateIoU(const cv::Rect2f& box1, const cv::Rect2f& box2) {
        float x1 = std::max(box1.x, box2.x);
        float y1 = std::max(box1.y, box2.y);
        float x2 = std::min(box1.x + box1.width, box2.x + box2.width);
        float y2 = std::min(box1.y + box1.height, box2.y + box2.height);
        
        if (x2 <= x1 || y2 <= y1) return 0.0f;
        
        float intersection = (x2 - x1) * (y2 - y1);
        float area1 = box1.width * box1.height;
        float area2 = box2.width * box2.height;
        float union_area = area1 + area2 - intersection;
        
        return intersection / union_area;
    }
    
    std::vector<Detection> Detect(const cv::Mat& image) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Preprocessing
        cv::Mat processed = Preprocess(image);
        
        // Create input tensor
        std::vector<int64_t> input_dims = {1, 3, input_height_, input_width_};
        size_t input_tensor_size = 1 * 3 * input_height_ * input_width_;
        
        std::vector<float> input_tensor_values(input_tensor_size);
        
        // Convert HWC to CHW format
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < input_height_; h++) {
                for (int w = 0; w < input_width_; w++) {
                    input_tensor_values[c * input_height_ * input_width_ + h * input_width_ + w] = 
                        processed.at<cv::Vec3f>(h, w)[c];
                }
            }
        }
        
        // Create input tensor
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_size,
            input_dims.data(), input_dims.size()
        );
        
        // Inference
        auto inference_start = std::chrono::high_resolution_clock::now();
        std::vector<Ort::Value> outputs = session_->Run(
            Ort::RunOptions{nullptr}, 
            input_names_.data(), &input_tensor, 1,
            output_names_.data(), output_names_.size()
        );
        auto inference_end = std::chrono::high_resolution_clock::now();
        
        // Postprocessing
        std::vector<Detection> detections = Postprocess(outputs, image.size());
        
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(inference_end - inference_start);
        
        std::cout << "Inference time: " << inference_time.count() << "ms" << std::endl;
        std::cout << "Total time: " << total_time.count() << "ms" << std::endl;
        std::cout << "FPS: " << std::fixed << std::setprecision(1) 
                  << 1000.0f / total_time.count() << std::endl;
        
        return detections;
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <onnx_model_path> <image_path>" << std::endl;
        std::cout << "Example: " << argv[0] << " models/fastestdet.onnx test_image.jpg" << std::endl;
        return -1;
    }
    
    std::string model_path = argv[1];
    std::string image_path = argv[2];
    
    // Create detector
    FastestDetONNX detector;
    
    // Load model
    if (!detector.LoadModel(model_path)) {
        return -1;
    }
    
    // Load image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }
    
    std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;
    
    // Detection
    std::vector<Detection> detections = detector.Detect(image);
    
    // Display results
    std::cout << "Detected " << detections.size() << " objects:" << std::endl;
    for (const auto& detection : detections) {
        const std::string& class_name = COCO_CLASSES[detection.class_id];
        std::cout << "  " << class_name 
                  << ": " << std::fixed << std::setprecision(1) << detection.confidence * 100 << "%" 
                  << " [" << (int)detection.bbox.x << "," << (int)detection.bbox.y 
                  << "," << (int)(detection.bbox.x + detection.bbox.width) 
                  << "," << (int)(detection.bbox.y + detection.bbox.height) << "]" << std::endl;
        
        // Draw detection box
        cv::rectangle(image, detection.bbox, cv::Scalar(0, 255, 0), 2);
        
        std::string label = class_name + ": " + std::to_string((int)(detection.confidence * 100)) + "%";
        cv::putText(image, label, 
                   cv::Point(detection.bbox.x, detection.bbox.y - 10), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }
    
    // Save result
    std::string output_path = "result_onnx.jpg";
    cv::imwrite(output_path, image);
    std::cout << "Result saved to: " << output_path << std::endl;
    
    return 0;
}
