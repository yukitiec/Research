#pragma once

#ifndef YOLOPOSE_BATCH_H
#define YOLOPOSE_BATCH_H

#include "stdafx.h"
#include "global_parameters.h"

extern const int LEFT;
extern const int RIGHT;
/* roi setting */
extern const bool bool_dynamic_roi; //adopt dynamic roi
extern const bool bool_rotate_roi;
//if true
extern const float max_half_diagonal;
extern const float min_half_diagonal;
//if false : static roi
extern const int roiWidthOF;
extern const int roiHeightOF;
extern const int roiWidthYolo;
extern const int roiHeightYolo;

extern std::queue<std::array<cv::Mat1b, 2>> queueFrame;
extern std::queue<int> queueFrameIndex;

/*  YOLO class definition  */
class YOLOPoseBatch
{
private:
    torch::jit::script::Module mdl;
    torch::DeviceType devicetype;
    torch::Device* device;

    std::string yolofilePath = "yolov8m-pose_320_640.torchscript";
    const int originalWidth = 640;
    const int originalHeight = 640;
    int frameWidth = 1280;
    int frameHeight = 640;
    const int yoloWidth = 640;
    const int yoloHeight = 320;
    const int boundary_img = 320;
    const cv::Size YOLOSize{ yoloWidth, yoloHeight };
    const float IoUThreshold = 0.1;
    const float ConfThreshold = 0.35;
    const float IoUThresholdIdentity = 0.25; // for maitainig consistency of tracking
    const int num_joints = 6; //number of tracked joints
    const float roi_direction_threshold = 2.0; //max gradient of neighborhood joints
    std::vector<float> default_neighbor{ (float)(std::pow(2,0.5) / 2),(float)(std::pow(2,0.5) / 2) }; //45 degree direction
    const int MIN_SEARCH = 10; //minimum search size
    const float min_ratio = 0.65;//minimum ratio for the max value
    /* initialize function */
    void initializeDevice()
    {
        // set device
        if (torch::cuda::is_available())
        {
            // device = new torch::Device(devicetype, 0);
            device = new torch::Device(torch::kCUDA);
            std::cout << "set cuda" << std::endl;
        }
        else
        {
            device = new torch::Device(torch::kCPU);
            std::cout << "set CPU" << std::endl;
        }
    }

    void loadModel()
    {
        // read param
        mdl = torch::jit::load(yolofilePath, *device);
        mdl.to(*device);
        mdl.eval();
        std::cout << "load model" << std::endl;
    }

public:
    // constructor for YOLODetect
    YOLOPoseBatch()
    {
        initializeDevice();
        loadModel();
        std::cout << "YOLOBatch construtor has finished!" << std::endl;
    };
    ~YOLOPoseBatch() { delete device; }; // Deconstructor

    void detect(cv::Mat1b& frame, int& frameIndex, int& counter, 
        std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver_left, std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver_right,
        std::queue<Yolo2optflow>& q_yolo2optflow_left, std::queue<Yolo2optflow>& q_yolo2optflow_right);

    void preprocessImg(cv::Mat1b& frame, torch::Tensor& imgTensor);

    void nonMaxSuppressionHuman(torch::Tensor& prediction, std::vector<torch::Tensor>& detectedBoxesHuman, float confThreshold, float iouThreshold);

    torch::Tensor xywh2xyxy(torch::Tensor& x);

    void nms(torch::Tensor& x, std::vector<torch::Tensor>& detectedBoxes, float& iouThreshold, bool& boolLeft, bool& boolRight);

    float calculateIoU(const torch::Tensor& box1, const torch::Tensor& box2);

    void keyPointsExtractor(std::vector<torch::Tensor>& detectedBboxesHuman, std::vector<std::vector<std::vector<int>>>& keyPoints, std::vector<int>& humanPos, const int& ConfThreshold);

    void drawCircle(cv::Mat1b& frame, std::vector<std::vector<std::vector<int>>>& ROI, int& counter);

    void push2Queue(cv::Mat1b& frame, int& frameIndex, std::vector<std::vector<std::vector<int>>>& keyPoints,
        std::vector<cv::Rect2i>& roiLatest, std::vector<int>& humanPos,
        std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver_left, std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver_right,
        std::queue<Yolo2optflow>& q_yolo2optflow_left, std::queue<Yolo2optflow>& q_yolo2optflow_right);

    void organizeRoi(cv::Mat1b& frame, int& frameIndex, bool& bool_left, std::vector<std::vector<int>>& pos, std::vector<std::vector<float>>& distances,
        std::vector<cv::Rect2i>& joints, std::vector<cv::Mat1b>& imgJoint, std::vector<std::vector<int>>& jointsCenter);
    
    void setRoi(int& frameIndex, cv::Mat1b& frame, bool& bool_left, std::vector<std::vector<float>>& distances,
        int& index_joint, std::vector<int>& compareJoints, std::vector<std::vector<int>>& pos,
        std::vector<cv::Rect2i>& joints, std::vector<cv::Mat1b>& imgJoint, std::vector<std::vector<int>>& jointsCenter);

    void defineRoi_left(int& frameIndex, cv::Mat1b& frame, int& index_joint, float& vx, float& vy,
        float& half_diagonal, std::vector<std::vector<int>>& pos,
        std::vector<cv::Rect2i>& joints, std::vector<cv::Mat1b>& imgJoint, std::vector<std::vector<int>>& jointsCenter);

    void defineRoi_right(int& frameIndex, cv::Mat1b& frame, int& index_joint, float& vx, float& vy, float& half_diagonal, 
        std::vector<std::vector<int>>& pos,
        std::vector<cv::Rect2i>& joints, std::vector<cv::Mat1b>& imgJoint, std::vector<std::vector<int>>& jointsCenter);

    void organize_left(cv::Mat1b& frame, int& frameIndex, std::vector<int>& pos, std::vector<cv::Rect2i>& joints, std::vector<cv::Mat1b>& imgJoint, std::vector<std::vector<int>>& jointsCenter);

    void organize_right(cv::Mat1b& frame, int& frameIndex, std::vector<int>& pos, std::vector<cv::Rect2i>& joints, std::vector<cv::Mat1b>& imgJoint, std::vector<std::vector<int>>& jointsCenter);
};

#endif