#pragma once
#ifndef YOLOPOSE_BATCH_H
#define YOLOPOSE_BATCH_H

#include "stdafx.h"

extern const int LEFT;
extern const int RIGHT;
extern std::queue<std::array<cv::Mat1b, 2>> queueFrame;
extern std::queue<int> queueFrameIndex;

/*  YOLO class definition  */
class YOLOPoseBatch
{

private:
    torch::jit::script::Module mdl;
    torch::DeviceType devicetype;
    torch::Device* device;

    std::string yolofilePath = "yolov8m-pose_640_1280.torchscript";
    const int originalWidth = 640;
    const int originalHeight = 640;
    int frameWidth = 640;
    int frameHeight = 1280;
    const int yoloWidth = 640;
    const int yoloHeight = 1280;
    const cv::Size YOLOSize{ yoloWidth, yoloHeight };
    const float IoUThreshold = 0.1;
    const float ConfThreshold = 0.4;
    const float IoUThresholdIdentity = 0.25; // for maitainig consistency of tracking
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

    void detect(cv::Mat1b& frame, int& frameIndex, int& counter, std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver_left, std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver_right,
        std::queue<std::vector<std::vector<cv::Mat1b>>>& queueYoloOldImgSearch_left, std::queue<std::vector<std::vector<cv::Rect2i>>>& queueYoloSearchRoi_left,
        std::queue<std::vector<std::vector<cv::Mat1b>>>& queueYoloOldImgSearch_right, std::queue<std::vector<std::vector<cv::Rect2i>>>& queueYoloSearchRoi_right); //, const int frameIndex, std::vector<std::vector<cv::Rect2i>>& posSaver, std::vector<std::vector<int>>& classSaver)

    void preprocessImg(cv::Mat1b& frame, torch::Tensor& imgTensor);
    
    void nonMaxSuppressionHuman(torch::Tensor& prediction, std::vector<torch::Tensor>& detectedBoxesHuman, float confThreshold, float iouThreshold);

    torch::Tensor xywh2xyxy(torch::Tensor& x);

    void nms(torch::Tensor& x, std::vector<torch::Tensor>& detectedBoxes, float& iouThreshold, bool& boolLeft, bool& boolRight);

    float calculateIoU(const torch::Tensor& box1, const torch::Tensor& box2);

    void keyPointsExtractor(std::vector<torch::Tensor>& detectedBboxesHuman, std::vector<std::vector<std::vector<int>>>& keyPoints, std::vector<int>& humanPos, const int& ConfThreshold);

    void drawCircle(cv::Mat1b& frame, std::vector<std::vector<std::vector<int>>>& ROI, int& counter);

    void push2Queue(cv::Mat1b& frame, int& frameIndex, std::vector<std::vector<std::vector<int>>>& keyPoints,
        std::vector<cv::Rect2i>& roiLatest, std::vector<int>& humanPos, std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver_left, std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver_right,
        std::queue<std::vector<std::vector<cv::Mat1b>>>& queueYoloOldImgSearch_left, std::queue<std::vector<std::vector<cv::Rect2i>>>& queueYoloSearchRoi_left,
        std::queue<std::vector<std::vector<cv::Mat1b>>>& queueYoloOldImgSearch_right, std::queue<std::vector<std::vector<cv::Rect2i>>>& queueYoloSearchRoi_right);

    void organize_left(cv::Mat1b& frame, int& frameIndex, std::vector<int>& pos, std::vector<cv::Rect2i>& joints, std::vector<cv::Mat1b>& imgJoint, std::vector<std::vector<int>>& jointsCenter);

    void organize_right(cv::Mat1b& frame, int& frameIndex, std::vector<int>& pos, std::vector<cv::Rect2i>& joints, std::vector<cv::Mat1b>& imgJoint, std::vector<std::vector<int>>& jointsCenter);
};

#endif