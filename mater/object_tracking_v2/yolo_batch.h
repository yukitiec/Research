#pragma once

#ifndef YOLO_BATCH_H
#define YOLO_BATCH_H

#include "stdafx.h"
#include "mosse.h"
#include "global_parameters.h"

extern std::queue<std::array<cv::Mat1b, 2>> queueFrame; // queue for frame
extern std::queue<int> queueFrameIndex;  // queue for frame index
//start signal
extern std::queue<bool> q_startTracker;
// data from Yolo to tracker
extern std::queue<Yolo2tracker> q_yolo2tracker_left, q_yolo2tracker_right;
//data from tracker to Yolo
extern std::queue<Tracker2yolo> q_tracker2yolo_left, q_tracker2yolo_right;
extern const bool boolGroundTruth;


/*  YOLO class definition  */
class YOLODetect_batch
{
private:
    torch::jit::script::Module mdl;
    torch::DeviceType devicetype;
    torch::Device* device;

    std::string yolofilePath = "yolov8n_last_320_640.torchscript";
    const int originalWidth = 640;
    const int orginalHeight = 640;
    const int frameWidth = 1280;
    const int frameHeight = 640;
    const int yoloWidth = 640;
    const int yoloHeight = 320;
    const cv::Size YOLOSize{ yoloWidth, yoloHeight };
    const float IoUThreshold = 0.3; //throwing :: 0.25
    const float ConfThreshold = 0.3;
    const float IoUThresholdIdentity = 0.3; // for maitainig consistency of tracking
    const float Rmse_identity = 10.0; // minimum rmse criteria
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
    YOLODetect_batch()
    {
        initializeDevice();
        loadModel();
        std::cout << "YOLO construtor has finished!" << std::endl;
    };
    ~YOLODetect_batch() { delete device; }; // Deconstructor

    //main 
    void detect(cv::Mat1b& frame, const int frameIndex, std::vector<std::vector<cv::Rect2d>>& posSaver_left, std::vector<std::vector<cv::Rect2d>>& posSaver_right,
        std::vector<std::vector<int>>& classSaver_left, std::vector<std::vector<int>>& classSaver_right,
        std::vector<int>& detectedFrame_left, std::vector<int>& detectedFrame_right, std::vector<int>& detectedFrameClass_left, std::vector<int>& detectedFrameClass_right, int counterIteration,bool& bool_startTracker);

    //preprocess img
    void preprocessImg(cv::Mat1b& frame, torch::Tensor& imgTensor);

    //get latest data from Tracker2yolo
    void getLatestData(std::vector<cv::Rect2d>& bboxes_left, std::vector<cv::Rect2d>& bboxes_right, std::vector<int>& classes_left, std::vector<int>& classes_right);
    
    //non-max suppression
    void non_max_suppression2(torch::Tensor& prediction, std::vector<torch::Tensor>& detectedBoxes0, std::vector<torch::Tensor>& detectedBoxes1);
    
    //non-max suppression unit module
    void non_max_suppression_unit(torch::Tensor& x, std::vector<torch::Tensor>& detectedBoxes);

    //center -> roi
    torch::Tensor xywh2xyxy(torch::Tensor& x);

    //delete overlapped bboxes
    void nms(torch::Tensor& x, std::vector<torch::Tensor>& detectedBoxes);

    //calculate IoU
    float calculateIoU(const torch::Tensor& box1, const torch::Tensor& box2);

    //organize detecion data
    void roiSetting(std::vector<torch::Tensor>& detectedBoxes, std::vector<cv::Rect2d>& existedRoi_left, std::vector<int>& existedClass_left, std::vector<cv::Rect2d>& newRoi_left, std::vector<int>& newClass_left,
        std::vector<cv::Rect2d>& existedRoi_right, std::vector<int>& existedClass_right, std::vector<cv::Rect2d>& newRoi_right, std::vector<int>& newClass_right,
        int candidateIndex,
        std::vector<cv::Rect2d>& bboxesCandidate_left, std::vector<int>& classIndexesTM_left, std::vector<cv::Rect2d>& bboxesCandidate_right, std::vector<int>& classIndexesTM_right);

    //IoU
    float calculateIoU_Rect2d(const cv::Rect2d& box1, const cv::Rect2d& box2);
    
    //RMSE
    float calculateRMSE_Rect2d(const cv::Rect2d& box1, const cv::Rect2d& box2);
   
    //comparison between Yolo and Tracker
    void comparisonTMYolo(std::vector<torch::Tensor>& detectedBoxes, int& candidateIndex, std::vector<cv::Rect2d>& bboxesYolo_left, std::vector<cv::Rect2d>& bboxesYolo_right,
        std::vector<int>& classIndexesTM_left, std::vector<int>& classIndexesTM_right,
        std::vector<cv::Rect2d>& bboxesCandidate_left, std::vector<cv::Rect2d>& bboxesCandidate_right,
        std::vector<cv::Rect2d>& existedRoi_left, std::vector<int>& existedClass_left, std::vector<cv::Rect2d>& existedRoi_right, std::vector<int>& existedClass_right);

    //tracker matching
    void matchingTracker(int& candidateIndex, std::vector<cv::Rect2d>& bboxesCandidate, std::vector<int>& classIndexesTM, std::vector<cv::Rect2d>& bboxesYolo, std::vector<cv::Rect2d>& existedRoi, std::vector<int>& existedClass);

    //newly detected bbox
    void newDetection(int& candidateIndex, std::vector<cv::Rect2d>& bboxesYolo, std::vector<cv::Rect2d>& newRoi, std::vector<int>& newClass);
    
    //no detection by YOLO
    void noYoloDetect(int& candidateIndex, std::vector<cv::Rect2d>& bboxesCandidate, std::vector<int>& classIndexesTM, std::vector<int>& existedClass);

    //push data to queue
    void push2Queue_left(std::vector<cv::Rect2d>& existedRoi, std::vector<cv::Rect2d>& newRoi,
        std::vector<int>& existedClass, std::vector<int>& newClass, cv::Mat1b& frame,
        std::vector<std::vector<cv::Rect2d>>& posSaver, std::vector<std::vector<int>>& classSaver,
        const int& frameIndex, std::vector<int>& detectedFrame, std::vector<int>& detectedFrameClass,
        std::queue<Yolo2tracker>& q_yolo2tracker);

    void push2Queue_right(std::vector<cv::Rect2d>& existedRoi, std::vector<cv::Rect2d>& newRoi,
        std::vector<int>& existedClass, std::vector<int>& newClass, cv::Mat1b& frame,
        std::vector<std::vector<cv::Rect2d>>& posSaver, std::vector<std::vector<int>>& classSaver,
        const int& frameIndex, std::vector<int>& detectedFrame, std::vector<int>& detectedFrameClass,
        std::queue<Yolo2tracker>& q_yolo2tracker);

    //update data
    void updateData_left(std::vector<cv::Rect2d>& existedRoi, std::vector<cv::Rect2d>& newRoi, std::vector<int>& existedClass, std::vector<int>& newClass,
        cv::Mat1b& frame, std::vector<cv::Rect2d>& updatedRoi, std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers, std::vector<cv::Mat1b>& updatedTemplates,
        std::vector<int>& updatedClassIndexes);

    void updateData_right(std::vector<cv::Rect2d>& existedRoi, std::vector<cv::Rect2d>& newRoi, std::vector<int>& existedClass, std::vector<int>& newClass,
        cv::Mat1b& frame, std::vector<cv::Rect2d>& updatedRoi, std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers, std::vector<cv::Mat1b>& updatedTemplates,
        std::vector<int>& updatedClassIndexes);
};

#endif