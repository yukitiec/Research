#pragma once

#ifndef OPTICALFLOW_H
#define OPTICALFLOW_H

#include "stdafx.h"
#include "global_parameters.h"

extern const float DIF_THRESHOLD;
extern const float MIN_MOVE; //minimum opticalflow movement
extern const int roiWidthOF;
extern const int roiHeightOF;
extern const int dense_vel_method;
extern const float MoveThreshold;
extern const float MAX_MOVE;
extern const bool boolChange;
extern const bool bool_dynamic_roi; //adopt dynamic roi
extern const bool bool_rotate_roi;

extern std::queue<std::array<cv::Mat1b, 2>> queueFrame;
extern std::queue<int> queueFrameIndex;

//queue
extern std::queue<Yolo2optflow> q_yolo2optflow_left, q_yolo2optflow_right;
extern std::queue<Optflow2optflow> q_optflow2optflow_left, q_optflow2optflow_right;

class OpticalFlow
{
private:
    const cv::Size ROISize{ roiWidthOF, roiHeightOF };
    const int originalWidth = 640;
    const int originalHeight = 640;
    //for dividing regions
    const std::vector<float> vec_pp{ 1,1 }; //theta = 45 degree
    const std::vector<float> vec_np{ -1,1 }; //theta = 135 degree
    const std::vector<float> vec_nn{ -1,-1 }; //theta = 225 degree
    const std::vector<float> vec_pn{ 1,-1 }; //theta = 315 
    const float MAX_CHANGE = 30; //max change of joints per frame
    const int optflow_method = 1;  //0 : DIS, 1: farneback(edge), 2: RLOF(Robust Local Optical flow)
    cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT)+(cv::TermCriteria::EPS), 20, 0.001); //optical flow calculation criteria
    const bool bool_moveROI = true;//done->true
    const int MIN_SEARCH = 15;
    const float alpha = 0.7; //predict velocity :: alpha*v_t+(1-alpha)*v_(t-1)
    const float max_vel = 15; //max veloctiy
    const int dis_mode = 0; //0 : ultrafast, 1:fast,2: medium
    const bool bool_manual_patch_dis = false;
    const int disPatch = 5;
    const int disStride = 1;
    const bool bool_check_pos = false; //false :: whether update roi all time with Yolo
    const float min_move_track = 0.0; //minimum movement by optflow between each detection frame
    const std::vector<float> defaultMove{ 0.0,0.0 };
public:
    OpticalFlow()
    {
        std::cout << "construct OpticalFlow class" << std::endl;
    }

    void main(cv::Mat1b& frame, const int& frameIndex, std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver,
        std::queue<Yolo2optflow>& q_yolo2optflow,
        std::queue<Optflow2optflow>& q_optflow2optflow,
        std::queue<std::vector<std::vector<std::vector<int>>>>& queueTriangulation);

    void getPreviousData(cv::Mat1b& frame, std::vector<std::vector<cv::Mat1b>>& previousImg, std::vector<std::vector<cv::Rect2i>>& searchRoi,
        std::vector<std::vector<std::vector<float>>>& moveDists, std::vector<std::vector<cv::Ptr<cv::DISOpticalFlow>>>& previousDIS,
        std::queue<Yolo2optflow>& q_yolo2optflow, std::queue<Optflow2optflow>& q_optflow2optflow);


    void getYoloData(std::vector<std::vector<cv::Mat1b>>& previousYoloImg, std::vector<std::vector<cv::Rect2i>>& searchYoloRoi,
        std::queue<Yolo2optflow>& q_yolo2optflow);

    void opticalFlow(const cv::Mat1b frame, const int& frameIndex,
        cv::Mat1b& previousImg, cv::Rect2i& searchRoi, std::vector<float>& previousMove, cv::Ptr<cv::DISOpticalFlow>& dis,
        cv::Mat1b& updatedImg, cv::Rect2i& updatedSearchRoi, cv::Ptr<cv::DISOpticalFlow>& updatedDIS, std::vector<float>& updatedMove, std::vector<int>& updatedPos);

    float calculateMedian(std::vector<float> vec);

    float calculateThirdQuarter(std::vector<float> vec);

    float calculateFirstQuarter(std::vector<float> vec);
};

#endif