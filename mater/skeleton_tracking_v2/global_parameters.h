#pragma once

#ifndef GLOBAL_PARAMETERS_H
#define GLOBAL_PARAMETERS_H

#include "stdafx.h"

extern std::mutex mtxRobot;
/* queueu definition */
/* frame queue */
extern std::queue<std::array<cv::Mat1b, 2>> queueFrame;
extern std::queue<int> queueFrameIndex;
/* yolo and optical flow */

/*3D position*/
extern std::queue<std::vector<std::vector<std::vector<int>>>> queueTriangulation_left;
extern std::queue<std::vector<std::vector<std::vector<int>>>> queueTriangulation_right;
/* from joints to robot control */
extern std::queue<std::vector<std::vector<std::vector<int>>>> queueJointsPositions;
/* notify danger */
extern std::queue<bool> queueDanger;

/* constant valude definition */
extern const std::string filename_left;
extern const std::string filename_right;
extern const int LEFT;
extern const int RIGHT;
extern const bool save;
extern const bool boolSparse;
extern const bool boolGray;
extern const bool boolBatch; //if yolo inference is run in concatenated img
extern const std::string methodDenseOpticalFlow; //"lucasKanade_dense","rlof"
extern const int dense_vel_method; //0: average, 1:max, 2 : median, 3 : third-quarter, 4 : first-quarter
extern const float qualityCorner;
/* roi setting */
extern const int roiWidthOF;
extern const int roiHeightOF;
extern const int roiWidthYolo;
extern const int roiHeightYolo;
extern const float MoveThreshold; //cancell background
extern const float epsironMove;//half range of back ground effect:: a-epsironMove<=flow<=a+epsironMove
/* dense optical flow skip rate */
extern const int skipPixel;
extern const float DIF_THRESHOLD; //threshold for adapting yolo detection's roi
extern const float MIN_MOVE; //minimum opticalflow movement
extern const float MAX_MOVE;
/*if exchange template of Yolo */
extern const bool boolChange;
/* save date */
extern const std::string file_yolo_left;
extern const std::string file_yolo_right;
extern const std::string file_of_left;
extern const std::string file_of_right;
extern const std::string file_3d;

/* 3D triangulation */
extern const int BASELINE; // distance between 2 cameras
// std::vector<std::vector<float>> cameraMatrix{ {179,0,160},{0,179,160},{0,0,1} }; //camera matrix from camera calibration

/* revise here based on camera calibration */
extern const cv::Mat cameraMatrix;
extern const cv::Mat distCoeffs;
/* transformation matrix from camera coordinate to robot base coordinate */
extern const std::vector<std::vector<float>> transform_cam2base;

//structure
struct Yolo2optflow {
	std::vector<std::vector<cv::Rect2i>> roi; //search ROI
	std::vector<std::vector<cv::Mat1b>> img_search; //search background img
};

struct Optflow2optflow {
	std::vector<std::vector<cv::Rect2i>> roi; //search ROI
	std::vector<std::vector<cv::Mat1b>> img_search; //search img
	std::vector<std::vector<std::vector<float>>> move; //previous target movement
	std::vector<std::vector<cv::Ptr<cv::DISOpticalFlow>>> ptr_dis; //DIS pointer
};

//queue
extern std::queue<Yolo2optflow> q_yolo2optflow_left, q_yolo2optflow_right;
extern std::queue<Optflow2optflow> q_optflow2optflow_left, q_optflow2optflow_right;


#endif