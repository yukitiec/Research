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
/* left */
extern std::queue<std::vector<std::vector<cv::Mat1b>>> queueYoloOldImgSearch_left;      // queue for old image for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Rect2i>>> queueYoloSearchRoi_left;        // queue for search roi for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Mat1b>>> queueOFOldImgSearch_left;        // queue for old image for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Rect2i>>> queueOFSearchRoi_left;          // queue for search roi for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<std::vector<float>>>> queuePreviousMove_left; // queue for saving previous ROI movement : [num human,6 joints, 2D movements]
/* right */
extern std::queue<std::vector<std::vector<cv::Mat1b>>> queueYoloOldImgSearch_right;      // queue for old image for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Rect2i>>> queueYoloSearchRoi_right;        // queue for search roi for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Mat1b>>> queueOFOldImgSearch_right;        // queue for old image for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Rect2i>>> queueOFSearchRoi_right;          // queue for search roi for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<std::vector<float>>>> queuePreviousMove_right; // queue for saving previous ROI movement : [num human,6 joints, 2D movements]
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
extern const int MoveThreshold; //cancell background
extern const float epsironMove;//half range of back ground effect:: a-epsironMove<=flow<=a+epsironMove
/* dense optical flow skip rate */
extern const int skipPixel;
extern const float DIF_THRESHOLD; //threshold for adapting yolo detection's roi
extern const float MIN_MOVE; //minimum opticalflow movement
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
#endif
