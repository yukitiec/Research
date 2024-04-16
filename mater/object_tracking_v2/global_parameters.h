#pragma once


#ifndef GLOBAL_PARAMETERS_H
#define GLOBAL_PARAMETERS_H

#include "stdafx.h"
#include "mosse.h"

extern const bool boolGroundTruth;
//video path
extern const std::string filename_left;
extern const std::string filename_right;
// camera : constant setting
extern const int LEFT_CAMERA;
extern const int RIGHT_CAMERA;
extern const int FPS;
// YOLO label
extern const int BALL;
extern const int BOX;
// tracker
extern const double threshold_mosse;//0.57; //PSR threshold

//Kalman filter setting
extern const double INIT_X;
extern const double INIT_Y;
extern const double INIT_VX;
extern const double INIT_VY;
extern const double INIT_AX;
extern const double INIT_AY;
extern const double NOISE_POS;
extern const double NOISE_VEL;
extern const double NOISE_ACC;
extern const double NOISE_SENSOR;
// tracking
extern const int COUNTER_VALID; //frames by official tracker
extern const int COUNTER_LOST; //frames by deleting tracker
extern const float MAX_ROI_RATE; //max change of roi
extern const float MIN_ROI_RATE; //minimum change of roi
extern const double MIN_IOU; //minimum IoU for identity
extern const double MAX_RMSE; //max RMSE fdor identity

/* 3d positioning by stereo camera */
extern const int BASELINE; // distance between 2 cameras
/* camera calibration result */
extern const cv::Mat cameraMatrix;
extern const cv::Mat distCoeffs;
/* transformation matrix from camera coordinate to robot base coordinate */
extern const std::vector<std::vector<float>> transform_cam2base;
//3d objects number
extern const int numObjects;

/* UR catching point */
extern const int TARGET_DEPTH; // catching point is 40 cm away from camera position

/* save file setting */
extern const std::string file_yolo_bbox_left;
extern const std::string file_yolo_class_left;
extern const std::string file_tm_bbox_left;
extern const std::string file_tm_class_left;
extern const std::string file_seq_left;
extern const std::string file_kf_left;
extern const std::string file_yolo_bbox_right;
extern const std::string file_yolo_class_right;
extern const std::string file_tm_bbox_right;
extern const std::string file_tm_class_right;
extern const std::string file_seq_right;
extern const std::string file_kf_right;
extern const std::string file_3d;
extern const std::string file_target;

//structure
//Yolo to Tracker
struct Yolo2tracker {
    std::vector<int> classIndex;
    std::vector<cv::Rect2d> bbox;
    std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>> tracker;
    std::vector<cv::Mat1b> templateImg;
};

//Tracker to Yolo
struct Tracker2yolo {
    std::vector<int> classIndex;
    std::vector<cv::Rect2d> bbox;
};

//Tracker to Tracker
struct Tracker2tracker {
    std::vector<int> classIndex;
    std::vector<cv::Rect2d> bbox;
    std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>> tracker;
    std::vector<cv::Mat1b> templateImg;
    std::vector<bool> scale; //if Yolo scale or tracker scale : scale ratio of search area against bbox
    std::vector<std::vector<int>> vel; //previous velocity
    cv::Mat1b previousImg;
    std::vector<int> num_notMove;//number of not move
};

//Tracker to Sequence
struct Tracker2seq {
    int frameIndex;
    std::vector<int> classIndex;
    std::vector<cv::Rect2d> bbox;
};

//queue data
//image
extern std::queue<std::array<cv::Mat1b, 2>> queueFrame; // queue for frame
extern std::queue<int> queueFrameIndex;  // queue for frame index
//start signal
extern std::queue<bool> q_startTracker;
//yolo to tracker
extern std::queue<Yolo2tracker> q_yolo2tracker_left, q_yolo2tracker_right;
//tracker to yolo
extern std::queue<Tracker2yolo> q_tracker2yolo_left, q_tracker2yolo_right;
//tracker2tracker
extern std::queue<Tracker2tracker> q_tracker2tracker_left, q_tracker2tracker_right;
//tracker2seq
extern std::queue<Tracker2seq> q_tracker2seq_left, q_tracker2seq_right;
//seq2tracker
extern std::queue<std::vector<std::vector<double>>> q_seq2tracker_left, q_seq2tracker_right;

// sequential data
extern std::vector<std::vector<std::vector<double>>> seqData_left, seqData_right; //storage for sequential data to share with triangulation.h

//matching
extern std::queue<std::vector<int>> queueUpdateLabels_left;
extern std::queue<std::vector<int>> queueUpdateLabels_right;
/* for predict */
extern std::queue< std::vector<std::vector<std::vector<int>>>> queue3DData;
//mutex
extern std::mutex mtxImg, mtxYoloLeft, mtxTMLeft, mtxTarget; // define mutex

#endif