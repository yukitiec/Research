#pragma once


#ifndef GLOBAL_PARAMETERS_H
#define GLOBAL_PARAMETERS_H

#include "stdafx.h"
#include "mosse.h"

extern const bool boolGroundTruth = false;
//video path
extern const std::string filename_left = "switching2_0111_left.mp4";
extern const std::string filename_right = "switching2_0111_right.mp4";
// camera : constant setting
extern const int LEFT_CAMERA = 0;
extern const int RIGHT_CAMERA = 1;
extern const int FPS = 400;
// YOLO label
extern const int BALL = 0;
extern const int BOX = 1;
// tracker
extern const double threshold_mosse = 5.0;//0.57; //PSR threshold

//Kalman filter setting
extern const double INIT_X = 0.0;
extern const double INIT_Y = 0.0;
extern const double INIT_VX = 0.0;
extern const double INIT_VY = 0.0;
extern const double INIT_AX = 0.0;
extern const double INIT_AY = 9.81;
extern const double NOISE_POS = 0.1;
extern const double NOISE_VEL = 1.0;
extern const double NOISE_ACC = 1.0;
extern const double NOISE_SENSOR = 0.1;
// tracking
extern const int COUNTER_VALID = 4; //frames by official tracker
extern const int COUNTER_LOST = 4; //frames by deleting tracker
extern const float MAX_ROI_RATE = 2.0; //max change of roi
extern const float MIN_ROI_RATE = 0.5; //minimum change of roi
extern const double MIN_IOU = 0.1; //minimum IoU for identity
extern const double MAX_RMSE = 30; //max RMSE for identity

/* 3d positioning by stereo camera */
extern const int BASELINE = 280; // distance between 2 cameras
/* camera calibration result */
extern const cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 369, 0, 320, // fx: focal length in x, cx: principal point x
    0, 369, 320,                           // fy: focal length in y, cy: principal point y
    0, 0, 1                                // 1: scaling factor
    );
extern const cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 1, 1, 1, 1, 1);
/* transformation matrix from camera coordinate to robot base coordinate */
extern const std::vector<std::vector<float>> transform_cam2base{ {1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1} };
//3d objects number
extern const int numObjects = 50;

/* UR catching point */
extern const int TARGET_DEPTH = 600; // catching point is 40 cm away from camera position

/* save file setting */
extern const std::string file_yolo_bbox_left = "yolo_bbox__left.csv";
extern const std::string file_yolo_class_left = "yolo_class_left.csv";
extern const std::string file_tm_bbox_left = "tm_bbox_left.csv";
extern const std::string file_tm_class_left = "tm_class_left.csv";
extern const std::string file_seq_left = "seqData_left.csv";
extern const std::string file_kf_left = "kfData_left.csv";
extern const std::string file_yolo_bbox_right = "yolo_bboxe_right.csv";
extern const std::string file_yolo_class_right = "yolo_class_right.csv";
extern const std::string file_tm_bbox_right = "tm_bbox_right.csv";
extern const std::string file_tm_class_right = "tm_class_right.csv";
extern const std::string file_seq_right = "seqData_right.csv";
extern const std::string file_kf_right = "kfData_right.csv";
extern const std::string file_3d = "triangulation.csv";
extern const std::string file_target = "target.csv";

// queue definitions
std::queue<std::array<cv::Mat1b, 2>> queueFrame; // queue for frame
std::queue<int> queueFrameIndex;  // queue for frame index

//Yolo signals
std::queue<bool> queueYolo_tracker2seq_left, queueYolo_tracker2seq_right;
std::queue<bool> queueYolo_seq2tri_left, queueYolo_seq2tri_right;
std::queue<bool> queue_tri2predict;

//mosse
std::queue<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>> queueTrackerYolo_left;
std::queue<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>> queueTrackerYolo_right;
std::queue<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>> queueTrackerMOSSE_left;
std::queue<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>> queueTrackerMOSSE_right;

// left cam
std::queue<std::vector<cv::Mat1b>> queueYoloTemplateLeft; // queue for yolo template : for real cv::Mat type
std::queue<std::vector<cv::Rect2d>> queueYoloBboxLeft;    // queue for yolo bbox
std::queue<std::vector<cv::Mat1b>> queueTMTemplateLeft;   // queue for templateMatching template img : for real cv::Mat
std::queue<std::vector<cv::Rect2d>> queueTMBboxLeft;      // queue for templateMatching bbox
std::queue<std::vector<int>> queueYoloClassIndexLeft;     // queue for class index
std::queue<std::vector<int>> queueTMClassIndexLeft;       // queue for class index
std::queue<std::vector<bool>> queueTMScalesLeft;          // queue for search area scale
std::queue<std::vector<std::vector<int>>> queueMoveLeft; //queue for saving previous move
std::queue<bool> queueLabelUpdateLeft;                    // for updating labels of sequence data
//std::queue<int> queueNumLabels;                           // current labels number -> for maintaining label number consistency
std::queue<bool> queueStartYolo_left; //if new Yolo inference can start
std::queue<bool> queueStartYolo_right; //if new Yolo inference can start

// right cam
std::queue<std::vector<cv::Mat1b>> queueYoloTemplateRight; // queue for yolo template : for real cv::Mat type
std::queue<std::vector<cv::Rect2d>> queueYoloBboxRight;    // queue for yolo bbox
std::queue<std::vector<cv::Mat1b>> queueTMTemplateRight;   // queue for templateMatching template img : for real cv::Mat
std::queue<std::vector<cv::Rect2d>> queueTMBboxRight;      // queue for TM bbox
std::queue<std::vector<int>> queueYoloClassIndexRight;     // queue for class index
std::queue<std::vector<int>> queueTMClassIndexRight;       // queue for class index
std::queue<std::vector<bool>> queueTMScalesRight;          // queue for search area scale
std::queue<std::vector<std::vector<int>>> queueMoveRight; //queue for saving previous move
std::queue<bool> queueLabelUpdateRight;                    // for updating labels of sequence data

//from tm to yolo
std::queue<std::vector<cv::Rect2d>> queueTM2YoloBboxLeft;      // queue for templateMatching bbox
std::queue<std::vector<int>> queueTM2YoloClassIndexLeft;     // queue for class index
std::queue<std::vector<cv::Rect2d>> queueTM2YoloBboxRight;      // queue for templateMatching bbox
std::queue<std::vector<int>> queueTM2YoloClassIndexRight;     // queue for class index

//from seq : kalman prediction
std::queue<std::vector<std::vector<double>>> queueKfPredictLeft; //{label, left,top,width,height}
std::queue<std::vector<std::vector<double>>> queueKfPredictRight;

// sequential data
std::vector<std::vector<std::vector<double>>> seqData_left, seqData_right; //storage for sequential data to share with triangulation.h
std::queue<int> queueTargetFrameIndex_left;                      // TM estimation frame
std::queue<int> queueTargetFrameIndex_right;                      // TM estimation frame
std::queue<std::vector<cv::Rect2d>> queueTargetBboxesLeft;  // bboxes from template matching for predict objects' trajectory
std::queue<std::vector<cv::Rect2d>> queueTargetBboxesRight; // bboxes from template matching for predict objects' trajectory
std::queue<std::vector<int>> queueTargetClassIndexesLeft;   // class from template matching for maintain consistency
std::queue<std::vector<int>> queueTargetClassIndexesRight;  // class from template matching for maintain consistency

//matching
std::queue<std::vector<int>> queueUpdateLabels_left;
std::queue<std::vector<int>> queueUpdateLabels_right;
/* for predict */
std::queue< std::vector<std::vector<std::vector<int>>>> queue3DData;
//mutex
std::mutex mtxImg, mtxYoloLeft, mtxTMLeft, mtxTarget; // define mutex


#endif