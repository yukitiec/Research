#include "stdafx.h"
#include "global_parameters.h"
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
extern const std::string file_yolo_bbox_left = "yolo_bbox_left.csv";
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

//queue data
//image
std::queue<std::array<cv::Mat1b, 2>> queueFrame; // queue for frame
std::queue<int> queueFrameIndex;  // queue for frame index
//start signal
std::queue<bool> q_startTracker;
//yolo to tracker
std::queue<Yolo2tracker> q_yolo2tracker_left, q_yolo2tracker_right;
//tracker to yolo
std::queue<Tracker2yolo> q_tracker2yolo_left, q_tracker2yolo_right;
//tracker2tracker
std::queue<Tracker2tracker> q_tracker2tracker_left, q_tracker2tracker_right;
//tracker2seq
std::queue<Tracker2seq> q_tracker2seq_left, q_tracker2seq_right;
//seq2tracker
std::queue<std::vector<std::vector<double>>> q_seq2tracker_left, q_seq2tracker_right;

// sequential data
std::vector<std::vector<std::vector<double>>> seqData_left, seqData_right; //storage for sequential data to share with triangulation.h

//matching
std::queue<std::vector<int>> queueUpdateLabels_left;
std::queue<std::vector<int>> queueUpdateLabels_right;
/* for predict */
std::queue< std::vector<std::vector<std::vector<int>>>> queue3DData;
//mutex
std::mutex mtxImg, mtxYoloLeft, mtxTMLeft, mtxTarget; // define mutex