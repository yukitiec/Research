#include "stdafx.h"
#include "global_parameters.h"

/* constant valude definition */
extern const std::string filename_left = "humanMotion_0119_left.mp4";
extern const std::string filename_right = "humanMotion_0119_right.mp4";
extern const int LEFT = 0;
extern const int RIGHT = 1;
extern const bool save = true;
extern const bool boolSparse = false;
extern const bool boolGray = true;
extern const bool boolBatch = true; //if yolo inference is run in concatenated img
extern const std::string methodDenseOpticalFlow = "farneback"; //"lucasKanade_dense","rlof"
extern const int dense_vel_method = 5; //0: average, 1:second largest , 2 : median, 3 : third-quarter, 4 : first-quarter, 5:4 region-based : most important direction adopted
extern const float qualityCorner = 0.01;
/* roi setting */
extern const bool bool_dynamic_roi = true; //adopt dynamic roi
extern const bool bool_rotate_roi = true;
//if true
extern const float max_half_diagonal = 60 * std::pow(2, 0.5);//70
extern const float min_half_diagonal = 25 * std::pow(2, 0.5);//15
//if false : static roi
extern const int roiWidthOF = 60;
extern const int roiHeightOF = 60;
extern const int roiWidthYolo = 60;
extern const int roiHeightYolo = 60;
extern const float MoveThreshold = 0.0; //cancell background
extern const float epsironMove = 0.05;//half range of back ground effect:: a-epsironMove<=flow<=a+epsironMove
/* dense optical flow skip rate */
extern const int skipPixel = 1;
extern const float DIF_THRESHOLD = 1.0; //threshold for adapting yolo detection's roi
extern const float MIN_MOVE = 1.0; //minimum opticalflow movement : square value
extern const float MAX_MOVE = 30.0;
/*if exchange template of Yolo */
extern const bool boolChange = true;
/* save date */
extern const std::string file_yolo_left = "yolo_left.csv";
extern const std::string file_yolo_right = "yolo_right.csv";
extern const std::string file_of_left = "opticalflow_left.csv";
extern const std::string file_of_right = "opticalflow_right.csv";
extern const std::string file_3d = "triangulation.csv";

/* 3D triangulation */
extern const int BASELINE = 208; // distance between 2 cameras
// std::vector<std::vector<float>> cameraMatrix{ {179,0,160},{0,179,160},{0,0,1} }; //camera matrix from camera calibration

/* revise here based on camera calibration */
extern const cv::Mat cameraMatrix = (cv::Mat_<float>(3, 3) << 297.0, 0, 151.5, // fx: focal length in x, cx: principal point x
    0, 297.5, 149.0,                           // fy: focal length in y, cy: principal point y
    0, 0, 1                                // 1: scaling factor
    );
extern const cv::Mat distCoeffs = (cv::Mat_<float>(1, 5) << -0.00896, -0.215, 0.00036, 0.0043, 0.391);
/* transformation matrix from camera coordinate to robot base coordinate */
extern const std::vector<std::vector<float>> transform_cam2base{ {1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1} };

std::mutex mtxRobot;
/* queueu definition */
/* frame queue */
std::queue<std::array<cv::Mat1b, 2>> queueFrame;
std::queue<int> queueFrameIndex;
/* yolo and optical flow */
/*3D position*/
std::queue<std::vector<std::vector<std::vector<int>>>> queueTriangulation_left;
std::queue<std::vector<std::vector<std::vector<int>>>> queueTriangulation_right;
/* from joints to robot control */
std::queue<std::vector<std::vector<std::vector<int>>>> queueJointsPositions;
/* notify danger */
std::queue<bool> queueDanger;

//queue
std::queue<Yolo2optflow> q_yolo2optflow_left, q_yolo2optflow_right;
std::queue<Optflow2optflow> q_optflow2optflow_left, q_optflow2optflow_right;