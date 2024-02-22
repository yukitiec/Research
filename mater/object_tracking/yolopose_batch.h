#pragma once

#ifndef YOLOPOSE_BATCH_H
#define YOLOPOSE_BATCH_H

#include "stdafx.h"
#include "global_parameters.h"
#include "utility.h"

extern const int LEFT;
extern const int RIGHT;

/* roi setting */
extern const bool bool_dynamic_roi; //adopt dynamic roi
extern const bool bool_rotate_roi;
//if true
extern const float max_half_diagonal;
extern const float min_half_diagonal;

extern const int roiSize_wrist;
extern const int roiSize_elbow;
extern const int roiSize_shoulder;

extern const std::string file_yolo_left;
extern const std::string file_yolo_right;
extern std::mutex mtxRobot, mtxYolo_left, mtxYolo_right;

extern std::queue<std::array<cv::Mat1b, 2>> queueFrame;
extern std::queue<int> queueFrameIndex;

/* left */
extern std::queue<std::vector<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>>> queueYoloTracker_left;      // queue for old image for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Mat1b>>> queueYoloTemplate_left; // queue for yolo template       // queue for old image for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Rect2i>>> queueYoloRoi_left;        // queue for search roi for optical flow. vector size is [num human,6]

/* right */
extern std::queue<std::vector<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>>> queueYoloTracker_right;      // queue for old image for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Mat1b>>> queueYoloTemplate_right;  // queue for old image for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Rect2i>>> queueYoloRoi_right;        // queue for search roi for optical flow. vector size is [num human,6]

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
    const int boundary_img = 320;// (int)(yoloWidth* (yoloWidth / frameWidth));
    const cv::Size YOLOSize{ yoloWidth, yoloHeight };
    const float IoUThreshold = 0.2;
    const float ConfThreshold = 0.2;
    const float IoUThresholdIdentity = 0.25; // for maitainig consistency of tracking
    const int num_joints = 6; //number of tracked joints
    const float roi_direction_threshold = 2.5; //max gradient of neighborhood joints
    std::vector<float> default_neighbor{ (float)(std::pow(2,0.5) / 2),(float)(std::pow(2,0.5) / 2) }; //45 degree direction
    const int MIN_SEARCH = 10; //mimnimum roi
    const float min_ratio = 0.55;//minimum ratio for the max value

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

    void detect(cv::Mat1b& frame, int& frameIndex, int& counter, std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver_left, std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver_right) //, const int frameIndex, std::vector<std::vector<cv::Rect2i>>& posSaver, std::vector<std::vector<int>>& classSaver)
    {
        /* inference by YOLO
         *  Args:
         *      frame : img
         *      posSaver : storage for saving detected position
         *      queueYoloTemplate : queue for pushing detected img
         *      queueYoloBbox : queue for pushing detected roi, or if available, get candidate position,
         *      queueClassIndex : queue for pushing detected
         */

         /* preprocess img */
        torch::Tensor imgTensor;
        preprocessImg(frame, imgTensor);
        // std::cout << "frame size:" << frame.cols << "," << frame.rows << std::endl;
        // std::cout << "finish preprocess" << std::endl;

        // std::cout << imgTensor.sizes() << std::endl;
        /* inference */
        torch::Tensor preds;

        std::vector<cv::Rect2i> roiLatest;
        /* get latest roi for comparing new detection */
        /*
        if (!queueOFSearchRoi.empty())
        {
          getLatestRoi(roiLatest);
        }
        */
        /* wrap to disable grad calculation */
        {
            torch::NoGradGuard no_grad;
            preds = mdl.forward({ imgTensor }).toTensor(); // preds shape : [1,6,2100]
        }
        // std::cout << "finish inference" << std::endl;
        preds = preds.permute({ 0, 2, 1 }); // change order : (1,56,2100) -> (1,2100,56)
        // std::cout << "preds size:" << preds.sizes() << std::endl;
        // std::cout << "preds size : " << preds << std::endl;
        std::vector<torch::Tensor> detectedBoxesHuman; //(n,56)
        /*detect human */
        nonMaxSuppressionHuman(preds, detectedBoxesHuman);
        //std::cout << "detectedBboxesHuman size=" << detectedBoxesHuman.size() << std::endl;
        /* get keypoints from detectedBboxesHuman -> shoulder,elbow,wrist */
        std::vector<std::vector<std::vector<int>>> keyPoints; // vector for storing keypoints
        std::vector<int> humanPos; //whether human is in left or right
        /* if human detected, extract keypoints */
        if (!detectedBoxesHuman.empty())
        {
            //std::cout << "Human detected!" << std::endl;
            keyPointsExtractor(detectedBoxesHuman, keyPoints, humanPos, ConfThreshold);
            /*push updated data to queue*/
            push2Queue(frame, frameIndex, keyPoints, roiLatest, humanPos, posSaver_left, posSaver_right);
            // std::cout << "frame size:" << frame.cols << "," << frame.rows << std::endl;
            /* draw keypoints in the frame */
            //drawCircle(frame, keyPoints, counter);
        }
    }

    void preprocessImg(cv::Mat1b& frame, torch::Tensor& imgTensor)
    {
        // run
        cv::Mat yoloimg; // define yolo img type
        //cv::imwrite("input.jpg", yoloimg);
        cv::cvtColor(frame, yoloimg, cv::COLOR_GRAY2RGB);
        cv::resize(yoloimg, yoloimg, YOLOSize);
        //cv::imwrite("yoloimg.jpg", yoloimg);
        //std::cout << "yoloImg.height" << yoloimg.rows << ", yoloimg.width" << yoloimg.cols << std::endl;
        imgTensor = torch::from_blob(yoloimg.data, { yoloimg.rows, yoloimg.cols, 3 }, torch::kByte); // vector to tensor
        imgTensor = imgTensor.permute({ 2, 0, 1 });                                                  // Convert shape from (H,W,C) -> (C,H,W)
        imgTensor = imgTensor.toType(torch::kFloat);                                               // convert to float type
        imgTensor = imgTensor.div(255);                                                            // normalization
        imgTensor = imgTensor.unsqueeze(0);                                                        //(1,3,320,320)
        imgTensor = imgTensor.to(*device);                                                         // transport data to GPU
    }

    void nonMaxSuppressionHuman(torch::Tensor& prediction, std::vector<torch::Tensor>& detectedBoxesHuman)
    {
        /* non max suppression : remove overlapped bbox
         * Args:
         *   prediction : (1,2100,,6)
         * Return:
         *   detectedbox0,detectedboxs1 : (n,6), (m,6), number of candidate
         */

        torch::Tensor xc = prediction.select(2, 4) > ConfThreshold;                       // get dimenseion 2, and 5th element of prediction : score of ball :: xc is "True" or "False"
        torch::Tensor x = prediction.index_select(1, torch::nonzero(xc[0]).select(1, 0)); // box, x0.shape : (1,n,6) : n: number of candidates
        x = x.index_select(1, x.select(2, 4).argsort(1, true).squeeze());                 // ball : sorted in descending order
        x = x.squeeze();                                                                  //(1,n,56) -> (n,56)
        bool boolLeft = false;
        bool boolRight = false;
        if (x.size(0) != 0)
        {
            /* 1 dimension */
            if (x.dim() == 1)
            {
                // std::cout << "top defined" << std::endl;
                detectedBoxesHuman.push_back(x.cpu());
            }
            /* more than 2 dimensions */
            else
            {
                // std::cout << "top defined" << std::endl;
                if (x[0][0].item<int>() <= boundary_img)
                {
                    //std::cout << "first Human is left" << std::endl;
                    detectedBoxesHuman.push_back(x[0].cpu());
                    boolLeft = true;
                }
                else if (x[0][0].item<int>() > boundary_img)
                {
                    //std::cout << "first human is right" << std::endl;
                    detectedBoxesHuman.push_back(x[0].cpu());
                    boolRight = true;
                }
                // std::cout << "push back top data" << std::endl;
                // for every candidates
                /* if adapt many humans, validate here */
                if (x.size(0) >= 2)
                {
                    //std::cout << "nms start" << std::endl;
                    nms(x, detectedBoxesHuman,  boolLeft, boolRight); // exclude overlapped bbox : 20 milliseconds
                    //std::cout << "num finished" << std::endl;
                }
            }
        }
    }

    torch::Tensor xywh2xyxy(torch::Tensor& x)
    {
        torch::Tensor y = x.clone();
        y[0] = x[0] - x[2] / 2; // left
        y[1] = x[1] - x[3] / 2; // top
        y[2] = x[0] + x[2] / 2; // right
        y[3] = x[1] + x[3] / 2; // bottom
        return y;
    }

    void nms(torch::Tensor& x, std::vector<torch::Tensor>& detectedBoxes,  bool& boolLeft, bool& boolRight)
    {
        /* calculate IoU for excluding overlapped bboxes
         *
         * bbox1,bbox2 : [left,top,right,bottom,score0,score1]
         *
         */
        int numBoxes = x.size(0);
        torch::Tensor box;
        int counter = 0;
        // there are some overlap between two bbox
        for (int i = 1; i < numBoxes; i++)
        {
            if (boolLeft && boolRight) break;
            //detect only 1 human in each image
            if ((x[i][0].item<int>() <= boundary_img && !boolLeft) || (x[i][0].item<int>() > boundary_img && !boolRight))
            {
                box = xywh2xyxy(x[i].slice(0, 0, 4)); //(xCenter,yCenter,width,height) -> (left,top,right,bottom)

                bool addBox = true; // if save bbox as a new detection

                for (const torch::Tensor& savedBox : detectedBoxes)
                {
                    float iou = calculateIoU(box, savedBox); // calculate IoU
                    /* same bbox : already found -> not add */
                    if (iou > IoUThreshold)
                    {
                        addBox = false;
                        break; // next iteration
                    }
                }
                /* new Template */
                if (addBox)
                {
                    detectedBoxes.push_back(x[i].cpu());
                    if (x[i][0].item<int>() <= boundary_img && !boolLeft) boolLeft = true;
                    if (x[i][0].item<int>() > boundary_img && !boolRight) boolRight = true;
                }
            }
        }
    }

    float calculateIoU(const torch::Tensor& box1, const torch::Tensor& box2)
    {
        float left = std::max(box1[0].item<float>(), box2[0].item<float>());
        float top = std::max(box1[1].item<float>(), box2[1].item<float>());
        float right = std::min(box1[2].item<float>(), box2[2].item<float>());
        float bottom = std::min(box1[3].item<float>(), box2[3].item<float>());

        if (left < right && top < bottom)
        {
            float intersection = (right - left) * (bottom - top);
            float area1 = ((box1[2] - box1[0]) * (box1[3] - box1[1])).item<float>();
            float area2 = ((box2[2] - box2[0]) * (box2[3] - box2[1])).item<float>();
            float unionArea = area1 + area2 - intersection;

            return intersection / unionArea;
        }

        return 0.0f; // No overlap
    }

    void keyPointsExtractor(std::vector<torch::Tensor>& detectedBboxesHuman, std::vector<std::vector<std::vector<int>>>& keyPoints, std::vector<int>& humanPos, const int& ConfThreshold)
    {
        int numDetections = detectedBboxesHuman.size();
        bool boolLeft = false;
        /* iterate for all detections of humand */
        for (int i = 0; i < numDetections; i++)
        {
            std::vector<std::vector<int>> keyPointsTemp;
            /* iterate for 3 joints positions */
            for (int j = 5; j < 11; j++)
            {
                /* if keypoints score meet criteria */
                if (detectedBboxesHuman[i][3 * j + 7].item<float>() > ConfThreshold)
                {
                    //left
                    if ((static_cast<int>((frameWidth / yoloWidth) * detectedBboxesHuman[i][3 * j + 7 - 2].item<int>()) <= originalWidth))
                    {
                        boolLeft = true; //left person
                        keyPointsTemp.push_back({ static_cast<int>((frameWidth / yoloWidth) * detectedBboxesHuman[i][3 * j + 7 - 2].item<int>()), static_cast<int>((frameHeight / yoloHeight) * detectedBboxesHuman[i][3 * j + 7 - 1].item<int>()) }); /*(xCenter,yCenter)*/
                    }
                    //right
                    else
                    {
                        boolLeft = false;
                        keyPointsTemp.push_back({ static_cast<int>((frameWidth / yoloWidth) * detectedBboxesHuman[i][3 * j + 7 - 2].item<int>() - originalWidth), static_cast<int>((frameHeight / yoloHeight) * detectedBboxesHuman[i][3 * j + 7 - 1].item<int>()) }); /*(xCenter,yCenter)*/
                    }
                }
                else
                {
                    keyPointsTemp.push_back({ -1, -1 });
                }
            }
            //std::cout << "boolLeft=" << boolLeft << std::endl;
            keyPoints.push_back(keyPointsTemp);
            if (boolLeft) humanPos.push_back(LEFT);
            else humanPos.push_back(RIGHT);
        }
    }

    void drawCircle(cv::Mat1b& frame, std::vector<std::vector<std::vector<int>>>& ROI, int& counter)
    {
        /*number of detections */
        for (int k = 0; k < ROI.size(); k++)
        {
            /*for all joints */
            for (int i = 0; i < ROI[k].size(); i++)
            {
                if (ROI[k][i][0] != -1)
                {
                    cv::circle(frame, cv::Point(ROI[k][i][0], ROI[k][i][1]), 5, cv::Scalar(125), -1);
                }
            }
        }
        std::string save_path = std::to_string(counter) + ".jpg";
        cv::imwrite(save_path, frame);
    }

    void push2Queue(cv::Mat1b& frame, int& frameIndex, std::vector<std::vector<std::vector<int>>>& keyPoints,
        std::vector<cv::Rect2i>& roiLatest, std::vector<int>& humanPos, std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver_left, std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver_right)
    {
        /* check roi Latest
         * if tracking was successful -> update and
         * else : update roi and imgSearch and calculate features. push data to queue
         */
        bool bool_left; //left or right person
        if (!keyPoints.empty())
        {
            std::vector<std::vector<cv::Rect2i>> humanJoints_left, humanJoints_right; // for every human
            std::vector<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>> trackerHuman_left, trackerHuman_right;
            std::vector<std::vector<cv::Mat1b>> TemplateHuman_left, TemplateHuman_right;
            std::vector<std::vector<std::vector<int>>> humanJointsCenter_left, humanJointsCenter_right;
            /* for every person */
            for (int i = 0; i < keyPoints.size(); i++)
            {
                std::vector<cv::Rect2i> joints; // for every joint
                std::vector<cv::Mat1b> TemplateJoint;
                std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>> trackerJoint;
                std::vector<std::vector<int>> jointsCenter;
                //left person
                if (humanPos[i] == LEFT)
                {
                    //std::cout << "left" << std::endl;
                    /* for every joints */
                    if (bool_dynamic_roi) //dynamic roi
                    {
                        bool_left = true;
                        std::vector<float> default_distances(num_joints, max_half_diagonal);
                        std::vector<std::vector<float>>  distances(num_joints, default_distances);
                        organizeRoi(frame, frameIndex, bool_left, keyPoints[i], distances, joints,trackerJoint, TemplateJoint, jointsCenter);
                        humanJoints_left.push_back(joints);
                        if (!TemplateJoint.empty())
                        {
                            TemplateHuman_left.push_back(TemplateJoint);
                            trackerHuman_left.push_back(trackerJoint);
                        }
                        humanJointsCenter_left.push_back(jointsCenter);
                    }
                    else if (!bool_dynamic_roi) //static roi
                    {
                        for (int j = 0; j < keyPoints[i].size(); j++)
                        {
                            organize_left(frame, frameIndex, keyPoints[i][j], j, joints,trackerJoint, TemplateJoint, jointsCenter);
                        }
                        humanJoints_left.push_back(joints);
                        if (!TemplateJoint.empty())
                        {
                            TemplateHuman_left.push_back(TemplateJoint);
                            trackerHuman_left.push_back(trackerJoint);
                        }
                        humanJointsCenter_left.push_back(jointsCenter);
                    }
                }
                //right person
                else
                {
                    if (bool_dynamic_roi) //dynamic roi
                    {
                        bool_left = false;
                        std::vector<float> default_distances(num_joints, max_half_diagonal);
                        std::vector<std::vector<float>>  distances(num_joints, default_distances);
                        organizeRoi(frame, frameIndex, bool_left, keyPoints[i], distances, joints,trackerJoint, TemplateJoint, jointsCenter);
                        humanJoints_right.push_back(joints);
                        if (!TemplateJoint.empty())
                        {
                            TemplateHuman_right.push_back(TemplateJoint);
                            trackerHuman_right.push_back(trackerJoint);
                        }
                        humanJointsCenter_right.push_back(jointsCenter);
                    }
                    else if (!bool_dynamic_roi)
                    {
                        //std::cout << "right" << std::endl;
                        /* for every joints */
                        for (int j = 0; j < keyPoints[i].size(); j++)
                        {
                            organize_right(frame, frameIndex, keyPoints[i][j], j, joints, trackerJoint, TemplateJoint, jointsCenter);
                        }
                        humanJoints_right.push_back(joints);
                        if (!TemplateJoint.empty())
                        {
                            TemplateHuman_right.push_back(TemplateJoint);
                            trackerHuman_right.push_back(trackerJoint);
                        }
                        humanJointsCenter_right.push_back(jointsCenter);
                    }
                }
            }
            // push data to queue
            //std::unique_lock<std::mutex> lock(mtxYolo); // exclude other accesses
            //pop before push
            if (!queueYoloRoi_left.empty()) queueYoloRoi_left.pop();
            if (!queueYoloRoi_right.empty()) queueYoloRoi_right.pop();
            if (!queueYoloTemplate_left.empty()) queueYoloTemplate_left.pop();
            if (!queueYoloTemplate_right.empty()) queueYoloTemplate_right.pop();
            if (!queueYoloTracker_left.empty()) queueYoloTracker_left.pop();
            if (!queueYoloTracker_right.empty()) queueYoloTracker_right.pop();
            queueYoloRoi_left.push(humanJoints_left);
            queueYoloRoi_right.push(humanJoints_right);
            /*for (std::vector<cv::Rect2i>& joints : humanJoints_left)
            {
                for (cv::Rect2i& joint : joints)
                    std::cout << joint.x << joint.y << joint.width << joint.height << std::endl;
            }*/
            if (!TemplateHuman_left.empty() && !trackerHuman_left.empty())
            {
                queueYoloTemplate_left.push(TemplateHuman_left);
                queueYoloTracker_left.push(trackerHuman_left);
            }
            if (!TemplateHuman_right.empty() && !trackerHuman_right.empty())
            {
                queueYoloTemplate_right.push(TemplateHuman_right);
                queueYoloTracker_right.push(trackerHuman_right);

            }
            posSaver_left.push_back(humanJointsCenter_left);
            posSaver_right.push_back(humanJointsCenter_right);
        }
    }

    void organizeRoi(cv::Mat1b& frame, int& frameIndex, bool& bool_left, std::vector<std::vector<int>>& pos,
        std::vector<std::vector<float>>& distances, std::vector<cv::Rect2i>& joints, std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& trackerJoint,std::vector<cv::Mat1b>& imgJoint, std::vector<std::vector<int>>& jointsCenter)
    {
        //calculate each joints distances -> save into vector
        //if pos[0] < 0 -> distances is remained
        float distance;
        auto start = std::chrono::high_resolution_clock::now();
        //calculate distance between each joint
        for (int i = 0; i < pos.size() - 1; i++) //for each joint
        {
            if (pos[i][0] > 0)//keypoints found
            {
                int j = i + 1;
                while (j < pos.size()) //calculate distances for each distances
                {
                    if (pos[j][0] > 0) //keypoints found
                    {
                        distance = std::pow((std::pow((pos[i][0] - pos[j][0]), 2) + std::pow((pos[i][1] - pos[j][1]), 2)), 0.5); //calculate distance
                        distances[i][j] = distance; //save distance
                        distances[j][i] = distance;
                    }
                    j++;
                }
            }
        }
        //setting roi for each joint
        for (int i = 0; i < pos.size(); i++)
        {

            if (pos[i][0] > 0) //found
            {
                if (i == 0)//left shoulder
                {
                    std::vector<int> joints_neighbor{ 1,2 };
                    setRoi(frameIndex, frame, bool_left, distances, i, joints_neighbor, pos, joints, trackerJoint,imgJoint, jointsCenter);
                }
                else if (i == 1)//right shoulder
                {
                    std::vector<int> joints_neighbor{ 0,3 };
                    setRoi(frameIndex, frame, bool_left, distances, i, joints_neighbor, pos, joints, trackerJoint, imgJoint, jointsCenter);
                }
                else if (i == 2)//left elbow
                {
                    std::vector<int> joints_neighbor{ 0,4 };
                    setRoi(frameIndex, frame, bool_left, distances, i, joints_neighbor, pos, joints, trackerJoint, imgJoint, jointsCenter);
                }
                else if (i == 3)//right elbow
                {
                    std::vector<int> joints_neighbor{ 1,5 };
                    setRoi(frameIndex, frame, bool_left, distances, i, joints_neighbor, pos, joints, trackerJoint, imgJoint, jointsCenter);
                }
                else if (i == 4)//left wrist
                {
                    std::vector<int> joints_neighbor{ 2, 2 };
                    setRoi(frameIndex, frame, bool_left, distances, i, joints_neighbor, pos, joints, trackerJoint, imgJoint, jointsCenter);
                }
                else if (i == 5)//right wrist
                {
                    std::vector<int> joints_neighbor{ 3, 3 };
                    setRoi(frameIndex, frame, bool_left, distances, i, joints_neighbor, pos, joints, trackerJoint, imgJoint, jointsCenter);
                }
            }
            else //not found
            {
                jointsCenter.push_back({ frameIndex, -1, -1 });
                joints.emplace_back(-1, -1, -1, -1);
            }
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "!!!! time taken by setting roi=" << duration.count() << " microseconds !!!!" << std::endl;
    }

    void setRoi(int& frameIndex, cv::Mat1b& frame, bool& bool_left, std::vector<std::vector<float>>& distances, int& index_joint, std::vector<int>& compareJoints, std::vector<std::vector<int>>& pos,
        std::vector<cv::Rect2i>& joints, std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& trackerJoint, std::vector<cv::Mat1b>& imgJoint, std::vector<std::vector<int>>& jointsCenter)
    {
        float vx, vy;
        auto distance_min = std::min_element(distances[index_joint].begin(), distances[index_joint].end());
        //std::cout << "minimum distance=" << (float)(*distance_min) << std::endl;
        float half_diagonal = std::max(min_half_diagonal, ((float)(1.0 - (float)((1.0 - min_ratio) * ((float)(*distance_min) - min_half_diagonal) / (max_half_diagonal - min_half_diagonal))) * (*distance_min)));//calculate half diagonal
        //std::cout << "half diagonal=" << half_diagonal << std::endl;
        if (bool_rotate_roi) //rotate roi
        {
            if (pos[compareJoints[0]][0] > 0)//right shoulder found
            {
                vx = ((float)(pos[compareJoints[0]][0] - pos[index_joint][0])) / distances[index_joint][compareJoints[0]]; //unit direction vector
                vy = ((float)(pos[compareJoints[0]][1] - pos[index_joint][1])) / distances[index_joint][compareJoints[0]];
                //std::cout << " first choice found :: unit direction vector: vx=" << vx << ", vy=" << vy << std::endl;
                if (std::abs(vx) / roi_direction_threshold <= std::abs(vy) && roi_direction_threshold * std::abs(vx) >= std::abs(vy))//withing good region
                {
                    if (bool_left)  //left
                        defineRoi_left(frameIndex, frame, index_joint, vx, vy, half_diagonal, pos, joints, trackerJoint,imgJoint, jointsCenter);
                    else //right
                        defineRoi_right(frameIndex, frame, index_joint, vx, vy, half_diagonal, pos, joints, trackerJoint, imgJoint, jointsCenter);
                }
                else //use default vector
                {
                    vx = default_neighbor[0];
                    vy = default_neighbor[1];
                    if (bool_left)  //left
                        defineRoi_left(frameIndex, frame, index_joint, vx, vy, half_diagonal, pos, joints, trackerJoint, imgJoint, jointsCenter);
                    else //right
                        defineRoi_right(frameIndex, frame, index_joint, vx, vy, half_diagonal, pos, joints, trackerJoint, imgJoint, jointsCenter);
                }
            }
            else if (pos[compareJoints[0]][0] <= 0 && pos[compareJoints[1]][0] > 0)//right shoulder not found, left elbow found
            {
                vx = ((float)(pos[compareJoints[1]][0] - pos[index_joint][0])) / distances[index_joint][compareJoints[1]]; //unit direction vector
                vy = ((float)(pos[compareJoints[1]][1] - pos[index_joint][1])) / distances[index_joint][compareJoints[1]];
                //std::cout << " second choice found :: unit direction vector: vx=" << vx << ", vy=" << vy << std::endl;
                if (std::abs(vx) / roi_direction_threshold <= std::abs(vy) && roi_direction_threshold * std::abs(vx) >= std::abs(vy))//withing good region
                {
                    if (bool_left)  //left
                        defineRoi_left(frameIndex, frame, index_joint, vx, vy, half_diagonal, pos, joints, trackerJoint, imgJoint, jointsCenter);
                    else //right
                        defineRoi_right(frameIndex, frame, index_joint, vx, vy, half_diagonal, pos, joints, trackerJoint, imgJoint, jointsCenter);
                }
                else //uuse default vector
                {
                    vx = default_neighbor[0];
                    vy = default_neighbor[1];
                    if (bool_left)  //left
                        defineRoi_left(frameIndex, frame, index_joint, vx, vy, half_diagonal, pos, joints, trackerJoint, imgJoint, jointsCenter);
                    else //right
                        defineRoi_right(frameIndex, frame, index_joint, vx, vy, half_diagonal, pos, joints, trackerJoint, imgJoint, jointsCenter);
                }
            }
            else //no neighbors found
            {
                //std::cout << "no neightbors :"<< std::endl;
                vx = default_neighbor[0];
                vy = default_neighbor[1];
                if (bool_left)  //left
                    defineRoi_left(frameIndex, frame, index_joint, vx, vy, half_diagonal, pos, joints, trackerJoint, imgJoint, jointsCenter);
                else //right
                    defineRoi_right(frameIndex, frame, index_joint, vx, vy, half_diagonal, pos, joints, trackerJoint, imgJoint, jointsCenter);
            }
        }
        else //not rotate roi
        {
            //std::cout << "no neightbors :"<< std::endl;
            vx = default_neighbor[0];
            vy = default_neighbor[1];
            if (bool_left)  //left
                defineRoi_left(frameIndex, frame, index_joint, vx, vy, half_diagonal, pos, joints, trackerJoint, imgJoint, jointsCenter);
            else //right
                defineRoi_right(frameIndex, frame, index_joint, vx, vy, half_diagonal, pos, joints, trackerJoint, imgJoint, jointsCenter);
        }
    }

    void defineRoi_left(int& frameIndex, cv::Mat1b& frame, int& index_joint, float& vx, float& vy, float& half_diagonal, std::vector<std::vector<int>>& pos,
        std::vector<cv::Rect2i>& joints, std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& trackerJoint, std::vector<cv::Mat1b>& imgJoint, std::vector<std::vector<int>>& jointsCenter)
    {
        float x1, x2, y1, y2; //candidate points for corners of rectangle
        int left, top, right, bottom; //bbox corners
        if ((index_joint == 4 || index_joint == 5) && (vx!=default_neighbor[0] && vy!=default_neighbor[1])) //wrist
        {
            x1 = pos[index_joint][0] + half_diagonal * vx;
            y1 = pos[index_joint][1] + half_diagonal * vy;
            x2 = pos[index_joint][0] - half_diagonal * vx;
            y2 = pos[index_joint][1] - half_diagonal * vy;
        }
        else //shoulder or elbow
        {
            x1 = pos[index_joint][0] + half_diagonal * vx;
            y1 = pos[index_joint][1] + half_diagonal * vy;
            x2 = pos[index_joint][0] - half_diagonal * vx;
            y2 = pos[index_joint][1] - half_diagonal * vy;
        }
        left = std::min(std::max((int)(std::min(x1, x2)), 0), originalWidth);
        right = std::max(std::min((int)(std::max(x1, x2)), originalWidth), 0);
        top = std::min(std::max((int)(std::min(y1, y2)), 0), originalHeight);
        bottom = std::max(std::min((int)(std::max(y1, y2)), originalHeight), 0);
        //std::cout << "left=" << left << ", right=" << right << ", top=" << top << ", bottom=" << bottom << std::endl;
        if ((right - left) > MIN_SEARCH && (bottom - top) > MIN_SEARCH)
        {
            cv::Rect2i roi(left, top, right - left, bottom - top);
            jointsCenter.push_back({ frameIndex,left,right,(right-left),(bottom-top)});
            joints.push_back(roi);
            cv::Ptr<cv::mytracker::TrackerMOSSE> tracker = cv::mytracker::TrackerMOSSE::create();
            tracker->init(frame, cv::Rect2d(roi.x,roi.y,roi.width,roi.height));
            trackerJoint.push_back(tracker);
            imgJoint.push_back(frame(roi));
            //std::cout << "||||| YOLO::roi.width = " << roi.width << ", roi.height = " << roi.height << std::endl;
        }
        else
        {
            jointsCenter.push_back({ frameIndex, -1, -1,-1,-1 });
            joints.emplace_back(-1, -1, -1, -1);
        }
    }

    void defineRoi_right(int& frameIndex, cv::Mat1b& frame, int& index_joint, float& vx, float& vy, float& half_diagonal, std::vector<std::vector<int>>& pos,
        std::vector<cv::Rect2i>& joints, std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& trackerJoint, std::vector<cv::Mat1b>& imgJoint, std::vector<std::vector<int>>& jointsCenter)
    {
        float x1, x2, y1, y2; //candidate points for corners of rectangle
        int left, top, right, bottom; //bbox corners
        x1 = pos[index_joint][0] + half_diagonal * vx;
        y1 = pos[index_joint][1] + half_diagonal * vy;
        x2 = pos[index_joint][0] - half_diagonal * vx;
        y2 = pos[index_joint][1] - half_diagonal * vy;
        left = std::min(std::max((int)(std::min(x1, x2)), 0), originalWidth);
        right = std::max(std::min((int)(std::max(x1, x2)), originalWidth), 0);
        top = std::min(std::max((int)(std::min(y1, y2)), 0), originalHeight);
        bottom = std::max(std::min((int)(std::max(y1, y2)), originalHeight), 0);
        //std::cout << "left=" << left << ", right=" << right << ", top=" << top << ", bottom=" << bottom << std::endl;
        if ((right - left) > MIN_SEARCH && (bottom - top) > MIN_SEARCH)
        {
            cv::Rect2i roi(left, top, right - left, bottom - top);
            jointsCenter.push_back({ frameIndex,roi.x,roi.y,roi.width,roi.height});
            joints.push_back(roi);
            roi.x += originalWidth;
            imgJoint.push_back(frame(roi));
            cv::Ptr<cv::mytracker::TrackerMOSSE> tracker = cv::mytracker::TrackerMOSSE::create();
            tracker->init(frame, cv::Rect2d(roi.x, roi.y, roi.width, roi.height));
            trackerJoint.push_back(tracker);
            //std::stringstream fileNameStream;
            //fileNameStream << "yolo-" << frameIndex << ".jpg";
            //std::string fileName = fileNameStream.str();
            //cv::imwrite(fileName, frame(roi));
            //std::cout << "||||| YOLO::roi.width = " << roi.width << ", roi.height = " << roi.height << std::endl;
        }
        else
        {
            jointsCenter.push_back({ frameIndex, -1, -1,-1,-1 });
            joints.emplace_back(-1, -1, -1, -1);
        }
    }

    void organize_left(cv::Mat1b& frame, int& frameIndex, std::vector<int>& pos, int& index, std::vector<cv::Rect2i>& joints, std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& trackerJoint, std::vector<cv::Mat1b>& TemplateJoint, std::vector<std::vector<int>>& jointsCenter)
    {
        if (static_cast<int>(pos[0]) >= 0)
        {
            int roiWidthYolo, roiHeightYolo;
            if (index == 0 || index == 1)
            {
                roiWidthYolo = roiSize_shoulder;
                roiHeightYolo = roiSize_shoulder;
            }
            else if (index == 2 || index == 3)
            {
                roiWidthYolo = roiSize_elbow;
                roiHeightYolo = roiSize_elbow;
            }
            else if (index == 4 || index == 5)
            {
                roiWidthYolo = roiSize_wrist;
                roiHeightYolo = roiSize_wrist;
            }
            int left = std::min(std::max(static_cast<int>(pos[0] - roiWidthYolo / 2), 0), originalWidth);
            int top = std::min(std::max(static_cast<int>(pos[1] - roiHeightYolo / 2), 0), originalHeight);
            int right = std::max(std::min(static_cast<int>(pos[0] + roiWidthYolo / 2), originalWidth), 0);
            int bottom = std::max(std::min(static_cast<int>(pos[1] + roiHeightYolo / 2), originalHeight), 0);
            if ((right - left) > MIN_SEARCH && (bottom - top) > MIN_SEARCH)
            {
                cv::Rect2i roi(left, top, right - left, bottom - top);
                jointsCenter.push_back({ frameIndex, roi.x,roi.y,roi.width,roi.height });
                joints.push_back(roi);
                cv::Mat1b Template = frame(roi);
                TemplateJoint.push_back(Template);
                cv::Ptr<cv::mytracker::TrackerMOSSE> tracker = cv::mytracker::TrackerMOSSE::create();
                tracker->init(frame, cv::Rect2d(roi.x, roi.y, roi.width, roi.height));
                trackerJoint.push_back(tracker);
            }
            else
            {
                jointsCenter.push_back({ frameIndex, -1, -1,-1,-1 });
                joints.emplace_back(-1, -1, -1, -1);
            }
            
        }
        /* keypoints can't be detected */
        else
        {
            jointsCenter.push_back({ frameIndex, -1, -1,-1,-1 });
            joints.emplace_back(-1, -1, -1, -1);
        }
    }

    void organize_right(cv::Mat1b& frame, int& frameIndex, std::vector<int>& pos, int& index, std::vector<cv::Rect2i>& joints, std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& trackerJoint, std::vector<cv::Mat1b>& TemplateJoint, std::vector<std::vector<int>>& jointsCenter)
    {
        if (static_cast<int>(pos[0]) >= 0)
        {
            int roiWidthYolo, roiHeightYolo;
            if (index == 0 || index == 1)
            {
                roiWidthYolo = roiSize_shoulder;
                roiHeightYolo = roiSize_shoulder;
            }
            else if (index == 2 || index == 3)
            {
                roiWidthYolo = roiSize_elbow;
                roiHeightYolo = roiSize_elbow;
            }
            else if (index == 4 || index == 5)
            {
                roiWidthYolo = roiSize_wrist;
                roiHeightYolo = roiSize_wrist;
            }
            int left = std::min(std::max(static_cast<int>(pos[0] - roiWidthYolo / 2), 0), originalWidth);
            int top = std::min(std::max(static_cast<int>(pos[1] - roiHeightYolo / 2), 0), originalHeight);
            int right = std::max(std::min(static_cast<int>(pos[0] + roiWidthYolo / 2), originalWidth), 0);
            int bottom = std::max(std::min(static_cast<int>(pos[1] + roiHeightYolo / 2), originalHeight), 0);
            if ((right - left) > MIN_SEARCH && (bottom - top) > MIN_SEARCH)
            {
                cv::Rect2i roi(left, top, right - left, bottom - top);
                joints.push_back(roi);
                roi.x += originalWidth;
                cv::Mat1b Template = frame(roi);
                //std::stringstream fileNameStream;
                //fileNameStream << "yolo-" << frameIndex << ".jpg";
                //std::string fileName = fileNameStream.str();
                //cv::imwrite(fileName, frame(roi));
                TemplateJoint.push_back(Template);
                jointsCenter.push_back({ frameIndex, left,top,(right - left),(bottom - top) });
                cv::Ptr<cv::mytracker::TrackerMOSSE> tracker = cv::mytracker::TrackerMOSSE::create();
                tracker->init(frame, cv::Rect2d(roi.x, roi.y, roi.width, roi.height));
                trackerJoint.push_back(tracker);
            }
            else {
                jointsCenter.push_back({ frameIndex, -1, -1,-1,-1 });
                joints.emplace_back(-1, -1, -1, -1);
            }
            
        }
        /* keypoints can't be detected */
        else
        {
            jointsCenter.push_back({ frameIndex, -1, -1,-1,-1 });
            joints.emplace_back(-1, -1, -1, -1);
        }
    }
};

#endif