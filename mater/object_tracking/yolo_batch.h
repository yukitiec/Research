#pragma once

#ifndef YOLO_BATCH_H
#define YOLO_BATCH_H

#include "stdafx.h"
#include "global_parameters.h"
#include "mosse.h"

extern const bool boolGroundTruth;
// queue definition
extern std::queue<std::array<cv::Mat1b, 2>> queueFrame; // queue for frame
extern std::queue<int> queueFrameIndex;  // queue for frame index

//mosse
extern std::queue<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>> queueTrackerYolo_left;
extern std::queue<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>> queueTrackerYolo_right;

// left cam
extern std::queue<std::vector<cv::Mat1b>> queueYoloTemplateLeft; // queue for yolo template : for real cv::Mat type
extern std::queue<std::vector<cv::Rect2d>> queueYoloBboxLeft;    // queue for yolo bbox
extern std::queue<std::vector<cv::Mat1b>> queueTMTemplateLeft;   // queue for templateMatching template img : for real cv::Mat
extern std::queue<std::vector<cv::Rect2d>> queueTMBboxLeft;      // queue for templateMatching bbox
extern std::queue<std::vector<int>> queueYoloClassIndexLeft;     // queue for class index
extern std::queue<std::vector<int>> queueTMClassIndexLeft;       // queue for class index
extern std::queue<std::vector<bool>> queueTMScalesLeft;          // queue for search area scale
extern std::queue<bool> queueLabelUpdateLeft;                    // for updating labels of sequence data
//std::queue<int> queueNumLabels;                           // current labels number -> for maintaining label number consistency
extern std::queue<bool> queueStartYolo_left; //if new Yolo inference can start
extern std::queue<bool> queueStartYolo_right; //if new Yolo inference can start

// right cam
extern std::queue<std::vector<cv::Mat1b>> queueYoloTemplateRight; // queue for yolo template : for real cv::Mat type
extern std::queue<std::vector<cv::Rect2d>> queueYoloBboxRight;    // queue for yolo bbox
extern std::queue<std::vector<cv::Mat1b>> queueTMTemplateRight;   // queue for templateMatching template img : for real cv::Mat
extern std::queue<std::vector<cv::Rect2d>> queueTMBboxRight;      // queue for TM bbox
extern std::queue<std::vector<int>> queueYoloClassIndexRight;     // queue for class index
extern std::queue<std::vector<int>> queueTMClassIndexRight;       // queue for class index
extern std::queue<std::vector<bool>> queueTMScalesRight;          // queue for search area scale
extern std::queue<bool> queueLabelUpdateRight;                    // for updating labels of sequence data


//from tm to yolo
extern std::queue<std::vector<cv::Rect2d>> queueTM2YoloBboxLeft;      // queue for templateMatching bbox
extern std::queue<std::vector<int>> queueTM2YoloClassIndexLeft;     // queue for class index
extern std::queue<std::vector<cv::Rect2d>> queueTM2YoloBboxRight;      // queue for templateMatching bbox
extern std::queue<std::vector<int>> queueTM2YoloClassIndexRight;     // queue for class index
/*
// 3D positioning ~ trajectory prediction
extern std::queue<int> queueTargetFrameIndex_left;                      // TM estimation frame
extern std::queue<int> queueTargetFrameIndex_right;
extern std::queue<std::vector<cv::Rect2d>> queueTargetBboxesLeft;  // bboxes from template matching for predict objects' trajectory
extern std::queue<std::vector<cv::Rect2d>> queueTargetBboxesRight; // bboxes from template matching for predict objects' trajectory
extern std::queue<std::vector<int>> queueTargetClassIndexesLeft;   // class from template matching for maintain consistency
extern std::queue<std::vector<int>> queueTargetClassIndexesRight;  // class from template matching for maintain consistency
*/

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
    const float IoUThreshold = 0.4; //throwing :: 0.25
    const float ConfThreshold = 0.4;
    const float IoUThresholdIdentity = 0.3; // for maitainig consistency of tracking
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

    void detect(cv::Mat1b& frame, const int frameIndex, std::vector<std::vector<cv::Rect2d>>& posSaver_left, std::vector<std::vector<cv::Rect2d>>& posSaver_right,
        std::vector<std::vector<int>>& classSaver_left, std::vector<std::vector<int>>& classSaver_right,
        std::vector<int>& detectedFrame_left, std::vector<int>& detectedFrame_right, std::vector<int>& detectedFrameClass_left, std::vector<int>& detectedFrameClass_right, int counterIteration)
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
        //std::cout << "finish preprocess" << std::endl;
        /* get latest data */
        std::vector<cv::Rect2d> bboxesCandidateTMLeft, bboxesCandidateTMRight; // for limiting detection area
        std::vector<int> classIndexesTMLeft, classIndexesTMRight;
        torch::Tensor preds;
        //auto start_inf = std::chrono::high_resolution_clock::now();
        if (!queueTM2YoloClassIndexLeft.empty() || !queueTM2YoloClassIndexRight.empty() )//|| counterIteration >= 3)
        {
            std::thread threadDataGetting(&YOLODetect_batch::getLatestData, this, std::ref(bboxesCandidateTMLeft), std::ref(bboxesCandidateTMRight), std::ref(classIndexesTMLeft), std::ref(classIndexesTMRight)); // get latest data
            //getLatestData(bboxesCandidateTMLeft, bboxesCandidateTMRight, classIndexesTMLeft, classIndexesTMRight);
            //std::cout << imgTensor.sizes() << std::endl;
            /* inference */
            /* wrap to disable grad calculation */
            {
                torch::NoGradGuard no_grad;
                preds = mdl.forward({ imgTensor }).toTensor(); // preds shape : [1,6,2100]
            }
            threadDataGetting.join();
        }
        //std::cout << imgTensor.sizes() << std::endl;
        else
        {
            // auto start = std::chrono::high_resolution_clock::now();
            /* wrap to disable grad calculation */
            {
                torch::NoGradGuard no_grad;
                preds = mdl.forward({ imgTensor }).toTensor(); // preds shape : [1,6,2100]
            }
        }
        //auto stop_inf = std::chrono::high_resolution_clock::now();
        //auto duration_inf = std::chrono::duration_cast<std::chrono::milliseconds>(stop_inf - start_inf);
        //std::cout << "-- Time taken by Yolo inference: " << duration_inf.count() << " milliseconds" << std::endl;
        //std::cout << "forward" << std::endl;
        /* postProcess */
        //auto start = std::chrono::high_resolution_clock::now();
        preds = preds.permute({ 0, 2, 1 }); // change order : (1,6,2100) -> (1,2100,6)
        std::vector<torch::Tensor> detectedBoxes0, detectedBoxes1; //(n,6),(m,6) :: including both left and right objects
        non_max_suppression2(preds, detectedBoxes0, detectedBoxes1);
        //auto stop = std::chrono::high_resolution_clock::now();
        //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        //std::cout << "--  Time taken by NMS: " << duration.count() << " milliseconds" << std::endl;
        //std::cout << "BBOX for Ball : " << detectedBoxes0.size() << " BBOX for BOX : " << detectedBoxes1.size() << std::endl;
        std::vector<cv::Rect2d> existedRoi_left, existedRoi_right, newRoi_left, newRoi_right;
        std::vector<int> existedClass_left, existedClass_right, newClass_left, newClass_right;
        /* Roi Setting : take care of dealing with TM data */
        /* ROI and class index management */
        //auto start_matching = std::chrono::high_resolution_clock::now();
        roiSetting(detectedBoxes0, existedRoi_left, existedClass_left, newRoi_left, newClass_left, existedRoi_right, existedClass_right, newRoi_right, newClass_right, BALL,
            bboxesCandidateTMLeft, classIndexesTMLeft, bboxesCandidateTMRight, classIndexesTMRight); //separate detection into left and right
        //std::cout << "roiSetting" << std::endl;
        /*if (!existedClass.empty())
        {
            std::cout << "existed class after roisetting of Ball:" << std::endl;
            for (const int& classIndex : existedClass)
            {
                std::cout << classIndex << " ";
            }
            std::cout << std::endl;
        }
        */
        roiSetting(detectedBoxes1, existedRoi_left, existedClass_left, newRoi_left, newClass_left, existedRoi_right, existedClass_right, newRoi_right, newClass_right, BOX,
            bboxesCandidateTMLeft, classIndexesTMLeft, bboxesCandidateTMRight, classIndexesTMRight);
        //auto stop_matching = std::chrono::high_resolution_clock::now();
        //auto duration_matching = std::chrono::duration_cast<std::chrono::milliseconds>(stop_matching - start_matching);
        //std::cout << "--   Time taken by Matching in Yolo : " << duration_matching.count() << " milliseconds" << std::endl;
        /* in Ball roisetting update all classIndexesTMLeft to existedClass, so here adapt existedClass as a reference class */
        /*if (!existedClass.empty())
        {
            std::cout << "existedClass after roiSetting of Box:" << std::endl;
            for (const int& classIndex : existedClass)
            {
                std::cout << classIndex << " ";
            }
            std::cout << std::endl;
        }*/
        /* push and save data */
        //auto start_push = std::chrono::high_resolution_clock::now();
        push2Queue_left(existedRoi_left, newRoi_left, existedClass_left, newClass_left, frame, posSaver_left, classSaver_left, frameIndex, detectedFrame_left, detectedFrameClass_left,
            queueYoloClassIndexLeft, queueYoloBboxLeft, queueYoloTemplateLeft, queueTrackerYolo_left);
        //std::cout << "push data to left" << std::endl;
        push2Queue_right(existedRoi_right, newRoi_right, existedClass_right, newClass_right, frame, posSaver_right, classSaver_right, frameIndex, detectedFrame_right, detectedFrameClass_right,
            queueYoloClassIndexRight, queueYoloBboxRight, queueYoloTemplateRight, queueTrackerYolo_right);
        //auto stop_push = std::chrono::high_resolution_clock::now();
        //auto duration_push = std::chrono::duration_cast<std::chrono::milliseconds>(stop_matching - start_matching);
        //std::cout << "--   Time taken by Pushing in Yolo : " << duration_push.count() << " milliseconds" << std::endl;
        //std::cout << "push data to right" << std::endl;
    }

    void preprocessImg(cv::Mat1b& frame, torch::Tensor& imgTensor)
    {
        // run
        //std::cout << frame.size() << std::endl;
        cv::Mat yoloimg; // define yolo img type
        cv::resize(frame, yoloimg, YOLOSize);
        cv::cvtColor(yoloimg, yoloimg, cv::COLOR_GRAY2RGB);
        imgTensor = torch::from_blob(yoloimg.data, { yoloimg.rows, yoloimg.cols, 3 }, torch::kByte); // vector to tensor
        imgTensor = imgTensor.permute({ 2, 0, 1 });                                                  // Convert shape from (H,W,C) -> (C,H,W)
        imgTensor = imgTensor.toType(torch::kFloat);                                               // convert to float type
        imgTensor = imgTensor.div(255);                                                            // normalization
        imgTensor = imgTensor.unsqueeze(0);                                                        // expand dims for Convolutional layer (height,width,1)
        imgTensor = imgTensor.to(*device);                                                         // transport data to GPU
    }

    void getLatestYolotData(std::vector<cv::Rect2d>& bboxes_left, std::vector<cv::Rect2d>& bboxes_right, std::vector<int>& classes_left, std::vector<int>& classes_right)
    {
        if (!queueYoloClassIndexLeft.empty())
        {
            classes_left = queueYoloClassIndexLeft.front();
            if (!queueYoloBboxLeft.empty())
            {
                bboxes_left = queueYoloBboxLeft.front(); // get new yolodata : {{x,y.width,height},...}
                /*std::cout << "left bboxes" << std::endl;
                for (cv::Rect2d& roi : bboxes_left)
                    std::cout << "left=" << roi.x << ", top=" << roi.y << ", width=" << roi.width << ", height=" << roi.height << std::endl;
                */
            }

            /* for debug */
            /*std::cout << ":: Left :: latest data " << std::endl;
            for (const int& label : classes_left)
                std::cout << label << " ";
            std::cout << std::endl;
            */

        }
        if (!queueYoloClassIndexRight.empty())
        {
            classes_right = queueYoloClassIndexRight.front();
            if (!queueYoloBboxRight.empty())
            {
                bboxes_right = queueYoloBboxRight.front(); // get new yolodata : {{x,y.width,height},...}
                /*std::cout << "left bboxes" << std::endl;
                for (cv::Rect2d& roi : bboxes_left)
                    std::cout << "left=" << roi.x << ", top=" << roi.y << ", width=" << roi.width << ", height=" << roi.height << std::endl;
                */
            }
            /* for debug */
            /*std::cout << ":: Right :: latest data " << std::endl;
            for (const int& label : classes_right)
                std::cout << label << " ";
            std::cout << std::endl;
            */

        }
    }

    void getLatestData(std::vector<cv::Rect2d>& bboxes_left, std::vector<cv::Rect2d>& bboxes_right, std::vector<int>& classes_left, std::vector<int>& classes_right)
    {
        // std::unique_lock<std::mutex> lock(mtxTMLeft); // Lock the mutex
        /* still didn't synchronize -> wait for next data */
        /* getting iteratively until TM class labels are synchronized with Yolo data */
        bool boolLeft = false;
        bool boolRight = false;
        while (true) // >(greater than for first exchange between Yolo and TM)
        {
            if (boolLeft && boolRight)
            {
                std::cout << "############## START YOLO INFERENCE ###############" << std::endl;
                break;
            }
            if (!queueStartYolo_left.empty())
            {
                //bool start = queueStartYolo_left.front();s
                std::cout << " ############## GET LEFT INFO #################" << std::endl;
                queueStartYolo_left.pop();
                boolLeft = true;
            }
            if (!queueStartYolo_right.empty())
            {
                std::cout << " ############## GET RIGHT INFO #################" << std::endl;
                queueStartYolo_right.pop();
                boolRight = true;
            }
        }

        // std::cout << "Left Img : Yolo bbox available from TM " << std::endl;
        if (!queueTM2YoloClassIndexLeft.empty())
        {
            classes_left = queueTM2YoloClassIndexLeft.front();
            if (!queueTM2YoloBboxLeft.empty()) bboxes_left = queueTM2YoloBboxLeft.front(); // get new yolodata : {{x,y.width,height},...}
            /* for debug */
            //std::cout << ":: Left :: latest data " << std::endl;
            //for (const int& label : classes_left)
            //    std::cout << label << " ";
            //std::cout << std::endl;
            if (!queueTM2YoloClassIndexLeft.empty()) queueTM2YoloClassIndexLeft.pop();
            if (!queueTM2YoloBboxLeft.empty()) queueTM2YoloBboxLeft.pop();

        }
        if (!queueTM2YoloClassIndexRight.empty())
        {
            classes_right = queueTM2YoloClassIndexRight.front();
            if (!queueTM2YoloBboxRight.empty()) bboxes_right = queueTM2YoloBboxRight.front(); // get new yolodata : {{x,y.width,height},...}
            /* for debug */
            //std::cout << ":: Right :: latest data " << std::endl;
            //for (const int& label : classes_right)
            //    std::cout << label << " ";
            //std::cout << std::endl;
            if (!queueTM2YoloClassIndexRight.empty()) queueTM2YoloClassIndexRight.pop();
            if (!queueTM2YoloBboxRight.empty()) queueTM2YoloBboxRight.pop();
        }
        //else
        //std::cout << "Right :: no data" << std::endl;
    }

    void non_max_suppression2(torch::Tensor& prediction, std::vector<torch::Tensor>& detectedBoxes0, std::vector<torch::Tensor>& detectedBoxes1)
    {
        /* non max suppression : remove overlapped bbox
         * Args:
         *   prediction : (1,2100,,6)
         * Return:
         *   detectedbox0,detectedboxs1 : (n,6), (m,6), number of candidate
         */

        torch::Tensor xc0 = prediction.select(2, 4) > ConfThreshold; // get dimenseion 2, and 5th element of prediction : score of ball :: xc is "True" or "False"
        torch::Tensor xc1 = prediction.select(2, 5) > ConfThreshold; // get dimenseion 2, and 5th element of prediction : score of ball :: xc is "True" or "False"

        torch::Tensor x0 = prediction.index_select(1, torch::nonzero(xc0[0]).select(1, 0)); // box, x0.shape : (1,n,6) : n: number of candidates
        torch::Tensor x1 = prediction.index_select(1, torch::nonzero(xc1[0]).select(1, 0)); // ball x1.shape : (1,m,6) : m: number of candidates

        x0 = x0.index_select(1, x0.select(2, 4).argsort(1, true).squeeze()); // ball : sorted in descending order
        x1 = x1.index_select(1, x1.select(2, 5).argsort(1, true).squeeze()); // box : sorted in descending order

        x0 = x0.squeeze(); //(1,n,6) -> (n,6)
        x1 = x1.squeeze(); //(1,m,6) -> (m,6)
        // std::cout << "sort detected data" << std::endl;
        // ball : non-max suppression
        if (x0.size(0) != 0 && x1.size(0) != 0)
        {
            std::thread thread_ball(&YOLODetect_batch::non_max_suppression_unit, this, std::ref(x0), std::ref(detectedBoxes0));
            non_max_suppression_unit(x1, detectedBoxes1);
            thread_ball.join();
        }
        else if (x0.size(0) != 0 && x1.size(0) == 0)
            non_max_suppression_unit(x0, detectedBoxes0);
        else if (x0.size(0) == 0 && x1.size(0) != 0)
            non_max_suppression_unit(x1, detectedBoxes1);
    }

    void non_max_suppression_unit(torch::Tensor& x, std::vector<torch::Tensor>& detectedBoxes)
    {
        /* 1 dimension */
        if (x.dim() == 1)
        {
            torch::Tensor bboxTop = xywh2xyxy(x.slice(0, 0, 4));
            // std::cout << "top defined" << std::endl;
            detectedBoxes.push_back(bboxTop.cpu());
        }
        /* 2 dimension */
        else
        {
            torch::Tensor bboxTop = xywh2xyxy(x[0].slice(0, 0, 4));
            // std::cout << "top defined" << std::endl;
            detectedBoxes.push_back(bboxTop.cpu());
            // std::cout << "push back top data" << std::endl;
            // for every candidates
            if (x.size(0) >= 2)
            {
                // std::cout << "nms start" << std::endl;
                nms(x, detectedBoxes); // exclude overlapped bbox : 20 milliseconds
                // std::cout << "num finished" << std::endl;
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

    void nms(torch::Tensor& x, std::vector<torch::Tensor>& detectedBoxes)
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
            box = xywh2xyxy(x[i].slice(0, 0, 4)); //(xCenter,yCenter,width,height) -> (left,top,right,bottom)

            bool addBox = true; // if save bbox as a new detection

            for (const torch::Tensor& savedBox : detectedBoxes)
            {
                float iou = calculateIoU(box, savedBox); // calculate IoU
                /* same bbox : already found -> nod add */
                if (iou > IoUThreshold)
                {
                    addBox = false;
                    break; // next iteration
                }
            }
            /* new tracker */
            if (addBox)
            {
                detectedBoxes.push_back(box.cpu());
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

    void roiSetting(std::vector<torch::Tensor>& detectedBoxes, std::vector<cv::Rect2d>& existedRoi_left, std::vector<int>& existedClass_left, std::vector<cv::Rect2d>& newRoi_left, std::vector<int>& newClass_left,
        std::vector<cv::Rect2d>& existedRoi_right, std::vector<int>& existedClass_right, std::vector<cv::Rect2d>& newRoi_right, std::vector<int>& newClass_right,
        int candidateIndex,
        std::vector<cv::Rect2d>& bboxesCandidate_left, std::vector<int>& classIndexesTM_left, std::vector<cv::Rect2d>& bboxesCandidate_right, std::vector<int>& classIndexesTM_right)
    {
        /*
         * Get current data before YOLO inference started.
         * First : Compare YOLO detection and TM detection
         * Second : if match : return new templates in the same order with TM
         * Third : if not match : adapt as a new templates and add after TM data
         * Fourth : return all class indexes including -1 (not tracked one) for maintainig data consistency
         */
         // std::cout << "bboxesYolo size=" << detectedBoxes.size() << std::endl;
         /* detected by Yolo */
        if (!detectedBoxes.empty())
        {
            //std::cout << "yolo detection exists" << std::endl;
            /* some trackers exist */
            if (!classIndexesTM_left.empty() || !classIndexesTM_right.empty())
            {
                /* constant setting */
                std::vector<cv::Rect2d> bboxesYolo_left, bboxesYolo_right; // for storing cv::Rect2d
                /* start comparison Yolo and TM data -> search for existed tracker */
                comparisonTMYolo(detectedBoxes, candidateIndex, bboxesYolo_left, bboxesYolo_right, classIndexesTM_left, classIndexesTM_right, bboxesCandidate_left, bboxesCandidate_right,
                    existedRoi_left, existedClass_left, existedRoi_right, existedClass_right);
                /* deal with new trackers */
                //left
                newDetection(candidateIndex, bboxesYolo_left, newRoi_left, newClass_left);
                //right
                newDetection(candidateIndex, bboxesYolo_right, newRoi_right, newClass_right);
            }
            /* No TM tracker exist */
            else
            {
                //std::cout << "No TM tracker exist " << std::endl;
                int numBboxes = detectedBoxes.size(); // num of detection
                int left, top, right, bottom;         // score0 : ball , score1 : box
                cv::Rect2d roi;

                /* convert torch::Tensor to cv::Rect2d */
                std::vector<cv::Rect2d> bboxesYolo_left, bboxesYolo_right;
                bboxesYolo_left.reserve(25);
                bboxesYolo_right.reserve(25);
                for (int i = 0; i < numBboxes; ++i)
                {
                    float expandrate[2] = { static_cast<float>(frameWidth) / static_cast<float>(yoloWidth), static_cast<float>(frameHeight) / static_cast<float>(yoloHeight) }; // resize bbox to fit original img size
                    // std::cout << "expandRate :" << expandrate[0] << "," << expandrate[1] << std::endl;
                    left = static_cast<int>(detectedBoxes[i][0].item().toFloat() * expandrate[0]);
                    top = static_cast<int>(detectedBoxes[i][1].item().toFloat() * expandrate[1]);
                    right = static_cast<int>(detectedBoxes[i][2].item().toFloat() * expandrate[0]);
                    bottom = static_cast<int>(detectedBoxes[i][3].item().toFloat() * expandrate[1]);
                    //left
                    if (left <= originalWidth)
                    {
                        newRoi_left.emplace_back(left, top, (right - left), (bottom - top));
                        newClass_left.push_back(candidateIndex);
                    }
                    //right
                    else
                    {
                        newRoi_right.emplace_back(left - originalWidth, top, (right - left), (bottom - top));
                        newClass_right.push_back(candidateIndex);
                    }
                }
            }
        }
        /* No object detected in Yolo -> return -1 class label */
        else
        {
            noYoloDetect(candidateIndex, bboxesCandidate_left, classIndexesTM_left, existedClass_left);
            noYoloDetect(candidateIndex, bboxesCandidate_right, classIndexesTM_right, existedClass_right);
        }
    }

    float calculateIoU_Rect2d(const cv::Rect2d& box1, const cv::Rect2d& box2)
    {
        float left = std::max(box1.x, box2.x);
        float top = std::max(box1.y, box2.y);
        float right = std::min((box1.x + box1.width), (box2.x + box2.width));
        float bottom = std::min((box1.y + box1.height), (box2.y + box2.height));

        if (left < right && top < bottom)
        {
            float intersection = (right - left) * (bottom - top);
            float area1 = box1.width * box1.height;
            float area2 = box2.width * box2.height;
            float unionArea = area1 + area2 - intersection;

            return intersection / unionArea;
        }

        return 0.0f; // No overlap
    }

    void comparisonTMYolo(std::vector<torch::Tensor>& detectedBoxes, int& candidateIndex, std::vector<cv::Rect2d>& bboxesYolo_left, std::vector<cv::Rect2d>& bboxesYolo_right,
        std::vector<int>& classIndexesTM_left, std::vector<int>& classIndexesTM_right,
        std::vector<cv::Rect2d>& bboxesCandidate_left, std::vector<cv::Rect2d>& bboxesCandidate_right,
        std::vector<cv::Rect2d>& existedRoi_left, std::vector<int>& existedClass_left, std::vector<cv::Rect2d>& existedRoi_right, std::vector<int>& existedClass_right)
    {
        /* constant setting */
        int numBboxes = detectedBoxes.size(); // num of detection
        int left, top, right, bottom;         // score0 : ball , score1 : box
        cv::Rect2d roi;                       // for updated Roi
        bool boolCurrentPosition = false;     // if current position is available

        /* convert torch::Tensor to cv::Rect2d */
        /* iterate for numBboxes */
        for (int i = 0; i < numBboxes; ++i)
        {
            float expandrate[2] = { static_cast<float>(frameWidth) / static_cast<float>(yoloWidth), static_cast<float>(frameHeight) / static_cast<float>(yoloHeight) }; // resize bbox to fit original img size
            //std::cout << "expandRate :x-axis " << expandrate[0] << ", y-axis " << expandrate[1] << std::endl;
            left = static_cast<int>(detectedBoxes[i][0].item().toFloat() * expandrate[0]);
            top = static_cast<int>(detectedBoxes[i][1].item().toFloat() * expandrate[1]);
            right = static_cast<int>(detectedBoxes[i][2].item().toFloat() * expandrate[0]);
            bottom = static_cast<int>(detectedBoxes[i][3].item().toFloat() * expandrate[1]);
            //left image
            if (right <= originalWidth) bboxesYolo_left.emplace_back(left, top, (right - left), (bottom - top));
            //right image
            else if (left >= originalWidth) bboxesYolo_right.emplace_back((left - originalWidth), top, (right - left), (bottom - top));
        }

        //std::cout << "finish converting torch::Tensor to cv::Rect2d" << std::endl;

        /*  compare detected bbox and TM bbox  */
        std::vector<cv::Rect2d> newDetections; // for storing newDetection

        /* start comparison */
        /* if found same things : push_back detected template and classIndex, else: make sure that push_back only -1 */
        //std::cout << "classIndexesTM size :: left = " << classIndexesTM_left.size() <<", right="<<classIndexesTM_right.size()<< std::endl;
        //left
        if (!classIndexesTM_left.empty()) matchingTracker(candidateIndex, bboxesCandidate_left, classIndexesTM_left, bboxesYolo_left, existedRoi_left, existedClass_left);
        if (!classIndexesTM_right.empty()) matchingTracker(candidateIndex, bboxesCandidate_right, classIndexesTM_right, bboxesYolo_right, existedRoi_right, existedClass_right);
    }

    void matchingTracker(int& candidateIndex, std::vector<cv::Rect2d>& bboxesCandidate, std::vector<int>& classIndexesTM, std::vector<cv::Rect2d>& bboxesYolo, std::vector<cv::Rect2d>& existedRoi, std::vector<int>& existedClass)
    {
        int indexMatch = 0;                    // index match
        int counterCandidateTM = 0; // number of candidate bbox
        bool boolIdentity = false; //if match or not
        // std::cout << "Comparison of TM and YOLO :: CandidateIndex:" << candidateIndex << std::endl;
        /* there is classes in TM trackeing */
        int counterIteration = 0;
        float max; //for checking max iou
        //std::cout << "bboxesYolo size=" << bboxesYolo.size() << std::endl;
        /*iterate for each clssIndexes of TM */
        for (const int classIndex : classIndexesTM)
        {
            max = IoUThresholdIdentity;      // set max value as threshold for lessening process volume
            //std::cout << classIndex << std::endl;
            /* bboxes Yolo still exist */
            if (!bboxesYolo.empty())
            {
                //std::cout << "$$$$$$$$ Couter Check :: counterIteration:"<<counterIteration<<", classIndexTM:"<<classIndex<<"counterCandidateTM : " << counterCandidateTM <<"classIndexesTM.size()="<<classIndexesTM.size()<<"bboxesCandidate.size()="<<bboxesCandidate.size()<<"bboxes YOLO size : " << bboxesYolo.size() << "existedClass.size()="<<existedClass.size()<<", existedRoi.size()="<<existedRoi.size()<<std::endl;
                /* TM tracker exist */
                if (classIndex >= 0)
                {
                    /* Tracker labels match like 0=0,1=1 */
                    if (candidateIndex == classIndex)
                    {
                        boolIdentity = false; // initialize
                        cv::Rect2d bboxTemp;  // temporal bbox storage
                        /* iterate for number of detected bboxes */
                        for (int counterCandidateYolo = 0; counterCandidateYolo < bboxesYolo.size(); counterCandidateYolo++)
                        {
                            //std::cout << counterCandidateYolo << std::endl;
                            //std::cout <<"bboxesCandidate.size()"<<bboxesCandidate.size()<<", bboxesCandidate.x = "<< bboxesCandidate[counterCandidateTM].x<<", top="<<bboxesCandidate[counterCandidateTM].y<<", width="<<bboxesCandidate[counterCandidateTM].width<<", height="<<bboxesCandidate[counterCandidateTM].height << std::endl;
                            //std::cout << bboxesYolo[counterCandidateYolo].x << std::endl;
                            float iou = calculateIoU_Rect2d(bboxesCandidate[counterCandidateTM], bboxesYolo[counterCandidateYolo]);
                            //std::cout << "iou=" << iou << std::endl;
                            if (iou >= max) // found similar bbox
                            {
                                max = iou;
                                bboxTemp = bboxesYolo[counterCandidateYolo]; // save top iou candidate
                                indexMatch = counterCandidateYolo;
                                boolIdentity = true;
                                //std::cout << "succeeded in matching : iou=" << max << std::endl;
                            }
                        }
                        /* find matched tracker */
                        if (boolIdentity)
                        {
                            //std::cout << "matching successful" << std::endl;
                            if (candidateIndex == 0)
                            {
                                // add data
                                existedClass.push_back(candidateIndex);
                            }
                            /* other label */
                            else
                            {
                                existedClass.at(counterIteration) = candidateIndex;
                            }
                            // std::cout << "TM and Yolo Tracker matched!" << std::endl;
                            existedRoi.push_back(bboxTemp);
                            // delete candidate bbox
                            bboxesYolo.erase(bboxesYolo.begin() + indexMatch); // erase detected bbox from bboxes Yolo -> number of bboxesYolo decrease
                        }
                        /* not found matched tracker -> return classIndex -1 to updatedClassIndexes */
                        else
                        {
                            //std::cout << "matching failed" << std::endl;
                            /* class label is 0 */
                            if (candidateIndex == 0)
                            {
                                // std::cout << "TM and Yolo Tracker didn't match" << std::endl;
                                existedClass.push_back(-1);
                            }
                            /* class label is other than 0 */
                            else
                            {
                                // std::cout << "TM and Yolo Tracker didn't match" << std::endl;
                                existedClass.at(counterIteration) = -1;
                            }
                        }
                        /* delete candidate bbox */
                        bboxesCandidate.erase(bboxesCandidate.begin() + counterCandidateTM); // delete TM latest roi to maintain roi order
                    }
                    /* other labels -> push back same label to maintain order only when candidateIndex=0 */
                    else if (candidateIndex < classIndex && candidateIndex == 0)
                    {
                        existedClass.push_back(classIndex);
                        counterCandidateTM++; // for maintain order of existed roi
                    }
                    /* only valid if classIndex != 0 */
                    else if (candidateIndex < classIndex && candidateIndex != 0)
                    {
                        counterCandidateTM++; // for maintaining order of existed roi
                    }
                }
                /* templateMatching Tracking was fault : classIndex = -1 */
                else
                {
                    /* if existedClass label is -1, return -1 only when candidate Index is 0. Then otherwise,
                     * only deal with the case when class label is equal to candidate index
                    */
                    if (candidateIndex == 0)
                    {
                        // std::cout << "add label -1 because we are label 1 even if we didn't experince" << std::endl;
                        existedClass.push_back(-1);
                    }
                }
            }
            /* bboxes Yolo already disappear -> previous trackers was failed here */
            else
            {
                /* candidate index == 0 */
                if (candidateIndex == 0)
                {
                    /* if same label -> failed to track in YOLO  */
                    if (classIndex == candidateIndex)
                    {
                        existedClass.push_back(-1);
                        if (!bboxesCandidate.empty())
                        {
                            bboxesCandidate.erase(bboxesCandidate.begin() + counterCandidateTM); // delete TM latest roi to maintain roi order
                        }
                    }
                    /* class label is 1,2,3,... */
                    else if (classIndex > candidateIndex)
                    {
                        existedClass.push_back(classIndex);
                        counterCandidateTM++; // maintain existedROI order
                    }
                    /* else classIndex != candidateIndex */
                    else if (classIndex == -1)
                    {
                        existedClass.push_back(-1);
                    }
                }
                /* class label is other than 0 */
                else
                {
                    if (classIndex == candidateIndex)
                    {
                        existedClass.at(counterIteration) = -1; // update label as -1
                        if (!bboxesCandidate.empty())
                        {
                            bboxesCandidate.erase(bboxesCandidate.begin() + counterCandidateTM); // delete TM latest roi to maintain roi order
                        }
                    }
                    else if (candidateIndex < classIndex)
                    {
                        counterCandidateTM++; // for maintaining order of existed roi
                    }
                }
            }
            counterIteration++;
        }
    }

    void newDetection(int& candidateIndex, std::vector<cv::Rect2d>& bboxesYolo, std::vector<cv::Rect2d>& newRoi, std::vector<int>& newClass)
    {
        int numNewDetection = bboxesYolo.size(); // number of new detections
        /* if there is a new detection */
        if (numNewDetection != 0)
        {
            for (int i = 0; i < numNewDetection; i++)
            {
                newRoi.push_back(bboxesYolo[i]);
                newClass.push_back(candidateIndex);
            }
        }
    }

    void noYoloDetect(int& candidateIndex, std::vector<cv::Rect2d>& bboxesCandidate, std::vector<int>& classIndexesTM, std::vector<int>& existedClass)
    {
        if (candidateIndex == 0)
        {
            if (!classIndexesTM.empty())
            {
                int counterCandidate = 0;
                for (const int& classIndex : classIndexesTM)
                {
                    /* if same label -> failed to track in YOLO  */
                    if (classIndex == candidateIndex)
                    {
                        existedClass.push_back(-1);
                        if (!bboxesCandidate.empty())
                        {
                            bboxesCandidate.erase(bboxesCandidate.begin() + counterCandidate); // erase existed roi to maintain roi order
                        }
                    }
                    /* else classIndex != candidateIndex */
                    else if (classIndex > candidateIndex && classIndex != -1)
                    {
                        existedClass.push_back(classIndex);
                        counterCandidate++;
                    }
                    /* else classIndex != candidateIndex */
                    else if (classIndex == -1)
                    {
                        existedClass.push_back(-1);
                    }
                }
            }
            /* No TM tracker */
            else
            {
                // std::cout << "No Detection , no tracker" << std::endl;
                /* No detection ,no trackers -> Nothing to do */
            }
        }
        /* if candidateIndex is other than 0 */
        else
        {
            if (!existedClass.empty())
            {
                int counterCandidate = 0;
                int counterIteration = 0;
                for (const int& classIndex : existedClass)
                {
                    // std::cout << "bboxesCandidate size = " << bboxesCandidate.size() << std::endl;
                    // std::cout << "counterCandidate=" << counterCandidate << std::endl;
                    /* if same label -> failed to track in YOLO  */
                    if (classIndex == candidateIndex)
                    {
                        existedClass.at(counterIteration) = -1;
                        if (!bboxesCandidate.empty())
                        {
                            bboxesCandidate.erase(bboxesCandidate.begin() + counterCandidate);
                        }
                    }
                    /* if classIndex is greater than candidateIndex -> keep bbox*/
                    else if (candidateIndex < classIndex)
                    {
                        counterCandidate++;
                    }
                    counterIteration++;
                }
            }
            /* No TM tracker */
            else
            {
                // std::cout << "No Detection , no tracker" << std::endl;
                /* No detection ,no trackers -> Nothing to do */
            }
        }
    }

    void push2Queue_left(std::vector<cv::Rect2d>& existedRoi, std::vector<cv::Rect2d>& newRoi,
        std::vector<int>& existedClass, std::vector<int>& newClass, cv::Mat1b& frame,
        std::vector<std::vector<cv::Rect2d>>& posSaver, std::vector<std::vector<int>>& classSaver,
        const int& frameIndex, std::vector<int>& detectedFrame, std::vector<int>& detectedFrameClass,
        std::queue<std::vector<int>>& queueYoloClassIndexLeft, std::queue<std::vector<cv::Rect2d>>& queueYoloBboxLeft,
        std::queue<std::vector<cv::Mat1b>>& queueYoloTemplateLeft, std::queue<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>>& queueTrackerYolo_left
    )
    {
        /*
         * push detection data to queueLeft
         */
        std::vector<cv::Rect2d> updatedRoi;
        updatedRoi.reserve(100);
        std::vector<cv::Mat1b> updatedTemplates;
        updatedTemplates.reserve(100);
        std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>> updatedTrackers;
        updatedTrackers.reserve(100);
        std::vector<int> updatedClassIndexes;
        updatedClassIndexes.reserve(300);
        /* update data */
        updateData_left(existedRoi, newRoi, existedClass, newClass, frame, updatedRoi, updatedTrackers, updatedTemplates,updatedClassIndexes);
        /* detection is successful */
        if (!updatedRoi.empty())
        {
            // std::cout << "detection succeeded" << std::endl;
            // save detected data
            posSaver.push_back(updatedRoi);
            classSaver.push_back(updatedClassIndexes);
            detectedFrame.push_back(frameIndex);
            detectedFrameClass.push_back(frameIndex);
            // push detected data
            // std::unique_lock<std::mutex> lock(mtxYoloLeft);
            /* initialize queueYolo for maintaining data consistency */
            if (!queueYoloBboxLeft.empty())
                queueYoloBboxLeft.pop();
            if (!queueYoloTemplateLeft.empty())
                queueYoloTemplateLeft.pop();
            if (!queueYoloClassIndexLeft.empty())
                queueYoloClassIndexLeft.pop();
            if (!queueTrackerYolo_left.empty())
                queueTrackerYolo_left.pop();
            /* finish initialization */
            queueYoloBboxLeft.push(updatedRoi);
            //MOSSE
            //std::cout << "Yolo :: updatedTrackers :: size=" << updatedTrackers.size() << std::endl;
            queueTrackerYolo_left.push(updatedTrackers);
            //Template Matching
            queueYoloTemplateLeft.push(updatedTemplates);
            //std::cout << " DETECTION SUCCESS :: YOLO class size = " << updatedClassIndexes.size() << std::endl;
            queueYoloClassIndexLeft.push(updatedClassIndexes);
        }
        /* no object detected -> return class label -1 if TM tracker exists */
        else
        {
            if (!updatedClassIndexes.empty())
            {
                // std::unique_lock<std::mutex> lock(mtxYoloLeft);
                /* initialize queueYolo for maintaining data consistency */
                if (!queueYoloClassIndexLeft.empty())
                    queueYoloClassIndexLeft.pop();
                queueYoloClassIndexLeft.push(updatedClassIndexes);
                detectedFrameClass.push_back(frameIndex);
                classSaver.push_back(updatedClassIndexes);
                //std::cout << " Detection FAILED :: YOLO class size = " << updatedClassIndexes.size() << std::endl;
                //int numLabels = updatedClassIndexes.size();
                //queueNumLabels.push(numLabels);
            }
            /* no class Indexes -> nothing to do */
            else
            {
                /* go trough */
            }
        }
    }

    void push2Queue_right(std::vector<cv::Rect2d>& existedRoi, std::vector<cv::Rect2d>& newRoi,
        std::vector<int>& existedClass, std::vector<int>& newClass, cv::Mat1b& frame,
        std::vector<std::vector<cv::Rect2d>>& posSaver, std::vector<std::vector<int>>& classSaver,
        const int& frameIndex, std::vector<int>& detectedFrame, std::vector<int>& detectedFrameClass,
        std::queue<std::vector<int>>& queueYoloClassIndexLeft, std::queue<std::vector<cv::Rect2d>>& queueYoloBboxLeft,
        std::queue<std::vector<cv::Mat1b>>& queueYoloTemplateLeft, std::queue<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>>& queueTrackerYolo_left
    )
    {
        /*
         * push detection data to queueLeft
         */
        std::vector<cv::Rect2d> updatedRoi;
        updatedRoi.reserve(100);
        std::vector<cv::Mat1b> updatedTemplates;
        updatedTemplates.reserve(100);
        std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>> updatedTrackers;
        updatedTrackers.reserve(100);
        std::vector<int> updatedClassIndexes;
        updatedClassIndexes.reserve(300);
        /* update data */
        //MOSSE
        updateData_right(existedRoi, newRoi, existedClass, newClass, frame, updatedRoi, updatedTrackers,updatedTemplates, updatedClassIndexes);
        /* detection is successful */
        if (!updatedRoi.empty())
        {
            // std::cout << "detection succeeded" << std::endl;
            // save detected data
            posSaver.push_back(updatedRoi);
            classSaver.push_back(updatedClassIndexes);
            detectedFrame.push_back(frameIndex);
            detectedFrameClass.push_back(frameIndex);
            // push detected data
            // std::unique_lock<std::mutex> lock(mtxYoloLeft);
            /* initialize queueYolo for maintaining data consistency */
            if (!queueYoloBboxLeft.empty())
                queueYoloBboxLeft.pop();
            if (!queueYoloTemplateLeft.empty())
                queueYoloTemplateLeft.pop();
            if (!queueYoloClassIndexLeft.empty())
                queueYoloClassIndexLeft.pop();
            if (!queueTrackerYolo_left.empty())
                queueTrackerYolo_left.pop();
            /* finish initialization */
            queueYoloBboxLeft.push(updatedRoi);
            //MOSSE
            //std::cout << "Yolo :: updatedTrackers :: size=" << updatedTrackers.size() << std::endl;
            queueTrackerYolo_left.push(updatedTrackers);
            //Template Matching
            queueYoloTemplateLeft.push(updatedTemplates);
            queueYoloClassIndexLeft.push(updatedClassIndexes);
            //std::cout << " DETECTION SUCCESS :: YOLO class size = " << updatedClassIndexes.size() << std::endl;
        }
        /* no object detected -> return class label -1 if TM tracker exists */
        else
        {
            if (!updatedClassIndexes.empty())
            {
                // std::unique_lock<std::mutex> lock(mtxYoloLeft);
                /* initialize queueYolo for maintaining data consistency */
                if (!queueYoloClassIndexLeft.empty())
                    queueYoloClassIndexLeft.pop();
                queueYoloClassIndexLeft.push(updatedClassIndexes);
                detectedFrameClass.push_back(frameIndex);
                classSaver.push_back(updatedClassIndexes);
                //std::cout << " DETECTION FAILED :: YOLO class size = " << updatedClassIndexes.size() << std::endl;
                //int numLabels = updatedClassIndexes.size();
                //queueNumLabels.push(numLabels);
            }
            /* no class Indexes -> nothing to do */
            else
            {
                /* go trough */
            }
        }
    }

    void updateData_left(std::vector<cv::Rect2d>& existedRoi, std::vector<cv::Rect2d>& newRoi, std::vector<int>& existedClass, std::vector<int>& newClass,
        cv::Mat1b& frame, std::vector<cv::Rect2d>& updatedRoi, std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers,std::vector<cv::Mat1b>& updatedTemplates,
        std::vector<int>& updatedClassIndexes)
    {
        // std::cout << "updateData function" << std::endl;
        /* firstly add existed class and ROi*/
        // std::cout << "Existed class" << std::endl;
        if (!existedRoi.empty())
        {
            /* update bbox and templates */
            for (cv::Rect2d& roi : existedRoi)
            {
                updatedRoi.push_back(roi);
                updatedTemplates.push_back(frame(roi));
                cv::Ptr<cv::mytracker::TrackerMOSSE> tracker = cv::mytracker::TrackerMOSSE::create();
                tracker->init(frame, roi);
                updatedTrackers.push_back(tracker);
            }
            for (const int& classIndex : existedClass)
            {
                updatedClassIndexes.push_back(classIndex);
                // std::cout << classIndex << " ";
            }
            // std::cout << std::endl;
        }
        else
        {
            if (!existedClass.empty())
            {
                for (const int& classIndex : existedClass)
                {
                    updatedClassIndexes.push_back(classIndex);
                    // std::cout << classIndex << " ";
                }
                // std::cout << std::endl;
            }
        }
        /* secondly add new roi and class */
        if (!newRoi.empty())
        {
            // std::cout << "new detection" << std::endl;
            for (cv::Rect2d& roi : newRoi)
            {
                updatedRoi.push_back(roi);
                updatedTemplates.push_back(frame(roi));
                cv::Ptr<cv::mytracker::TrackerMOSSE> tracker = cv::mytracker::TrackerMOSSE::create();
                tracker->init(frame, roi);
                updatedTrackers.push_back(tracker);
            }
            for (const int& classIndex : newClass)
            {
                updatedClassIndexes.push_back(classIndex);
                // std::cout << classIndex << " ";
            }
            // std::cout << std::endl;
        }
        else
        {
            if (!newClass.empty())
            {
                for (const int& classIndex : newClass)
                {
                    updatedClassIndexes.push_back(classIndex);
                    // std::cout << classIndex << " ";
                }
                // std::cout << std::endl;
            }
        }
    }

    void updateData_right(std::vector<cv::Rect2d>& existedRoi, std::vector<cv::Rect2d>& newRoi, std::vector<int>& existedClass, std::vector<int>& newClass,
        cv::Mat1b& frame, std::vector<cv::Rect2d>& updatedRoi, std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers, std::vector<cv::Mat1b>& updatedTemplates,
        std::vector<int>& updatedClassIndexes)
    {
        // std::cout << "updateData function" << std::endl;
        /* firstly add existed class and ROi*/
        // std::cout << "Existed class" << std::endl;
        if (!existedRoi.empty())
        {
            /* update bbox and templates */
            for (cv::Rect2d& roi : existedRoi)
            {
                updatedRoi.push_back(roi);
                roi.x += (double)originalWidth; //add original image width for template image
                updatedTemplates.push_back(frame(roi));
                cv::Ptr<cv::mytracker::TrackerMOSSE> tracker = cv::mytracker::TrackerMOSSE::create();
                tracker->init(frame, roi);
                updatedTrackers.push_back(tracker);
            }
            for (const int& classIndex : existedClass)
            {
                updatedClassIndexes.push_back(classIndex);
                // std::cout << classIndex << " ";
            }
            // std::cout << std::endl;
        }
        else
        {
            if (!existedClass.empty())
            {
                for (const int& classIndex : existedClass)
                {
                    updatedClassIndexes.push_back(classIndex);
                    // std::cout << classIndex << " ";
                }
                // std::cout << std::endl;
            }
        }
        /* secondly add new roi and class */
        if (!newRoi.empty())
        {
            // std::cout << "new detection" << std::endl;
            for (cv::Rect2d& roi : newRoi)
            {
                updatedRoi.push_back(roi);
                roi.x += (double)originalWidth;
                updatedTemplates.push_back(frame(roi));
                cv::Ptr<cv::mytracker::TrackerMOSSE> tracker = cv::mytracker::TrackerMOSSE::create();
                tracker->init(frame, roi);
                updatedTrackers.push_back(tracker);
            }
            for (const int& classIndex : newClass)
            {
                updatedClassIndexes.push_back(classIndex);
                // std::cout << classIndex << " ";
            }
            // std::cout << std::endl;
        }
        else
        {
            if (!newClass.empty())
            {
                for (const int& classIndex : newClass)
                {
                    updatedClassIndexes.push_back(classIndex);
                    // std::cout << classIndex << " ";
                }
                // std::cout << std::endl;
            }
        }
    }
};

#endif 