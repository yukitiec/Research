#pragma once

/*
* tracker.h
*   - MOT class
*   - template matching
*/

#ifndef TRACKER_H
#define TRACKER_H

#include "stdafx.h"
#include "global_parameters.h"
#include "mosse.h"
#include "kalmanfilter.h"
#include "utility.h"

//Yolo signals
extern std::queue<bool> queueYolo_tracker2seq_left, queueYolo_tracker2seq_right;

// tracker
extern const bool boolMOSSE;
extern const double threshold_mosse;

// queue definition
extern std::queue<std::array<cv::Mat1b, 2>> queueFrame; // queue for frame
extern std::queue<int> queueFrameIndex;  // queue for frame index

//mosse
extern std::queue<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>> queueTrackerYolo_left;
extern std::queue<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>> queueTrackerYolo_right;
extern std::queue<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>> queueTrackerMOSSE_left;
extern std::queue<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>> queueTrackerMOSSE_right;

// left cam
extern std::queue<std::vector<cv::Mat1b>> queueYoloTemplateLeft; // queue for yolo template : for real cv::Mat type
extern std::queue<std::vector<cv::Rect2d>> queueYoloBboxLeft;    // queue for yolo bbox
extern std::queue<std::vector<cv::Mat1b>> queueTMTemplateLeft;   // queue for templateMatching template img : for real cv::Mat
extern std::queue<std::vector<cv::Rect2d>> queueTMBboxLeft;      // queue for templateMatching bbox
extern std::queue<std::vector<int>> queueYoloClassIndexLeft;     // queue for class index
extern std::queue<std::vector<int>> queueTMClassIndexLeft;       // queue for class index
extern std::queue<std::vector<bool>> queueTMScalesLeft;          // queue for search area scale
extern std::queue<std::vector<std::vector<int>>> queueMoveLeft; //queue for saving previous move
//std::queue<int> queueNumLabels;                           // current labels number -> for maintaining label number consistency
extern std::queue<bool> queueStartYolo_left; //if new Yolo inference can start

// right cam
extern std::queue<std::vector<cv::Mat1b>> queueYoloTemplateRight; // queue for yolo template : for real cv::Mat type
extern std::queue<std::vector<cv::Rect2d>> queueYoloBboxRight;    // queue for yolo bbox
extern std::queue<std::vector<cv::Mat1b>> queueTMTemplateRight;   // queue for templateMatching template img : for real cv::Mat
extern std::queue<std::vector<cv::Rect2d>> queueTMBboxRight;      // queue for TM bbox
extern std::queue<std::vector<int>> queueYoloClassIndexRight;     // queue for class index
extern std::queue<std::vector<int>> queueTMClassIndexRight;       // queue for class index
extern std::queue<std::vector<bool>> queueTMScalesRight;          // queue for search area scale
extern std::queue<std::vector<std::vector<int>>> queueMoveRight; //queue for saving previous move
extern std::queue<bool> queueStartYolo_right; //if new Yolo inference can start

//from tm to yolo
extern std::queue<std::vector<cv::Rect2d>> queueTM2YoloBboxLeft;      // queue for templateMatching bbox
extern std::queue<std::vector<int>> queueTM2YoloClassIndexLeft;     // queue for class index
extern std::queue<std::vector<cv::Rect2d>> queueTM2YoloBboxRight;      // queue for templateMatching bbox
extern std::queue<std::vector<int>> queueTM2YoloClassIndexRight;     // queue for class index

// 3D positioning ~ trajectory prediction
extern std::queue<int> queueTargetFrameIndex_left;                      // TM estimation frame
extern std::queue<int> queueTargetFrameIndex_right;
extern std::queue<std::vector<cv::Rect2d>> queueTargetBboxesLeft;  // bboxes from template matching for predict objects' trajectory
extern std::queue<std::vector<cv::Rect2d>> queueTargetBboxesRight; // bboxes from template matching for predict objects' trajectory
extern std::queue<std::vector<int>> queueTargetClassIndexesLeft;   // class from template matching for maintain consistency
extern std::queue<std::vector<int>> queueTargetClassIndexesRight;  // class from template matching for maintain consistency

//from seq : kalman prediction
extern std::queue<std::vector<std::vector<double>>> queueKfPredictLeft; //{label, left,top,width,height}
extern std::queue<std::vector<std::vector<double>>> queueKfPredictRight;

class TemplateMatching
{
private:
    // template matching constant value setting
    const double scaleXTM = 1.5; //2.0 //for throwing //1.5 //for switching // search area scale compared to roi
    const double scaleYTM = 1.5;
    const double scaleXYolo = 3.0; //3.5 //for throwing //2.0 //for switching
    const double scaleYYolo = 3.0; //3.5 //for throwing
    const double matchingThreshold = 0.0;             // matching threshold
    const int MATCHINGMETHOD = cv::TM_CCOEFF_NORMED; // //cv::TM_SQDIFF_NORMED -> unique background, cv::TM_CCOEFF_NORMED :: ‘ŠŠÖŒW”, cv::TM_CCORR_NORMED -> patterned background // TM_SQDIFF_NORMED is good for small template
    const double MoveThreshold = 0.0;                 // move threshold of objects

    bool bool_multithread = false; //done -> false if run MOSSE in multiple threads
    bool bool_iouCheck = true; //done -> true;check current tracker iou -> if overlapped
    const float IoU_overlapped = 0.5; //done->0.6~75
    std::vector<int> defaultMove{ 0, 0 };
    const double gamma = 0.3; //ration of current and past for tracker velocity, the larger, more important current :: gamma*current+(1-gamma)*past
    const double MIN_SEARCH = 10;
    const int MAX_VEL = 40; // max roi move 
    const bool bool_dynamicScale = true; //done -> true :: scaleXTM ::smaller is good //dynamic scale according to current motion
    const bool bool_TBD = false; //done-> false //if true : update position too
    const bool bool_kf = true; //done->true
    const bool bool_skip = true; //skip updating for occlusion and switching prevention
    const bool bool_check_psr = true; //done->true //which tracker adopt : detection or tracking
    const double min_keep_psr = 5.0;
    const bool bool_augment = true;
public:
    // vector for saving position
    //left
    std::vector<std::vector<cv::Rect2d>> posSaverTMLeft;
    std::vector<std::vector<int>> classSaverTMLeft;

    //right
    std::vector<std::vector<cv::Rect2d>> posSaverTMRight;
    std::vector<std::vector<int>> classSaverTMRight;
    //detected frame
    //left
    std::vector<int> detectedFrameLeft;
    std::vector<int> detectedFrameClassLeft;
    //right
    std::vector<int> detectedFrameRight;
    std::vector<int> detectedFrameClassRight;

    float t_elapsed = 0;
    //constructor
    TemplateMatching()
    {
        std::cout << "construtor of template matching" << std::endl;
    }

    void main()
    {
        //constructor
        Utility utTM;
        //initialization
        while (!queueTrackerYolo_left.empty())
            queueTrackerYolo_left.pop();
        while (!queueTrackerYolo_right.empty())
            queueTrackerYolo_right.pop();
        while (!queueTrackerMOSSE_left.empty())
            queueTrackerMOSSE_left.pop();
        while (!queueTrackerMOSSE_right.empty())
            queueTrackerMOSSE_right.pop();
        int countIteration = 0;
        int counterFinish = 0;
        /* sleep for 2 seconds */
        // std::this_thread::sleep_for(std::chrono::seconds(30));
        //  for each img iterations
        int counterStart = 0;
        while (true)
        {
            //std::cout << "counterStart=" << counterStart << std::endl;

            if (!queueYoloClassIndexLeft.empty() || !queueYoloClassIndexRight.empty())
            {
                std::cout << "TM :: by starting " << 2 - counterStart << std::endl;
                if (counterStart == 2)
                    break;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                while (true)
                {
                    if (!queueYoloClassIndexLeft.empty()) queueYoloClassIndexLeft.pop();
                    if (!queueYoloClassIndexRight.empty()) queueYoloClassIndexRight.pop();
                    if (!queueYoloBboxLeft.empty()) queueYoloBboxLeft.pop();
                    if (!queueYoloBboxRight.empty()) queueYoloBboxRight.pop();
                    if (!queueTrackerYolo_left.empty()) queueTrackerYolo_left.pop();
                    if (!queueTrackerYolo_right.empty()) queueTrackerYolo_right.pop();
                    if (!queueYoloTemplateLeft.empty()) queueYoloTemplateLeft.pop();
                    if (!queueYoloTemplateRight.empty()) queueYoloTemplateRight.pop();

                    if (queueYoloClassIndexLeft.empty() && queueYoloClassIndexRight.empty())
                    {
                        counterStart++;
                        std::cout << "remove data" << std::endl;
                        break;
                    }
                }
            }
        }

        std::cout << "start tracking" << std::endl;
        while (true) // continue until finish
        {
            //std::cout << " -- " << countIteration << " -- " << std::endl;
            // get img from queue
            //std::cout << "get imgs" << std::endl;
            if (queueFrame.empty())
            {
                if (counterFinish == 10)
                {
                    break;
                }
                counterFinish++;
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                std::cout << "Tracker :: By finish : remain count is " << (10 - counterFinish) << std::endl;
                continue;
            }
            else if (!queueFrame.empty())
            {
                if ((!queueYoloClassIndexLeft.empty() && !queueYoloClassIndexRight.empty()) || !queueTrackerMOSSE_left.empty() || !queueTrackerMOSSE_right.empty() || !queueKfPredictLeft.empty() || !queueKfPredictRight.empty()) //some tracker exist
                {
                    //std::cout << !queueTrackerYolo_left.empty() << !queueTrackerYolo_right.empty() << !queueTrackerMOSSE_left.empty() << !queueTrackerMOSSE_right.empty() << !queueKfPredictLeft.empty() << !queueKfPredictRight.empty() << std::endl;
                    counterFinish = 0; // reset
                    std::array<cv::Mat1b, 2> frames;
                    int frameIndex;
                    bool boolImgs = utTM.getImagesFromQueueTM(frames, frameIndex);
                    cv::Mat1b frame_left = frames[LEFT_CAMERA];
                    cv::Mat1b frame_right = frames[RIGHT_CAMERA];
                    if ((frame_left.rows > 0 && frame_right.rows > 0))
                    {
                        /*start template matching process */
                        auto start = std::chrono::high_resolution_clock::now();
                        std::thread thread_left(&TemplateMatching::templateMatching, this, std::ref(frame_left), std::ref(frameIndex), std::ref(posSaverTMLeft), std::ref(classSaverTMLeft), std::ref(detectedFrameLeft), std::ref(detectedFrameClassLeft),
                            std::ref(queueTMClassIndexLeft), std::ref(queueTMBboxLeft), std::ref(queueTMTemplateLeft), std::ref(queueTrackerMOSSE_left), std::ref(queueTMScalesLeft), std::ref(queueMoveLeft),
                            std::ref(queueYoloClassIndexLeft), std::ref(queueYoloBboxLeft), std::ref(queueYoloTemplateLeft), std::ref(queueTrackerYolo_left), std::ref(queueStartYolo_left),
                            std::ref(queueTargetFrameIndex_left), std::ref(queueTargetClassIndexesLeft), std::ref(queueTargetBboxesLeft), std::ref(queueYolo_tracker2seq_left),
                            std::ref(queueKfPredictLeft), std::ref(queueTM2YoloClassIndexLeft), std::ref(queueTM2YoloBboxLeft));
                        //right
                        templateMatching(frame_right, frameIndex, posSaverTMRight, classSaverTMRight, detectedFrameRight, detectedFrameClassRight,
                            queueTMClassIndexRight, queueTMBboxRight, queueTMTemplateRight, queueTrackerMOSSE_right, queueTMScalesRight, queueMoveRight,
                            queueYoloClassIndexRight, queueYoloBboxRight, queueYoloTemplateRight, queueTrackerYolo_right, queueStartYolo_right,
                            queueTargetFrameIndex_right, queueTargetClassIndexesRight, queueTargetBboxesRight, queueYolo_tracker2seq_right,
                            queueKfPredictRight, queueTM2YoloClassIndexRight, queueTM2YoloBboxRight);
                        thread_left.join();
                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                        float time_iteration = static_cast<float>(duration.count());
                        t_elapsed = t_elapsed + time_iteration;
                        if (time_iteration < 2500)
                        {
                            std::cout << "Tracker :: " << time_iteration << " microseconds" << std::endl;
                        }
                        else
                        {
                            int frame_delete = static_cast<int>((time_iteration / 2500) - 1);
                            if (frame_delete >= 1)
                            {
                                std::cout << "Tracker :: " << time_iteration << " microseconds and " << frame_delete << " frame will be deleted" << std::endl;
                                for (int i = 0; i < frame_delete; i++)
                                {
                                    if (!queueFrame.empty()) queueFrame.pop();
                                    if (!queueFrameIndex.empty()) queueFrameIndex.pop();
                                }
                            }
                        }
                        //std::cout << "Tracking -- " << countIteration << " --" << std::endl;
                        //std::cout << "time taken by tracking :: " << duration.count() << " microsecodes" << std::endl;
                        countIteration++;
                    }
                }
            }
        }
        if (countIteration != 0) std::cout << "Tracking process speed :: " << static_cast<int>(countIteration / t_elapsed * 1000000) << " Hz for" << countIteration << "cycles" << std::endl;
        // check data
        std::cout << "position saver : TM : " << std::endl;
        std::cout << " : Left : " << std::endl;
        std::cout << "posSaverTMLeft size:" << posSaverTMLeft.size() << ", detectedFrame size:" << detectedFrameLeft.size() << std::endl;
        utTM.checkStorageTM(posSaverTMLeft, detectedFrameLeft, file_tm_bbox_left);
        std::cout << "Class saver : TM : " << std::endl;
        std::cout << "classSaverTMLeft size:" << classSaverTMLeft.size() << ", detectedFrameClass size:" << detectedFrameClassLeft.size() << std::endl;
        utTM.checkClassStorageTM(classSaverTMLeft, detectedFrameClassLeft, file_tm_class_left);
        std::cout << " : Right : " << std::endl;
        std::cout << "posSaverTMRight size:" << posSaverTMRight.size() << ", detectedFrame size:" << detectedFrameRight.size() << std::endl;
        utTM.checkStorageTM(posSaverTMRight, detectedFrameRight, file_tm_bbox_right);
        std::cout << "Class saver : TM : " << std::endl;
        std::cout << "classSaverTMRight size:" << classSaverTMRight.size() << ", detectedFrameClass size:" << detectedFrameClassRight.size() << std::endl;
        utTM.checkClassStorageTM(classSaverTMRight, detectedFrameClassRight, file_tm_class_right);
    }

    /* Template Matching :: Left */
    void templateMatching(cv::Mat1b& img, const int& frameIndex,
        std::vector<std::vector<cv::Rect2d>>& posSaver, std::vector<std::vector<int>>& classSaver, std::vector<int>& detectedFrame, std::vector<int>& detectedFrameClass,
        std::queue<std::vector<int>>& queueTMClassIndex, std::queue<std::vector<cv::Rect2d>>& queueTMBbox,
        std::queue<std::vector<cv::Mat1b>>& queueTMTemplate, std::queue<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>>& queueTrackerMOSSE, std::queue<std::vector<bool>>& queueTMScales,
        std::queue<std::vector<std::vector<int>>>& queueMove, std::queue<std::vector<int>>& queueYoloClassIndex, std::queue<std::vector<cv::Rect2d>>& queueYoloBbox,
        std::queue<std::vector<cv::Mat1b>>& queueYoloTemplate, std::queue<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>>& queueTrackerYolo, std::queue<bool>& queueStartYolo,
        std::queue<int>& queueTargetFrameIndex, std::queue<std::vector<int>>& queueTargetClassIndexes, std::queue<std::vector<cv::Rect2d>>& queueTargetBboxes, std::queue<bool>& queueYolo_tracker2seq,
        std::queue<std::vector<std::vector<double>>>& queueKfPredict, std::queue<std::vector<int>>& queueTM2YoloClassIndex, std::queue<std::vector<cv::Rect2d>>& queueTM2YoloBbox
    )
    {
        // for updating templates
        std::vector<cv::Rect2d> updatedBboxes;
        updatedBboxes.reserve(30);
        std::vector<cv::Mat1b> updatedTemplates;
        updatedTemplates.reserve(30);
        std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>> updatedTrackers;
        updatedTrackers.reserve(30);
        std::vector<std::vector<int>> updatedMove;
        updatedMove.reserve(30);
        std::vector<int> updatedClasses;
        updatedClasses.reserve(300);

        // get Template Matching data
        std::vector<int> classIndexTM;
        classIndexTM.reserve(30);
        int numTrackersTM = 0;
        std::vector<cv::Rect2d> bboxesTM;
        bboxesTM.reserve(30);
        std::vector<cv::Mat1b> templatesTM;
        templatesTM.reserve(30);
        std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>> trackers_mosse;
        trackers_mosse.reserve(30);
        std::vector<bool> boolScalesTM;
        boolScalesTM.reserve(30);   // search area scale
        std::vector<std::vector<int>> previousMoveTM;
        previousMoveTM.reserve(30);
        bool boolTrackerTM = false; // whether tracking is successful
        bool boolTrackerYolo = false; //whether yolo data is available
        //get data
        getData(img, boolTrackerTM, classIndexTM, bboxesTM, trackers_mosse, templatesTM, boolScalesTM, previousMoveTM, numTrackersTM, queueTMClassIndex, queueTMBbox, queueTrackerMOSSE,queueTMTemplate, queueTMScales, queueMove, queueKfPredict);

        if (!queueYoloClassIndex.empty())
        {
            /* get Yolo data and update Template matchin data */
            organizeData(img,boolScalesTM, previousMoveTM, boolTrackerYolo, classIndexTM, trackers_mosse, templatesTM, bboxesTM, updatedTrackers, updatedTemplates, updatedBboxes, updatedClasses, updatedMove, numTrackersTM, queueYoloClassIndex, queueYoloBbox, queueTrackerYolo, queueYoloTemplate);
        }
        /* template from yolo isn't available but TM tracker exist */
        else if (boolTrackerTM)
        {
            updatedTrackers = trackers_mosse;
            updatedTemplates = templatesTM;
            updatedBboxes = bboxesTM;
            updatedClasses = classIndexTM;
            updatedMove = previousMoveTM;
        }
        /* no template is available */
        else
        {
            // nothing to do
        }
        //finish  preprocess
        //std::cout << "BoolTrackerTM :" << boolTrackerTM << ", boolTrackerYolo : " << boolTrackerYolo << std::endl;
        /*  Template Matching Process */
        if (boolTrackerTM || boolTrackerYolo)
        {
            //std::cout << "template matching process has started" << std::endl;
            int counterTracker = 0;
            int numClasses = updatedClasses.size();
            int numTrackers = updatedBboxes.size();
            //std::cout << "number of classes = " << numClasses << ", number of templates = " << numTrackers << ", number of bboxes = "<<updatedBboxes.size()<<", number of scales="<<boolScalesTM.size()<<std::endl;
            // Initialization for storing TM detection results
            // templates
            int rows = 1;
            int cols = 1;
            int defaultValue = 0;
            std::vector<cv::Mat1b> updatedTemplatesTM(numTrackers);//, cv::Mat1b(rows, cols, defaultValue));
            //trackers
            std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>> updatedTrackers_mosse(numTrackers);
            // bbox
            cv::Rect2d defaultROI(-1, -1, -1, -1);
            std::vector<cv::Rect2d> updatedBboxesTM(numTrackers, defaultROI);
            // class labels
            std::vector<int> updatedClassesTM(numClasses, -1);
            // scales
            bool defaultBool = false;
            std::vector<bool> updatedSearchScales(numTrackers, defaultBool);
            std::vector<std::vector<int>> updatedMoveTM(numTrackers); //roi move
            // finish initialization

            // template matching
            //std::cout << "processTM start" << std::endl;
            //tracking process
            process(updatedClasses, updatedTrackers, updatedTemplates, boolScalesTM, updatedBboxes, updatedMove, img, updatedTrackers_mosse, updatedTemplatesTM, updatedBboxesTM, updatedClassesTM, updatedSearchScales, updatedMoveTM);
            
            //std::cout << "processTM finish" << std::endl;
            //std::cout << "after MOSSE :: updatedTrackers = " << updatedTrackers_mosse.size() << ", updatedBboxesTM = " << updatedBboxesTM.size() << ", updatedClassesTM = " << updatedClassesTM.size() << "updatedScales =" << updatedSearchScales.size() << std::endl;
            //std::cout << "queueTrackerYolo_ size =" << queueTrackerYolo_.size() << std::endl;
            if (!updatedBboxesTM.empty())
            {
                if (!queueTMBbox.empty()) queueTMBbox.pop();
                queueTMBbox.push(updatedBboxesTM); // push roi
                posSaver.push_back(updatedBboxesTM);// save current position to the vector
                //sequence data
                if (!queueTargetFrameIndex.empty()) queueTargetFrameIndex.pop();
                if (!queueTargetClassIndexes.empty()) queueTargetClassIndexes.pop();
                if (!queueTargetBboxes.empty()) queueTargetBboxes.pop();
                queueTargetFrameIndex.push(frameIndex);
                queueTargetClassIndexes.push(updatedClassesTM);
                queueTargetBboxes.push(updatedBboxesTM);
                //search scales
                if (!updatedSearchScales.empty())
                {
                    if (!queueTMScales.empty()) queueTMScales.pop(); // pop before push
                    queueTMScales.push(updatedSearchScales);
                }
                //moveDistance
                if (!updatedMoveTM.empty())
                {
                    if (!queueMove.empty()) queueMove.pop();
                    queueMove.push(updatedMoveTM);
                }
                //MOSSE
                if (!updatedTrackers_mosse.empty())
                {
                    if (!queueTrackerMOSSE.empty()) queueTrackerMOSSE.pop();
                    queueTrackerMOSSE.push(updatedTrackers_mosse); // push template image
                }
                //Template Matching
                if (!updatedTemplatesTM.empty())
                {
                    if (!queueTMTemplate.empty()) queueTMTemplate.pop();
                    queueTMTemplate.push(updatedTemplatesTM); // push template image
                }
            }
            else //failed
            {
                if (!queueTMScales.empty()) queueTMScales.pop();
                if (!queueTMBbox.empty()) queueTMBbox.pop();
                if (!queueTrackerMOSSE.empty()) queueTrackerMOSSE.pop();
                if (!queueTMTemplate.empty()) queueTMTemplate.pop();
            }

            if (!updatedClassesTM.empty())
            {
                if (!queueTMClassIndex.empty()) queueTMClassIndex.pop();
                queueTMClassIndex.push(updatedClassesTM);
                classSaver.push_back(updatedClassesTM); // save current class to the saver
            }
            /* if yolo data is avilable -> send signal to target predict to change labels*/
            if (boolTrackerYolo)
            {
                //queueLabelUpdate.push(true);
                //std::cout << "TM2Yolo :: updatedClassIndex :: " << std::endl;
                //for (int& label : updatedClassesTM)
                //    std::cout << label;
                //std::cout << std::endl;
                if (!queueTM2YoloClassIndex.empty())  queueTM2YoloClassIndex.pop();
                if (!updatedClassesTM.empty()) queueTM2YoloClassIndex.push(updatedClassesTM);
                if (!queueTM2YoloBbox.empty()) queueTM2YoloBbox.pop();
                if (!updatedBboxesTM.empty()) queueTM2YoloBbox.push(updatedBboxesTM);
                if (!queueStartYolo.empty()) queueStartYolo.pop();
                queueStartYolo.push(true);
                queueYolo_tracker2seq.push(true); //for letting sequence.h know Yolo detected
                //std::cout << " :: TM :: class size = " << updatedClassesTM.size() << ", roi size=" << updatedBboxesTM.size() << std::endl;
                //std::cout << "push true to queueStartYolo :: TM" << std::endl;
            }
            else
            {
                //queueLabelUpdate.push(false);
            }
            detectedFrame.push_back(frameIndex);
            detectedFrameClass.push_back(frameIndex);
        }
        else // no template or bbox -> nothing to do
        {
            if (!classIndexTM.empty())
            {
                if (!queueTMClassIndex.empty()) queueTMClassIndex.pop();
                queueTMClassIndex.push(classIndexTM);
                classSaver.push_back(classIndexTM); // save current class to the saver
                detectedFrameClass.push_back(frameIndex);
            }
            if (!queueTMScales.empty()) queueTMScales.pop();
            if (!queueTMBbox.empty()) queueTMBbox.pop();
            if (!queueTrackerMOSSE.empty()) queueTrackerMOSSE.pop();
            if (!queueTMTemplate.empty()) queueTMTemplate.pop();
            if (!queueMove.empty()) queueMove.pop();
            else
            {
                //std::this_thread::sleep_for(std::chrono::milliseconds(3));
                // nothing to do
            }
        }
    }

    void getData(cv::Mat1b& frame, bool& boolTrackerTM, std::vector<int>& classIndexTM, std::vector<cv::Rect2d>& bboxesTM, std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& trackers_mosse,
        std::vector<cv::Mat1b>& templatesTM, std::vector<bool>& boolScalesTM, std::vector<std::vector<int>>& previousMove,
        int& numTrackersTM, std::queue<std::vector<int>>& queueTMClassIndex, std::queue<std::vector<cv::Rect2d>>& queueTMBbox, std::queue < std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>>& queueTrackerMOSSE,
        std::queue<std::vector<cv::Mat1b>>& queueTMTemplate, std::queue<std::vector<bool>>& queueTMScales,
        std::queue<std::vector<std::vector<int>>>& queueMove, std::queue<std::vector<std::vector<double>>>& queueKfPrediction)
    {
        if (!queueTMClassIndex.empty())
        {
            classIndexTM = queueTMClassIndex.front();
            numTrackersTM = classIndexTM.size();
        }
        if (!queueTMBbox.empty() && !queueTrackerMOSSE.empty() && !queueTMTemplate.empty() && !queueTMScales.empty() && !queueMove.empty())
        {
            //std::cout << "previous TM tracker is available" << std::endl;
            boolTrackerTM = true;

            bboxesTM = queueTMBbox.front();
            trackers_mosse = queueTrackerMOSSE.front();
            templatesTM = queueTMTemplate.front();
            boolScalesTM = queueTMScales.front();
            previousMove = queueMove.front();
        }
        else
            boolTrackerTM = false;
        if (bool_kf)
        {
            if (!queueKfPrediction.empty())
            {
                std::vector<std::vector<double>> kf_predictions = queueKfPrediction.front();
                queueKfPrediction.pop();
                int counter_label = 0;
                int counter_tracker = 0;
                for (std::vector<double>& kf_predict : kf_predictions)
                {
                    if (!kf_predict.empty() && classIndexTM[counter_label] < 0 && classIndexTM[counter_label]!=-2) //revival
                    {
                        classIndexTM[counter_label] = (int)kf_predict[0]; //update label
                        cv::Rect2d newRoi((double)std::min(std::max((int)kf_predict[1], 0), (frame.cols - (int)kf_predict[3] - 1)), (double)std::min(std::max((int)kf_predict[2], 0), (frame.rows - (int)kf_predict[4] - 1)), (double)kf_predict[3], (double)kf_predict[4]);
                        bboxesTM.insert(bboxesTM.begin() + counter_tracker, newRoi);
                        cv::Ptr<cv::mytracker::TrackerMOSSE> tracker = cv::mytracker::TrackerMOSSE::create();
                        tracker->init(frame, newRoi);
                        trackers_mosse.insert(trackers_mosse.begin() + counter_tracker, tracker);
                        templatesTM.insert(templatesTM.begin() + counter_tracker, frame(newRoi));
                        boolScalesTM.insert(boolScalesTM.begin() + counter_tracker, false);
                        previousMove.insert(previousMove.begin() + counter_tracker, defaultMove);
                        boolTrackerTM = true;
                    }
                    if (classIndexTM[counter_label] >= 0) counter_tracker++;
                    counter_label++;
                }
            }
        }
    }

    void organizeData(cv::Mat1b& frame,std::vector<bool>& boolScalesTM, std::vector<std::vector<int>>& previousMove, bool& boolTrackerYolo, std::vector<int>& classIndexTM, std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& trackers_mosse,
        std::vector<cv::Mat1b>& templatesTM, std::vector<cv::Rect2d>& bboxesTM, 
        std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers, std::vector<cv::Mat1b>& updatedTemplates, std::vector<cv::Rect2d>& updatedBboxes, std::vector<int>& updatedClasses, 
        std::vector<std::vector<int>>& updatedMove, int& numTrackersTM,
        std::queue<std::vector<int>>& queueYoloClassIndex, std::queue<std::vector<cv::Rect2d>>& queueYoloBbox, 
        std::queue<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>>& queueTrackerYolo,std::queue<std::vector<cv::Mat1b>>& queueYoloTemplate)
    {
        //std::unique_lock<std::mutex> lock(mtxYolo); // Lock the mutex
        //std::cout << "TM :: Yolo data is available" << std::endl;
        boolTrackerYolo = true;
        if (!boolScalesTM.empty())
        {
            boolScalesTM.clear(); // clear all elements of scales
        }
        // get Yolo data
        std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>> trackersYolo;
        trackersYolo.reserve(10); // get new data
        std::vector<cv::Mat1b> templatesYolo;
        templatesYolo.reserve(10); // get new data
        std::vector<cv::Rect2d> bboxesYolo;
        bboxesYolo.reserve(10); // get current frame data
        std::vector<int> classIndexesYolo;
        classIndexesYolo.reserve(150);
        //get Yolo data
        getYoloData(trackersYolo,templatesYolo, bboxesYolo, classIndexesYolo, queueYoloClassIndex, queueYoloBbox, queueTrackerYolo, queueYoloTemplate); // get new frame
        // combine Yolo and TM data, and update latest data
        combineYoloTMData(frame,classIndexesYolo, classIndexTM, trackersYolo, trackers_mosse, templatesYolo, templatesTM, bboxesYolo, bboxesTM, previousMove, updatedTrackers, updatedTemplates, updatedBboxes, updatedClasses, boolScalesTM, updatedMove, numTrackersTM);
    }

    void getYoloData(std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& newTrackers, std::vector<cv::Mat1b>& newTemplates, std::vector<cv::Rect2d>& newBboxes, std::vector<int>& newClassIndexes,
        std::queue<std::vector<int>>& queueYoloClassIndex, std::queue<std::vector<cv::Rect2d>>& queueYoloBbox, 
        std::queue<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>>& queueTrackerYolo, std::queue<std::vector<cv::Mat1b>>& queueYoloTemplate)
    {
        if (!queueYoloTemplate.empty())
        {
            newTemplates = queueYoloTemplate.front();
            queueYoloTemplate.pop();
        }
        if (!queueTrackerYolo.empty())
        {
            newTrackers = queueTrackerYolo.front();
            queueTrackerYolo.pop();
        }
        if (!queueYoloBbox.empty())
        {
            newBboxes = queueYoloBbox.front();
            queueYoloBbox.pop();
        }
        newClassIndexes = queueYoloClassIndex.front();
        queueYoloClassIndex.pop();
        //std::cout << "TM get data from Yolo :: class label :: size=" << newClassIndexes.size() << std::endl;;
        //for (const int& classIndex : newClassIndexes)
        //{
        //    std::cout << classIndex << " ";
        //}
        //std::cout << std::endl;
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

    void combineYoloTMData(cv::Mat1b& frame,std::vector<int>& classIndexesYolo, std::vector<int>& classIndexTM, 
        std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& trackersYolo, std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& trackers_mosse,
        std::vector<cv::Mat1b>& templatesYolo, std::vector<cv::Mat1b>& templatesTM,
        std::vector<cv::Rect2d>& bboxesYolo, std::vector<cv::Rect2d>& bboxesTM, std::vector<std::vector<int>>& previousMove, 
        std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers, std::vector<cv::Mat1b>& updatedTemplates, std::vector<cv::Rect2d>& updatedBboxes,
        std::vector<int>& updatedClasses, std::vector<bool>& boolScalesTM, std::vector<std::vector<int>>& updatedMove, const int& numTrackersTM)
    {
        int counterYolo = 0;
        int counterTM = 0;      // for counting TM adaptations
        int counterClassTM = 0; // for counting TM class counter
        // organize current situation : determine if tracker is updated with Yolo or TM, and is deleted
        // think about tracker continuity : tracker survival : (not Yolo Tracker) and (not TM tracker)
        int numPastLabels = classIndexTM.size();
        /* should check carefully -> compare num of detection */
        for (const int& classIndex : classIndexesYolo)
        {
            ///if (!classIndexTM.empty()) //when comment out comment out this line, too!
            //   std::cout <<"numPastLabels="<<numPastLabels<<", classIndexTM.size()="<<classIndexTM.size()<<", bboxesTM.size()="<<bboxesTM.size()<<"classIndexTM="<<classIndexTM[counterClassTM]<< ",classIndexesYolo.size()=" << classIndexesYolo.size() << ", classYolo:" << classIndex << ", bboxesYolo.size()=" << bboxesYolo.size() << "counterClassTM=" << counterClassTM << ", counterTM=" << counterTM << std::endl;
            /* after 2nd time from YOLO data */
            if (numPastLabels > 0)
            {
                //std::cout << "TM tracker already exist" << std::endl;
                /* first numTrackersTM is existed Templates -> if same label, update else, unless tracking was successful lost tracker */
                if (counterClassTM < numPastLabels) // numTrackersTM : num ber of class indexes
                {
                    /*update tracker*/
                    if (classIndex >= 0)
                    {
                        /* if classIndex != -1, update tracker. and if tracker of TM is successful, search aream can be limited */
                        if (classIndex == classIndexTM[counterClassTM])
                        {
                            //std::cout<<"classIndex="<<classIndexTM[counterClassTM]<<", tracker addess"<<trackers_mosse[counterTM]<<", template img size="<<
                            if (bool_check_psr) //check which tracker to adopt
                            {
                                //std::cout << "tracker psr=" << trackers_mosse[counterTM]->previous_psr << std::endl;
                                if ((trackers_mosse[counterTM]->previous_psr) >= min_keep_psr)
                                {
                                    //std::cout << "keep rameined tracker" << std::endl;
                                    updatedTrackers.push_back(trackers_mosse[counterTM]); // update template to YOLO's one
                                    updatedTemplates.push_back(templatesTM[counterTM]); // update template to YOLO's one
                                }
                                else
                                {
                                    updatedTrackers.push_back(trackersYolo[counterYolo]); // update template to YOLO's one
                                    updatedTemplates.push_back(templatesYolo[counterYolo]); // update template to YOLO's one
                                }
                            }
                            else
                            {
                                updatedTrackers.push_back(trackersYolo[counterYolo]); // update template to YOLO's one
                                updatedTemplates.push_back(templatesYolo[counterYolo]); // update template to YOLO's one
                            }
                            //updatedTrackers.push_back(trackersYolo[counterYolo]); // update template to YOLO's one
                            if (bool_TBD) updatedBboxes.push_back(bboxesTM[counterYolo]);// update bbox with Yolo data
                            else updatedBboxes.push_back(bboxesTM[counterTM]);          // update bbox with TM one
                            updatedClasses.push_back(classIndex);                       // update class
                            boolScalesTM.push_back(true);                               // scale is set to TM
                            updatedMove.push_back(previousMove[counterTM]);
                            counterTM++;
                            counterYolo++;
                            counterClassTM++;
                        }
                        /* trakcer of TM was failed */
                        else
                        {
                            updatedTrackers.push_back(trackersYolo[counterYolo]); // update template to YOLO's one
                            updatedTemplates.push_back(templatesYolo[counterYolo]); // update template to YOLO's one
                            updatedBboxes.push_back(bboxesYolo[counterYolo]);       // update bbox to YOLO's one
                            updatedClasses.push_back(classIndex);                       // update class to YOLO's one
                            boolScalesTM.push_back(false);                              // scale is set to Yolo
                            updatedMove.push_back(defaultMove);
                            counterYolo++;
                            counterClassTM++;
                        }
                    }
                    /* tracker not found in YOLO */
                    else
                    {
                        /* template matching was successful -> keep tracking */
                        if (classIndexTM[counterClassTM] >= 0)
                        {
                            updatedTrackers.push_back(trackers_mosse[counterTM]); // update tracker to TM's one
                            updatedTemplates.push_back(templatesTM[counterTM]); // update tracker to TM's one
                            updatedBboxes.push_back(bboxesTM[counterTM]);       // update bbox to TM's one
                            updatedClasses.push_back(classIndexTM[counterClassTM]);
                            boolScalesTM.push_back(true); // scale is set to TM
                            updatedMove.push_back(previousMove[counterTM]);
                            counterTM++;
                            counterClassTM++;
                        }
                        /* both tracking was failed -> lost */
                        else
                        {
                            updatedClasses.push_back(classIndex);
                            counterClassTM++;
                        }
                    }
                }
                /* new tracker -> add new templates * maybe in this case all calss labels should be positive, not -1 */
                else
                {
                    if (classIndex >= 0)
                    {
                        //std::cout << "add new tracker" << std::endl;
                        updatedTrackers.push_back(trackersYolo[counterYolo]); // update template to YOLO's one
                        updatedTemplates.push_back(templatesYolo[counterYolo]); // update template to YOLO's one
                        updatedBboxes.push_back(bboxesYolo[counterYolo]);       // update bbox to YOLO's one
                        updatedClasses.push_back(classIndex);                       // update class to YOLO's one
                        boolScalesTM.push_back(false);                              // scale is set to Yolo
                        updatedMove.push_back(defaultMove);
                        counterYolo++;
                    }
                    /* this is for exception, but prepare for emergency*/
                    else
                    {
                        //std::cout << "this is exception:: even if new tracker, class label is -1. Should revise code " << std::endl;
                        updatedClasses.push_back(classIndex);
                    }
                }
            }
            /* for the first time from YOLO data */
            else
            {
                //std::cout << "first time of TM" << std::endl;
                /* tracker was successful */
                if (classIndex >= 0)
                {
                    updatedTrackers.push_back(trackersYolo[counterYolo]); // update template to YOLO's one
                    updatedTemplates.push_back(templatesYolo[counterYolo]); // update template to YOLO's one
                    updatedBboxes.push_back(bboxesYolo[counterYolo]);       // update bbox to YOLO's one
                    updatedClasses.push_back(classIndex);                       // update class to YOLO's one
                    boolScalesTM.push_back(false);                              // scale is set to Yolo
                    updatedMove.push_back(defaultMove);
                    counterYolo++;
                }
                /* tracker was not found in YOLO */
                else
                {
                    updatedClasses.push_back(classIndex);
                }
            }
        }
        //IoU check
        if (bool_iouCheck)
        {
            if (updatedBboxes.size() >= 2)
            {
                //std::cout << " /////////////////////////// check Duplicated trackers" << std::endl;
                int counterLabels = 0;
                std::vector<int> labels_on; //successful tracker index
                for (int& label : updatedClasses) //gatcher successful trackers index
                {
                    if (label >= 0)
                    {
                        labels_on.push_back(counterLabels);
                    }
                    counterLabels++;//increment counter of templates
                }
                int counter_template = 0;
                cv::Rect2d roi_template;
                while (true)
                {
                    if (counter_template >= updatedBboxes.size() - 1) break;
                    roi_template = updatedBboxes[counter_template]; //base template
                    int i = counter_template + 1;
                    while (true)
                    {
                        if (i >= updatedBboxes.size()) break; //terminate condition
                        if (updatedClasses[labels_on[counter_template]] == updatedClasses[labels_on[i]]) //same label
                        {
                            float iou = calculateIoU_Rect2d(roi_template, updatedBboxes[i]);
                            //std::cout << "iou=" << iou << std::endl;
                            if (iou >= IoU_overlapped) //duplicated tracker -> delete template,roi and scales and convert class label to -1
                            {
                                std::cout << " //////////////////////////////// overlapped tracker: iou=" << iou << std::endl;
                                if (bool_augment)
                                {
                                    //augment tracker and delete new tracker
                                    double left = std::min(updatedBboxes[i].x, updatedBboxes[counter_template].x);
                                    double right = std::max((updatedBboxes[i].x + updatedBboxes[i].width), (updatedBboxes[counter_template].x + updatedBboxes[counter_template].width));
                                    double top = std::min(updatedBboxes[i].y, updatedBboxes[counter_template].y);
                                    double bottom = std::max((updatedBboxes[i].y + updatedBboxes[i].height), (updatedBboxes[counter_template].y + updatedBboxes[counter_template].height));
                                    if ((0 < left && left < right && right < frame.cols) && (0 < top && top < bottom && bottom < frame.rows))
                                    {
                                        cv::Rect2d newRoi(left, top, (right - left), (bottom - top));
                                        updatedBboxes[counter_template] = newRoi;
                                        updatedTemplates[counter_template] = frame(newRoi);
                                        cv::Ptr<cv::mytracker::TrackerMOSSE> tracker = cv::mytracker::TrackerMOSSE::create();
                                        updatedTrackers[counter_template] = tracker;
                                        updatedMove[counter_template][0] = (int)((updatedMove[counter_template][0] + updatedMove[i][0]) / 2);
                                        updatedMove[counter_template][1] = (int)((updatedMove[counter_template][1] + updatedMove[i][1]) / 2);
                                        //delete tracker
                                        updatedTrackers.erase(updatedTrackers.begin() + i);
                                        updatedTemplates.erase(updatedTemplates.begin() + i);
                                        updatedBboxes.erase(updatedBboxes.begin() + i);
                                        boolScalesTM.erase(boolScalesTM.begin() + i);
                                        updatedClasses[labels_on[i]] = -2;
                                        updatedMove.erase(updatedMove.begin() + i);
                                        labels_on.erase(labels_on.begin() + i);
                                    }
                                    else
                                    {
                                        if ((updatedTrackers[i]->previous_psr) >= (updatedTrackers[counter_template]->previous_psr))
                                        {
                                            //exchange trackere
                                            updatedTrackers[counter_template] = updatedTrackers[i];
                                            updatedTemplates[counter_template] = updatedTemplates[i];
                                            updatedBboxes[counter_template] = updatedBboxes[i];
                                            updatedMove[counter_template] = updatedMove[i];
                                        }
                                        updatedTrackers.erase(updatedTrackers.begin() + i);
                                        updatedTemplates.erase(updatedTemplates.begin() + i);
                                        updatedBboxes.erase(updatedBboxes.begin() + i);
                                        boolScalesTM.erase(boolScalesTM.begin() + i);
                                        updatedClasses[labels_on[i]] = -2;
                                        updatedMove.erase(updatedMove.begin() + i);
                                        labels_on.erase(labels_on.begin() + i);
                                    }
                                }
                                else
                                {
                                    if ((updatedTrackers[i]->previous_psr) >= (updatedTrackers[counter_template]->previous_psr))
                                    {
                                        //exchange trackere
                                        updatedTrackers[counter_template] = updatedTrackers[i];
                                        updatedTemplates[counter_template] = updatedTemplates[i];
                                        updatedBboxes[counter_template] = updatedBboxes[i];
                                        updatedMove[counter_template] = updatedMove[i];
                                    }
                                    updatedTrackers.erase(updatedTrackers.begin() + i);
                                    updatedTemplates.erase(updatedTemplates.begin() + i);
                                    updatedBboxes.erase(updatedBboxes.begin() + i);
                                    boolScalesTM.erase(boolScalesTM.begin() + i);
                                    updatedClasses[labels_on[i]] = -2;
                                    updatedMove.erase(updatedMove.begin() + i);
                                    labels_on.erase(labels_on.begin() + i);
                                }
                                
                            }
                            else i++;
                        }
                        else i++; //other labels
                    }
                    counter_template++;
                }
            }
        }
    }

    void process(std::vector<int>& updatedClasses, std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers, std::vector<cv::Mat1b>& updatedTemplates, std::vector<bool>& boolScalesTM, 
        std::vector<cv::Rect2d>& updatedBboxes, std::vector<std::vector<int>>& updatedMove, cv::Mat1b& img,
        std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers_mosse, std::vector<cv::Mat1b>& updatedTemplatesTM, 
        std::vector<cv::Rect2d>& updatedBboxesTM, std::vector<int>& updatedClassesTM, std::vector<bool>& updatedSearchScales, std::vector<std::vector<int>>& updatedMoveTM)
    {
        int counterTracker = 0; // counter for number of tracker
        int counter = 0;        // counter for all classes
        // iterate for each tracking classes
        /*std::cout << "check updateClasses: ";
        for (const int& classIndex : updatedClasses)
        {
            std::cout << classIndex << " ";
        }
        std::cout << std::endl;
        */
        if (bool_multithread)
        {
            std::vector<std::thread> threadTracking; // prepare threads
            for (const int& classIndexTM : updatedClasses)
            {
                if (classIndexTM >= 0)
                {
                    //template matching
                    threadTracking.emplace_back(&TemplateMatching::matchingTemplate, this, classIndexTM, counterTracker, counter,
                        std::ref(updatedTemplates), std::ref(boolScalesTM), std::ref(updatedBboxes), std::ref(updatedMove), std::ref(img),
                        std::ref(updatedTrackers_mosse), std::ref(updatedTemplatesTM), std::ref(updatedBboxesTM), std::ref(updatedClassesTM), std::ref(updatedSearchScales), std::ref(updatedMoveTM));
                    //MOSSE
                    threadTracking.emplace_back(&TemplateMatching::track_mosse, this, classIndexTM, counterTracker, counter,
                        std::ref(updatedTrackers), std::ref(updatedTemplates),std::ref(boolScalesTM), std::ref(updatedBboxes), std::ref(updatedMove), std::ref(img),
                        std::ref(updatedTrackers_mosse), std::ref(updatedTemplatesTM), std::ref(updatedBboxesTM), std::ref(updatedClassesTM), std::ref(updatedSearchScales), std::ref(updatedMoveTM));
                    // std::cout << counter << "-th thread has started" << std::endl;
                    counterTracker ++;
                }
                else if (classIndexTM == -2)
                    updatedClassesTM[counter] = -2;
                counter ++;
            }
            int counterThread = 0;
            if (!threadTracking.empty())
            {
                for (std::thread& thread : threadTracking)
                {
                    thread.join();
                    counterThread++;
                }
                //std::cout << counterThread << " threads have finished!" << std::endl;
            }
            else
            {
                //std::cout << "no thread has started" << std::endl;
            }
        }
        else//run in one thread
        {
            for (const int& classIndexTM : updatedClasses)
            {
                if (classIndexTM >= 0)
                {
                    //template matching
                    std::thread threadTM(&TemplateMatching::matchingTemplate,this,classIndexTM, counterTracker, counter, std::ref(updatedTemplates), std::ref(boolScalesTM), 
                        std::ref(updatedBboxes), std::ref(updatedMove), std::ref(img),std::ref(updatedTrackers_mosse), std::ref(updatedTemplatesTM), std::ref(updatedBboxesTM), std::ref(updatedClassesTM), std::ref(updatedSearchScales), std::ref(updatedMoveTM));
                    //mosse
                    track_mosse(classIndexTM, counterTracker, counter, updatedTrackers, updatedTemplates,boolScalesTM, updatedBboxes, updatedMove, img, updatedTrackers_mosse, updatedTemplatesTM,updatedBboxesTM, updatedClassesTM, updatedSearchScales, updatedMoveTM);
                    threadTM.join();
                    counterTracker ++;
                }
                else if (classIndexTM == -2)
                    updatedClassesTM[counter] = -2;
                counter++;
            }
        }
        // organize data -> if templates not updated -> erase data]
        int counter_tracker = 0;
        /*std::cout << "updated data result :: ";
        for (int i = 0; i < updatedClassesTM.size(); i++)
        {
            if (updatedClassesTM[i] >= 0)
            {
                std::cout << "classlabel=" << updatedClasses[i] << std::endl;
                std::cout << " bbox = " << updatedBboxesTM[counter_tracker];
                std::cout << ", updatedTemplates.rows = " << updatedTemplatesTM[counter_tracker].rows;
                std::cout << ", !tracker.empty()" << updatedTrackers_mosse[counter_tracker];
                std::cout << ", updatedSearchScales = " << updatedSearchScales[counter_tracker];
                std::cout << ", updatedMoveTM = " << updatedMoveTM[counter_tracker] << std::endl;
                counter_tracker++;
            }
        }
        */
        int counterCheck = 0;
        int counter_label = 0;
        //std::cout << "Before deleting : : updatedBboxesTM.size()=" << updatedBboxesTM.size() << std::endl;
        while (true)
        {
            if (counterCheck >= updatedBboxesTM.size()) break;
            //std::cout << "counterCheck=" << counterCheck << ", updatedBboxesTM.size()=" << updatedBboxesTM.size() <<"updatedTemplatesTM.size()="<<updatedTemplatesTM.size()<<", updatedTrackers_mosse.size()"<<updatedTrackers_mosse.size()<<", updatedSearchScalesTM.size()="<<updatedSearchScales.size()<<", updatedMove.size()"<<updatedMoveTM.size() << std::endl;
            // not updated
            //std::cout <<"tracker address : " << updatedTrackers_mosse[counterCheck] << std::endl;
            if (updatedBboxesTM[counterCheck].width <= 0)
            {
                //std::cout << "delete data" << std::endl;
                updatedTrackers_mosse.erase(updatedTrackers_mosse.begin() + counterCheck);
                updatedTemplatesTM.erase(updatedTemplatesTM.begin() + counterCheck);
                updatedBboxesTM.erase(updatedBboxesTM.begin() + counterCheck);
                updatedSearchScales.erase(updatedSearchScales.begin() + counterCheck);
                updatedMoveTM.erase(updatedMoveTM.begin() + counterCheck);
                if (updatedClassesTM[counter_label] >= 0) updatedClassesTM[counter_label] = -1;
                //std::cout << "finish deleting" << std::endl;
            }
            else
                counterCheck++;
            counter_label++;
            //std::cout << "counterCheck=" << counterCheck << ", counter_label=" << counter_label << std::endl;
        }
        //std::cout << "after deleting : : updatedBboxesTM.size()=" << updatedBboxesTM.size() << std::endl;
    }
   
    void matchingTemplate(const int classIndexTM, int counterTracker, int counter, 
        std::vector<cv::Mat1b>& updatedTemplates, std::vector<bool>& boolScalesTM, std::vector<cv::Rect2d>& updatedBboxes, std::vector<std::vector<int>>& updatedMove, cv::Mat1b& img,
        std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers_mosse, std::vector<cv::Mat1b>& updatedTemplatesTM, 
        std::vector<cv::Rect2d>& updatedBboxesTM, std::vector<int>& updatedClassesTM, std::vector<bool>& updatedSearchScales, std::vector<std::vector<int>>& updatedMoveTM)
    {
        int leftSearch, topSearch, rightSearch, bottomSearch;
        const cv::Mat1b& templateImg = updatedTemplates[counterTracker]; // for saving memory, using reference data
        std::vector<int> previousMove = updatedMove[counterTracker];
        int deltaX_past = previousMove[0]; int deltaY_past = previousMove[1];
        cv::Rect2d previousRoi = updatedBboxes[counterTracker];
        //std::cout << "TM :: previousMove :: deltaX = " << deltaX_past << ", deltaY_past = " << deltaY_past << std::endl;
        if (boolScalesTM[counterTracker]) // scale is set to TM : smaller search area
        {
            double scale_x = scaleXTM;
            double scale_y = scaleYTM;
            if (bool_dynamicScale)
            {
                scale_x = static_cast<double>(std::max(std::min((1.1 + (std::abs(deltaX_past) / previousRoi.width)), scaleXYolo), scaleXTM));
                scale_y = static_cast<double>(std::max(std::min((1.1 + (std::abs(deltaY_past) / previousRoi.height)), scaleYYolo), scaleYTM));
            }
            leftSearch = std::min(std::max(0, static_cast<int>(previousRoi.x + deltaX_past - (scale_x - 1) * previousRoi.width / 2)), img.cols);
            topSearch = std::min(std::max(0, static_cast<int>(previousRoi.y + deltaY_past - (scale_y - 1) * previousRoi.height / 2)),img.rows);
            rightSearch = std::max(0,std::min(img.cols, static_cast<int>(previousRoi.x + deltaX_past + (scale_x + 1) * previousRoi.width / 2)));
            bottomSearch = std::max(0,std::min(img.rows, static_cast<int>(previousRoi.y + deltaY_past + (scale_y+ 1) * previousRoi.height / 2)));
        }
        else // scale is set to YOLO : larger search area
        {
            leftSearch = std::min(img.cols,std::max(0, static_cast<int>(previousRoi.x + deltaX_past - (scaleXYolo - 1) * previousRoi.width / 2)));
            topSearch = std::min(img.rows,std::max(0, static_cast<int>(previousRoi.y + deltaY_past - (scaleYYolo - 1) * previousRoi.height / 2)));
            rightSearch = std::max(0,std::min(img.cols, static_cast<int>(previousRoi.x + deltaX_past + (scaleXYolo + 1) * previousRoi.width / 2)));
            bottomSearch = std::max(0,std::min(img.rows, static_cast<int>(previousRoi.y + deltaY_past + (scaleYYolo + 1) * previousRoi.height / 2)));
        }
        if ((rightSearch - leftSearch) > MIN_SEARCH && (bottomSearch - topSearch) > MIN_SEARCH)
        {
            cv::Rect2d searchArea(leftSearch, topSearch, (rightSearch - leftSearch), (bottomSearch - topSearch));
            //std::cout << "img size : width = " << img.cols << ", height = " << img.rows << std::endl;
            //std::cout << "croppdeImg size: left=" << searchArea.x << ", top=" << searchArea.y << ", width=" << searchArea.width << ", height=" << searchArea.height << std::endl;
            cv::Mat1b croppedImg = img.clone();
            croppedImg = croppedImg(searchArea); // crop img
            cv::Mat result; // for saving template matching results
            int result_cols = croppedImg.cols - templateImg.cols + 1;
            int result_rows = croppedImg.rows - templateImg.rows + 1;
            //std::cout << "result_cols :" << result_cols << ", result_rows:" << result_rows << std::endl;
            // template seems to go out of frame
            if (result_cols <= 0 || result_rows <= 0)
            {
                //std::cout << "template seems to go out from frame" << std::endl;
            }
            else
            {
                //std::cout << "croppedImg :: left=" << leftSearch << ", top=" << topSearch << ", right=" << rightSearch << ", bottom=" << bottomSearch << std::endl;
                result.create(result_rows, result_cols, CV_32FC1); // create result array for matching quality+
                // const char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED"; 2 is not so good
                cv::matchTemplate(croppedImg, templateImg, result, MATCHINGMETHOD); // template Matching
                //std::cout << "finish matchTemplate" << std::endl;
                double minVal;    // minimum score
                double maxVal;    // max score
                cv::Point minLoc; // minimum score left-top points
                cv::Point maxLoc; // max score left-top points
                cv::Point matchLoc;

                cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat()); // In C++, we should prepare type-defined box for returns, which is usually pointer
                //std::cout << "matching template :: score :: minVal = " << minVal << ", maxVal=" << maxVal << std::endl;
                // std::cout << "minmaxLoc: " << maxVal << std::endl;

                // meet matching criteria :: setting bbox
                // std::cout << "max value : " << maxVal << std::endl;
                /* find matching object */
                if ((MATCHINGMETHOD == cv::TM_SQDIFF_NORMED && minVal <= matchingThreshold) || ((MATCHINGMETHOD == cv::TM_CCOEFF_NORMED || MATCHINGMETHOD == cv::TM_CCORR_NORMED) && maxVal >= matchingThreshold))
                {
                    if (MATCHINGMETHOD == cv::TM_SQDIFF || MATCHINGMETHOD == cv::TM_SQDIFF_NORMED)
                    {
                        matchLoc = minLoc;
                    }
                    else
                    {
                        matchLoc = maxLoc;
                    }
                    int leftRoi, topRoi, rightRoi, bottomRoi;
                    cv::Mat1b newTemplate;
                    cv::Rect2d roi,roi_cropped;
                    leftRoi = std::min(img.cols,std::max(0, static_cast<int>(matchLoc.x + leftSearch)));
                    topRoi = std::min(img.rows,std::max(0, static_cast<int>(matchLoc.y + topSearch)));
                    rightRoi = std::max(0,std::min(img.cols, static_cast<int>(leftRoi + templateImg.cols)));
                    bottomRoi =std::max(0, std::min(img.rows, static_cast<int>(topRoi + templateImg.rows)));
                    // update roi
                    roi.x = leftRoi;
                    roi.y = topRoi;
                    roi.width = rightRoi - leftRoi;
                    roi.height = bottomRoi - topRoi;
                    if (roi.width > 0 && roi.height > 0)
                    {
                        double deltaX_current = (roi.x + roi.width / 2) - (previousRoi.x + previousRoi.width / 2); double deltaY_current = (roi.y + roi.height / 2) - (previousRoi.y + previousRoi.height / 2);
                        //std::cout << "TM : current :: deltaX = " << deltaX_current << ", deltaY = " << deltaY_current << std::endl;
                        /* moving constraints */
                        if (std::pow(deltaX_current, 2) + std::pow(deltaY_current, 2) >= MoveThreshold)
                        {
                            // update information
                            if (updatedBboxesTM.at(counterTracker).x == -1) //default data
                            {
                                //update roi with template matching result
                                cv::Ptr<cv::mytracker::TrackerMOSSE>& tracker = cv::mytracker::TrackerMOSSE::create();
                                //roi_cropped.x = std::max(0, static_cast<int>(matchLoc.x));
                                //roi_cropped.y = std::max(0, static_cast<int>(matchLoc.y));
                                //roi_cropped.width = templateImg.cols;
                                //roi_cropped.height = templateImg.rows;
                                tracker->init(img, roi); //init tracker
                                //std::cout << "TM :: finish initing tracker" << std::endl;
                                //finish mosse
                                if (updatedBboxesTM.at(counterTracker).x == -1) updatedTemplatesTM.at(counterTracker) = img(roi);
                                if (updatedBboxesTM.at(counterTracker).x == -1) updatedClassesTM.at(counter) = classIndexTM; // only class labels updated with counter index
                                if (updatedBboxesTM.at(counterTracker).x == -1) updatedSearchScales.at(counterTracker) = true;
                                //std::cout << "TM :: pushing data" << std::endl;
                                int deltaX_future, deltaY_future;
                                if ((gamma * deltaX_current + (1 - gamma) * (double)deltaX_past) <= 0) //negative
                                    deltaX_future = std::max((int)(gamma * deltaX_current + (1 - gamma) * (double)deltaX_past), -MAX_VEL);
                                else //positive
                                    deltaX_future = std::min((int)(gamma * deltaX_current + (1 - gamma) * (double)deltaX_past), MAX_VEL);
                                if ((gamma * deltaY_current + (1 - gamma) * (double)deltaY_past) <= 0) //negative
                                    deltaY_future = std::max((int)(gamma * deltaY_current + (1 - gamma) * (double)deltaY_past), -MAX_VEL);
                                else //positive
                                    deltaY_future = std::min((int)(gamma * deltaY_current + (1 - gamma) * (double)deltaY_past), MAX_VEL);
                                if (updatedBboxesTM.at(counterTracker).x == -1) updatedMoveTM.at(counterTracker) = std::vector<int>{ deltaX_future,deltaY_future };
                                if (updatedBboxesTM.at(counterTracker).x == -1) updatedTrackers_mosse.at(counterTracker) = tracker;
                                //std::cout << "TM :: push init tracker" << std::endl;
                                if (updatedBboxesTM.at(counterTracker).x == -1) updatedBboxesTM.at(counterTracker) = roi;
                                //std::cout << "roi.x=" << roi.x << ", roi.y=" << roi.y << std::endl;
                                //std::cout << "  ------- succeed in tracking with TM ------------------ " << std::endl;
                            }
                            //else
                            //std::cout << "TM didn't push data" << std::endl;
                        }
                    }
                }
            }
        }
    }

    void track_mosse(const int classIndexTM, int counterTracker, int counter,
        std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers, std::vector<cv::Mat1b>& updatedTemplates, std::vector<bool>& boolScalesTM, std::vector<cv::Rect2d>& updatedBboxes,
        std::vector<std::vector<int>>& updatedMove, cv::Mat1b& img,
        std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers_mosse, std::vector<cv::Mat1b>& updatedTemplatesTM, 
        std::vector<cv::Rect2d>& updatedBboxesTM, std::vector<int>& updatedClassesTM, std::vector<bool>& updatedSearchScales, std::vector<std::vector<int>>& updatedMoveTM)
    {
        /* template exist -> start template matching */
        // get bbox from queue for limiting search area
        int leftSearch, topSearch, rightSearch, bottomSearch;
        cv::Ptr<cv::mytracker::TrackerMOSSE>& tracker = updatedTrackers[counterTracker];
        std::vector<int> previousMove = updatedMove[counterTracker];
        int deltaX_past = previousMove[0]; int deltaY_past = previousMove[1];
        cv::Rect2d previousRoi = updatedBboxes[counterTracker];
        //std::cout << "MOSSE :: previousMove :: deltaX = " << deltaX_past << ", deltaY = " << deltaY_past << std::endl;
        if (boolScalesTM[counterTracker]) // scale is set to TM : smaller search area
        {
            double scale_x = scaleXTM;
            double scale_y = scaleYTM;
            if (bool_dynamicScale)
            {
                scale_x = static_cast<double>(std::max(std::min((1.1 + (deltaX_past / previousRoi.width)), scaleXYolo), scaleXTM));
                scale_y = static_cast<double>(std::max(std::min((1.1 + (deltaY_past / previousRoi.height)), scaleYYolo), scaleYTM));
            }
            leftSearch = std::min(img.cols,std::max(0, static_cast<int>(previousRoi.x + deltaX_past - (scale_x - 1) * previousRoi.width / 2)));
            topSearch = std::min(img.rows,std::max(0, static_cast<int>(previousRoi.y + deltaY_past - (scale_y - 1) * previousRoi.height / 2)));
            rightSearch = std::max(0,std::min(img.cols, static_cast<int>(previousRoi.x + deltaX_past + (scale_x + 1) * previousRoi.width / 2)));
            bottomSearch = std::max(0,std::min(img.rows, static_cast<int>(previousRoi.y + deltaY_past + (scale_y + 1) * previousRoi.height / 2)));
        }
        else // scale is set to YOLO : larger search area
        {
            leftSearch = std::min(std::max(0, static_cast<int>(previousRoi.x + deltaX_past - (scaleXYolo - 1) * previousRoi.width / 2)), img.cols);
            topSearch = std::min(std::max(0, static_cast<int>(previousRoi.y + deltaY_past - (scaleYYolo - 1) * previousRoi.height / 2)), img.rows);
            rightSearch = std::min(img.cols, static_cast<int>(previousRoi.x + deltaX_past + (scaleXYolo + 1) * previousRoi.width / 2));
            bottomSearch = std::min(img.rows, static_cast<int>(previousRoi.y + deltaY_past + (scaleYYolo + 1) * previousRoi.height / 2));
        }
        if ((rightSearch - leftSearch) > 0 && (bottomSearch - topSearch) > 0)
        {
            cv::Rect2d searchArea(leftSearch, topSearch, (rightSearch - leftSearch), (bottomSearch - topSearch));
            //std::cout << "img size : width = " << img.cols << ", height = " << img.rows << std::endl;
            //std::cout << "croppdeImg size: left=" << searchArea.x << ", top=" << searchArea.y << ", width=" << searchArea.width << ", height=" << searchArea.height << std::endl;
            cv::Mat1b croppedImg = img.clone();
            croppedImg = croppedImg(searchArea); // crop img
            //convert roi from image coordinate to local search area coordinate
            cv::Rect2d croppedRoi;
            croppedRoi.x = previousRoi.x - searchArea.x;
            croppedRoi.y = previousRoi.y - searchArea.y;
            croppedRoi.width = previousRoi.width;
            croppedRoi.height = previousRoi.height;
            // MOSSE Tracker
            double psr = tracker->update(croppedImg, croppedRoi, updatedMove[counterTracker], true, bool_skip, threshold_mosse);
            //std::cout << "MOSSE :: PSR=" << psr << std::endl;
            //tracking was successful
            if (psr > threshold_mosse)
            {
                cv::Rect2d newRoi;
                int leftRoi = std::min(std::max(0, static_cast<int>(croppedRoi.x + leftSearch)), img.cols);
                int topRoi = std::min(std::max(0, static_cast<int>(croppedRoi.y + topSearch)), img.rows);
                int rightRoi = std::max(std::min(img.cols, static_cast<int>(leftRoi + croppedRoi.width)), 0);
                int bottomRoi = std::max(std::min(img.rows, static_cast<int>(topRoi + croppedRoi.height)), 0);
                newRoi.x = leftRoi; newRoi.y = topRoi; newRoi.width = rightRoi - leftRoi; newRoi.height = bottomRoi - topRoi;
                if (newRoi.width > 0 && newRoi.height > 0)
                {
                    double deltaX_current = (newRoi.x + newRoi.width / 2) - (previousRoi.x + previousRoi.width / 2); double deltaY_current = (newRoi.y + newRoi.height / 2) - (previousRoi.y + previousRoi.height / 2);
                    //std::cout << "MOSSE : current :: deltaX = " << deltaX_current << ", deltaY = " << deltaY_current << std::endl;
                    /* moving constraints */
                    if (std::pow(deltaX_current, 2) + std::pow(deltaY_current, 2) >= MoveThreshold)
                    {
                        // update information
                        updatedBboxesTM.at(counterTracker) = newRoi;
                        updatedTrackers_mosse.at(counterTracker) = tracker;
                        if (deltaX_current == deltaX_past && deltaY_current == deltaY_current)
                            updatedTemplatesTM.at(counterTracker) = updatedTemplates[counterTracker];
                        else
                            updatedTemplatesTM.at(counterTracker) = img(newRoi); //update templates
                        updatedClassesTM.at(counter) = classIndexTM; // only class labels updated with counter index
                        updatedSearchScales.at(counterTracker) = true;
                        //std::cout << "MOSSE :: pushing data" << std::endl;
                        int deltaX_future, deltaY_future;
                        if ((gamma * deltaX_current + (1 - gamma) * (double)deltaX_past) <= 0) //negative
                            deltaX_future = std::max((int)(gamma * deltaX_current + (1 - gamma) * (double)deltaX_past), -MAX_VEL);
                        else //positive
                            deltaX_future = std::min((int)(gamma * deltaX_current + (1 - gamma) * (double)deltaX_past), MAX_VEL);
                        if ((gamma * deltaY_current + (1 - gamma) * (double)deltaY_past) <= 0) //negative
                            deltaY_future = std::max((int)(gamma * deltaY_current + (1 - gamma) * (double)deltaY_past), -MAX_VEL);
                        else //positive
                            deltaY_future = std::min((int)(gamma * deltaY_current + (1 - gamma) * (double)deltaY_past), MAX_VEL);
                        updatedMoveTM.at(counterTracker) = std::vector<int>{ deltaX_future,deltaY_future };
                        //std::cout << "  ------- succeed in tracking with MOSSE ------------------ " << std::endl;
                        //std::cout << "roi.x=" << newRoi.x << ",roi.y=" << newRoi.y << ",roi.width=" << newRoi.width << ",roi.height=" << newRoi.height << std::endl;
                    }
                }
            }
        }

    }

};

#endif