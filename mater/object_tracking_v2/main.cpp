#include "stdafx.h"
#include "yolo_batch.h"
#include "tracker.h"
#include "utility.h"
#include "sequence.h"
#include "prediction.h"
#include "global_parameters.h"
#include "triangulation.h"
#include "mosse.h"


//saveFile
extern const std::string file_yolo_bbox_left;
extern const std::string file_yolo_class_left;
extern const std::string file_yolo_bbox_right;
extern const std::string file_yolo_class_right;

// camera : constant setting
extern const int LEFT_CAMERA;
extern const int RIGHT_CAMERA;

// tracker
extern const bool boolMOSSE;
extern const double threshold_mosse;

// declare function
/* yolo detection */
void yoloDetect();

void yoloDetect()
{
    /* Yolo Detection Thread
     * Args:
     *   queueFrame : frame
     *   queueFrameIndex : frame index
     *   queueYoloTemplate : detected template img
     *   queueYoloBbox : detected template bbox
     */

    Utility utYolo;
    YOLODetect_batch yolo;
    float t_elapsed = 0;

    //left
    std::vector<std::vector<cv::Rect2d>> posSaverYoloLeft;
    std::vector<std::vector<int>> classSaverYoloLeft;
    //right
    std::vector<std::vector<cv::Rect2d>> posSaverYoloRight;
    std::vector<std::vector<int>> classSaverYoloRight;
    //detectedFrame
    //left
    std::vector<int> detectedFrameLeft;
    std::vector<int> detectedFrameClassLeft;
    //right
    std::vector<int> detectedFrameRight;
    std::vector<int> detectedFrameClassRight;
    // vector for saving position

    int frameIndex;
    int countIteration = 0;
    /* while queueFrame is empty wait until img is provided */
    int counterFinish = 0; // counter before finish
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    bool bool_startTracker = false;
    while (true)
    {
        bool boolImgs = false;
        std::array<cv::Mat1b, 2> frames;
        int frameIndex;
        if (!queueFrame.empty())
        {
            frames = queueFrame.front();
            frameIndex = queueFrameIndex.front();
            queueFrame.pop();
            queueFrameIndex.pop();
            boolImgs = true;
        }
        //bool boolImgs = utYolo.getImagesFromQueueYolo(frames, frameIndex);
        if (!boolImgs)
        {
            if (counterFinish > 10)
            {
                break;
            }
            // No more frames in the queue, exit the loop
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            std::cout << "Yolo :: remaining counter until finish is :" << 10 - counterFinish << std::endl;
            counterFinish++;
            continue;
        }
        if (frames[LEFT_CAMERA].rows > 0 && frames[RIGHT_CAMERA].rows > 0)
        {
            counterFinish = 0;
            //concatenate 2 imgs horizontally
            auto start_pre = std::chrono::high_resolution_clock::now();
            cv::Mat1b concatFrame;
            cv::hconcat(frames[LEFT_CAMERA], frames[RIGHT_CAMERA], concatFrame);
            auto stop_pre = std::chrono::high_resolution_clock::now();
            auto duration_pre = std::chrono::duration_cast<std::chrono::milliseconds>(stop_pre - start_pre);
            /*start yolo detection */
            auto start = std::chrono::high_resolution_clock::now();
            yolo.detect(concatFrame, frameIndex, posSaverYoloLeft, posSaverYoloRight, classSaverYoloLeft, classSaverYoloRight,
                detectedFrameLeft, detectedFrameRight, detectedFrameClassLeft, detectedFrameClassRight, countIteration,bool_startTracker);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            t_elapsed = t_elapsed + static_cast<float>(duration.count());
            countIteration++;
            std::cout << "Yolo -- " << countIteration << " --" << std::endl;
            //std::cout << "#######" << countIteration << "---- Time taken by YOLO detection : " << duration.count() << " milliseconds #################" << std::endl;
        }
    }
    if (countIteration != 0) std::cout << " Yolo detection process speed :: " << static_cast<int>(countIteration / t_elapsed * 1000) << " Hz for " << countIteration << " cycles" << std::endl;
    /* check data */
    std::cout << "position saver : Yolo : " << std::endl;
    std::cout << " : Left : " << std::endl;
    std::cout << "posSaverYoloLeft size:" << posSaverYoloLeft.size() << ", detectedFrame size:" << detectedFrameLeft.size() << std::endl;
    utYolo.checkStorage(posSaverYoloLeft, detectedFrameLeft, file_yolo_bbox_left);
    std::cout << "classSaverYoloLeft size:" << classSaverYoloLeft.size() << ", detectedFrameClass size:" << detectedFrameClassLeft.size() << std::endl;
    utYolo.checkClassStorage(classSaverYoloLeft, detectedFrameClassLeft, file_yolo_class_left);
    std::cout << " : Right : " << std::endl;
    std::cout << "posSaverYoloRight size:" << posSaverYoloRight.size() << ", detectedFrame size:" << detectedFrameRight.size() << std::endl;
    utYolo.checkStorage(posSaverYoloRight, detectedFrameRight, file_yolo_bbox_right);
    std::cout << "classSaverYoloLeft size:" << classSaverYoloRight.size() << ", detectedFrameClass size:" << detectedFrameClassRight.size() << std::endl;
    utYolo.checkClassStorage(classSaverYoloRight, detectedFrameClassRight, file_yolo_class_right);
}


/* main function */
int main()
{
    /* video inference */
    //constructor 
    Utility ut;
    TemplateMatching tm;
    Sequence seq;
    Triangulation tri;
    Prediction predict;

    /* video inference */;
    cv::VideoCapture capture_left(filename_left);
    if (!capture_left.isOpened())
    {
        // error in opening the video input
        std::cerr << "Unable to open left file!" << std::endl;
        return 0;
    }
    cv::VideoCapture capture_right(filename_right);
    if (!capture_right.isOpened())
    {
        // error in opening the video input
        std::cerr << "Unable to open right file!" << std::endl;
        return 0;
    }
    int counter = 0;

    // multi thread code
    std::thread threadYolo(yoloDetect);
    std::cout << "start Yolo thread" << std::endl;
    std::thread threadTemplateMatching(&TemplateMatching::main, tm,std::ref(q_startTracker));
    std::cout << "start template matching thread" << std::endl;
    //std::thread threadRemoveImg(&Utility::removeFrame, ut);
    //std::cout << "remove frame has started" << std::endl;
    std::thread threadSeq(&Sequence::main, seq);
    //std::thread threadTri(&Triangulation::main, tri);
    //std::cout << "start triangulation thread" << std::endl;
    //std::thread threadPred(&Prediction::main, predict);
    //std::cout << "start prediction thread" << std::endl;


    while (true)
    {
        // Read the next frame
        cv::Mat frame_left, frame_right;
        capture_left >> frame_left;
        capture_right >> frame_right;
        counter++;
        if (frame_left.empty() || frame_right.empty())
            break;
        cv::Mat1b frameGray_left, frameGray_right;
        cv::cvtColor(frame_left, frameGray_left, cv::COLOR_RGB2GRAY);
        cv::cvtColor(frame_right, frameGray_right, cv::COLOR_RGB2GRAY);
        std::array<cv::Mat1b, 2> frames = { frameGray_left,frameGray_right };
        // cv::Mat1b frameGray;
        //  cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
        ut.pushFrame(frames, counter);
    }

    // std::thread threadTargetPredict(targetPredict);
    threadYolo.join();
    threadTemplateMatching.join();
    //threadRemoveImg.join();
    threadSeq.join();
    //threadTri.join();
    //threadPred.join();
    return 0;
}