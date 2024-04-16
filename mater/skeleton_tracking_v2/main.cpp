#include "stdafx.h"
#include "yolo_batch.h"
#include "optflow.h"
#include "utility.h"
#include "triangulation.h"
#include "global_parameters.h"

extern std::queue<std::array<cv::Mat1b, 2>> queueFrame;
extern std::queue<int> queueFrameIndex;

//queue
extern std::queue<Yolo2optflow> q_yolo2optflow_left, q_yolo2optflow_right;
extern std::queue<Optflow2optflow> q_optflow2optflow_left, q_optflow2optflow_right;

/*3D position*/
extern std::queue<std::vector<std::vector<std::vector<int>>>> queueTriangulation_left;
extern std::queue<std::vector<std::vector<std::vector<int>>>> queueTriangulation_right;

/* constant valude definition */
extern const std::string filename_left;
extern const std::string filename_right;
extern const bool save;
extern const bool boolSparse;
extern const bool boolGray;
extern const bool boolBatch;
extern const int LEFT;
extern const int RIGHT;
extern const std::string methodDenseOpticalFlow; //"lucasKanade_dense","rlof"
extern const float qualityCorner;
/* roi setting */
extern const int roiWidthOF;
extern const int roiHeightOF;
extern const int roiWidthYolo;
extern const int roiHeightYolo;
extern const float MoveThreshold;
extern const float epsironMove;
/* dense optical flow skip rate */
extern const int skipPixel;
/*if exchange template of Yolo */
extern const bool boolChange;

/* save date */
extern const std::string file_yolo_left;
extern const std::string file_yolo_right;
extern const std::string file_of_left;
extern const std::string file_of_right;

std::queue<bool> q_startOptflow;

/* Declaration of function */
void yolo();
void denseOpticalFlow(std::queue<bool>&);

void yolo()
{
    /* constructor of YOLOPoseEstimator */
    //if (boolBatch) 
    YOLOPoseBatch yolo;
    //else YOLOPose yolo_left, yolo_right;
    Utility utyolo;
    int count_yolo = 0;
    float t_elapsed = 0;
    /* prepare storage */
    std::vector<std::vector<std::vector<std::vector<int>>>> posSaver_left; //[sequence,numHuman,joints,element] :{frameIndex,xCenter,yCenter}
    std::vector<std::vector<std::vector<std::vector<int>>>> posSaver_right; //[sequence,numHuman,joints,element] :{frameIndex,xCenter,yCenter}
    posSaver_left.reserve(300);
    posSaver_right.reserve(300);
    if (queueFrame.empty())
    {
        while (queueFrame.empty())
        {
            if (!queueFrame.empty())
            {
                break;
            }
            //std::cout << "wait for images" << std::endl;
        }
    }
    /* frame is available */
    else
    {
        int counter = 1;
        int counterFinish = 0;
        //auto start_whole = std::chrono::high_resolution_clock::now();
        while (true)
        {
            //auto stop_whole = std::chrono::high_resolution_clock::now();
            //auto duration_whole = std::chrono::duration_cast<std::chrono::seconds>(stop_whole - start_whole);
            //if ((float)duration_whole.count() > 30.0) break;
            if (counterFinish == 10)
            {
                break;
            }
            /* frame can't be available */
            if (queueFrame.empty())
            {
                counterFinish++;
                /* waiting */
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            /* frame available -> start yolo pose estimation */
            else
            {
                counterFinish = 0;
                std::array<cv::Mat1b, 2> frames;
                int frameIndex;
                auto start = std::chrono::high_resolution_clock::now();
                utyolo.getImages(frames, frameIndex);
                //auto stop_getImg = std::chrono::high_resolution_clock::now();
                //auto duration_get = std::chrono::duration_cast<std::chrono::milliseconds>(stop_getImg - start);
                //t_elapsed = t_elapsed + static_cast<float>(duration_get.count());
                //count_yolo++;
                //std::cout << " ### Time taken by getting img : " << duration_get.count() << " milliseconds ### " << std::endl;
                //if (boolBatch)
                //{
                cv::Mat1b concatFrame;
                //std::cout << "frames[LEFT]:" << frames[LEFT].rows << "," << frames[LEFT].cols << ", frames[RIGHT]:" << frames[RIGHT].rows << "," << frames[RIGHT].cols << std::endl;
                if (frames[LEFT].rows > 0 && frames[RIGHT].rows > 0)
                {
                    cv::hconcat(frames[LEFT], frames[RIGHT], concatFrame);//concatenate 2 imgs horizontally
                    yolo.detect(concatFrame, frameIndex, counter, posSaver_left, posSaver_right, q_yolo2optflow_left, q_yolo2optflow_right);
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                    t_elapsed = t_elapsed + static_cast<float>(duration.count());
                    count_yolo++;
                    q_startOptflow.push(true);
                    std::cout << " ### Time taken by YOLO detection : " << duration.count() << " milliseconds ### " << std::endl;
                }
            }
        }
    }
    std::cout << "YOLO" << std::endl;
    std::cout << " Process Speed : " << static_cast<int>(count_yolo / t_elapsed * 1000) << " Hz for " << count_yolo << " cycles" << std::endl;
    std::cout << "*** LEFT ***" << std::endl;
    std::cout << "posSaver_left size=" << posSaver_left.size() << std::endl;
    utyolo.saveYolo(posSaver_left, file_yolo_left);
    std::cout << "*** RIGHT ***" << std::endl;
    std::cout << "posSaver_right size=" << posSaver_right.size() << std::endl;
    utyolo.saveYolo(posSaver_right, file_yolo_right);
}

void denseOpticalFlow(std::queue<bool>& q_startOptflow)
{
    /* construction of class */
    OpticalFlow of;
    Utility utof;
    int count_of = 0;
    float t_elapsed = 0;
    /* prepare storage */
    std::vector<std::vector<std::vector<std::vector<int>>>> posSaver_left; //[sequence,numHuman,numJoints,position] :{frameIndex,xCenter,yCenter}
    std::vector<std::vector<std::vector<std::vector<int>>>> posSaver_right; //[sequence,numHuman,numJoints,position] :{frameIndex,xCenter,yCenter}
    posSaver_left.reserve(2000);
    posSaver_right.reserve(2000);

    int counterStart = 0;
    while (true)
    {
        if (counterStart == 3) break;
        if (!q_startOptflow.empty()) {
            counterStart++;
            q_startOptflow.pop();
        }
    }
    std::cout << "start Optflow" << std::endl;
    /* frame is available */
    int counter = 1;
    int counterFinish = 0;
    while (true)
    {
        if (counterFinish == 10)
        {
            break;
        }
        if (queueFrame.empty())
        {
            counterFinish++;
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
        /* frame available */
        else
        {
            counterFinish = 0;
            //std::cout << "start opticalflow tracking" << std::endl;
            /* get images from queue */
            std::array<cv::Mat1b, 2> frames;
            int frameIndex;
            auto start = std::chrono::high_resolution_clock::now();
            utof.getImages(frames, frameIndex);
            cv::Mat1b frame_left = frames[0];
            cv::Mat1b frame_right = frames[1];
            if (frame_left.rows > 0 && frame_right.rows > 0)
            {
                std::thread thread_OF_left(&OpticalFlow::main, &of, std::ref(frame_left), std::ref(frameIndex), std::ref(posSaver_left), std::ref(q_yolo2optflow_left), std::ref(q_optflow2optflow_left), std::ref(queueTriangulation_left));
                //std::thread thread_OF_right(&OpticalFlow::main, &of, std::ref(frame_right), std::ref(frameIndex), std::ref(posSaver_right), std::ref(q_yolo2optflow_right), std::ref(q_optflow2optflow_right), std::ref(queueTriangulation_right));
                of.main(frame_right, frameIndex, posSaver_right, q_yolo2optflow_right, q_optflow2optflow_right,queueTriangulation_right);
                //std::cout << "both OF threads have started" << std::endl;
                thread_OF_left.join();
                //thread_OF_right.join();
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                //std::cout << "***** posSaver_left size=" << posSaver_left.size() << ", posSaver_right size=" << posSaver_right.size() << "********" << std::endl;
                count_of++;
                t_elapsed = t_elapsed + static_cast<float>(duration.count());
                std::cout << "<<< Time taken by OpticalFlow : " << duration.count() << " microseconds >>>" << std::endl;
                if (!queueFrame.empty() && !queueFrameIndex.empty())
                {
                    if (static_cast<float>(duration.count()) > 2500)
                    {
                        int count_delete = static_cast<int>(static_cast<float>(duration.count()) / 2500) - 1;
                        if (count_delete >= 1)
                        {
                            for (int i = 0; i < count_delete; i++)
                            {
                                if (!queueFrame.empty()) queueFrame.pop();
                                if (!queueFrameIndex.empty()) queueFrameIndex.pop();
                            }
                        }
                    }
                }
            }
        }
    }
    std::cout << "Optical Flow" << std::endl;
    std::cout << " Process Speed : " << static_cast<int>(count_of / t_elapsed * 1000000) << " Hz for " << count_of << " cycles" << std::endl;
    std::cout << "*** LEFT ***" << std::endl;
    utof.save(posSaver_left, file_of_left);
    std::cout << "*** RIGHT ***" << std::endl;
    utof.save(posSaver_right, file_of_right);
}

int main()
{
    /* image inference */
    /*
    cv::Mat img = cv::imread("video/0019.jpg");
    cv::Mat1b imgGray;
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
    std::cout << img.size()<< std::endl;
    int counter = 1;
    yoloPoseEstimator.detect(imgGray,counter);
    */
    //constructor 
    Utility ut;
    Triangulation tri;

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
    /* start multiThread */
    std::thread threadYolo(yolo);
    std::thread threadOF(denseOpticalFlow,std::ref(q_startOptflow));
    std::thread thread3d(&Triangulation::main, tri);
    while (true)
    {
        // Read the next frame
        cv::Mat frame_left, frame_right;
        capture_left >> frame_left;
        capture_right >> frame_right;
        counter++;
        //std::cout << "left size=" << frame_left.size() << ", right size=" << frame_right.size() << std::endl;
        if (frame_left.empty() || frame_right.empty())
            break;
        cv::Mat1b frameGray_left, frameGray_right;
        cv::cvtColor(frame_left, frameGray_left, cv::COLOR_RGB2GRAY);
        cv::cvtColor(frame_right, frameGray_right, cv::COLOR_RGB2GRAY);
        std::array<cv::Mat1b, 2> frames = { frameGray_left,frameGray_right };
        // cv::Mat1b frameGray;
        //  cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
        ut.pushImg(frames, counter);
    }
    threadYolo.join();
    threadOF.join();
    thread3d.join();

    return 0;
}
