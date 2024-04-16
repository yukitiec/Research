#include "stdafx.h"
#include "tracker.h"
#include "global_parameters.h"


void TemplateMatching::main(std::queue<bool>& q_startTracker)
{
    //constructor
    Utility utTM;
    int countIteration = 0;
    int counterFinish = 0;
    int counterStart = 0;
    while (true)
    {
        if (counterStart == 3)
            break;
        if (!q_startTracker.empty()) {
            q_startTracker.pop();
            counterStart++;
            std::cout << "Tracker :: by starting " << 3 - counterStart << std::endl;
        }
    }

    std::cout << "start tracking" << std::endl;
    while (true) // continue until finish
    {
        if (queueFrame.empty())
        {
            if (counterFinish == 10)
                break;
            counterFinish++;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            std::cout << "Tracker :: By finish : remain count is " << (10 - counterFinish) << std::endl;
            continue;
        }
        else if (!queueFrame.empty())
        {
            if ((!q_yolo2tracker_left.empty() && !q_yolo2tracker_right.empty()) || !q_tracker2tracker_left.front().bbox.empty() || !q_tracker2tracker_right.front().bbox.empty() || !q_seq2tracker_left.empty() || !q_seq2tracker_right.empty()) //some tracker exist
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
                        std::ref(q_tracker2tracker_left), std::ref(q_yolo2tracker_left), std::ref(q_tracker2yolo_left), std::ref(q_tracker2seq_left),
                        std::ref(q_seq2tracker_left));
                    //right
                    templateMatching(frame_right, frameIndex, posSaverTMRight, classSaverTMRight, detectedFrameRight, detectedFrameClassRight,
                        q_tracker2tracker_right, q_yolo2tracker_right, q_tracker2yolo_right, q_tracker2seq_right,
                        q_seq2tracker_right);
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
                            std::cout << "Tracker :: " << time_iteration << " microseconds and " << frame_delete << " frames will be deleted" << std::endl;
                            for (int i = 0; i < frame_delete; i++)
                            {
                                if (!queueFrame.empty()) queueFrame.pop();
                                if (!queueFrameIndex.empty()) queueFrameIndex.pop();
                            }
                        }
                    }
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

void TemplateMatching::templateMatching(cv::Mat1b& img, const int& frameIndex,
    std::vector<std::vector<cv::Rect2d>>& posSaver, std::vector<std::vector<int>>& classSaver, std::vector<int>& detectedFrame, std::vector<int>& detectedFrameClass,
    std::queue<Tracker2tracker>& q_tracker2tracker, std::queue<Yolo2tracker>& q_yolo2tracker, std::queue<Tracker2yolo>& q_tracker2yolo,
    std::queue<Tracker2seq>& q_tracker2seq, std::queue<std::vector<std::vector<double>>>& q_seq2tracker)
{
    //prepare containers
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
    std::vector<int> num_notMove; 
    std::vector<int> updated_num_notMove;

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
    cv::Mat1b previousImg; //previous image
    bool boolTrackerTM = false; // whether tracking is successful
    bool boolTrackerYolo = false; //whether yolo data is available


    //get data
    getData(img, boolTrackerTM, classIndexTM, bboxesTM, trackers_mosse, templatesTM, boolScalesTM, previousMoveTM, numTrackersTM, previousImg, num_notMove,q_tracker2tracker, q_seq2tracker);

    if (!q_yolo2tracker.empty())
    {
        /* get Yolo data and update Template matching data */
        organizeData(img, classIndexTM, bboxesTM, trackers_mosse, templatesTM, boolScalesTM, previousMoveTM, previousImg, boolTrackerYolo,
            updatedClasses, updatedBboxes, updatedTrackers, updatedTemplates, updatedMove, numTrackersTM, num_notMove,updated_num_notMove,q_yolo2tracker);
    }
    /* template from yolo isn't available but TM tracker exist */
    else if (boolTrackerTM)
    {
        updatedTrackers = trackers_mosse;
        updatedTemplates = templatesTM;
        updatedBboxes = bboxesTM;
        updatedClasses = classIndexTM;
        updatedMove = previousMoveTM;
        updated_num_notMove = num_notMove;
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
        std::vector<int> updated_num_notMove_tm(numTrackers,MAX_NOTMOVE);
        // finish initialization

        //std::cout << "processTM start" << std::endl;
        // 
        //std::cout << "Before :: num of notMove=" << updated_num_notMove.size() << "updatedClasses.size()="<<updatedClasses.size()<<", bboxesTM.size()=" << updatedBboxes.size() << std::endl;
        //tracking process
        process(updatedClasses, updatedBboxes, updatedTrackers, updatedTemplates, boolScalesTM, updatedMove, updated_num_notMove,img, updatedClassesTM, updatedBboxesTM, updatedTrackers_mosse, updatedTemplatesTM, updatedSearchScales, updatedMoveTM,updated_num_notMove_tm);
        //std::cout << "After :: num of notMove=" << updated_num_notMove_tm.size() << "updatedClasses.size()=" << updatedClassesTM.size() << ", bboxesTM.size()=" << updatedBboxesTM.size() << std::endl;
        //push latest data to Yolo
        if (boolTrackerYolo)
        {
            Tracker2yolo new_tracker2yolo;
            //std::cout << "updatedClassesTM.size()=" << updatedClassesTM.size() << std::endl;
            if (!q_tracker2yolo.empty()) q_tracker2yolo.pop();
            if (!updatedClassesTM.empty()) new_tracker2yolo.classIndex = updatedClassesTM;
            if (!updatedBboxesTM.empty()) new_tracker2yolo.bbox = updatedBboxesTM;
            q_tracker2yolo.push(new_tracker2yolo);
        }

        //Organize tracking data
        if (!updatedBboxesTM.empty())
        {
            //tracker2tracker -> pop() before push()
            if (!q_tracker2tracker.empty()) q_tracker2tracker.pop();
            Tracker2tracker update;
            //classIndex
            update.classIndex = updatedClassesTM;
            classSaver.push_back(updatedClassesTM); // save current class to the saver
            //bbox
            update.bbox = updatedBboxesTM;
            posSaver.push_back(updatedBboxesTM);// save current position to the vector
            //tracker
            update.tracker = updatedTrackers_mosse;
            //template
            update.templateImg = updatedTemplatesTM;
            //search scales
            update.scale = updatedSearchScales;
            //velocity
            update.vel = updatedMoveTM;
            //previousImg
            update.previousImg = img;
            update.num_notMove = updated_num_notMove_tm;
            //push data
            q_tracker2tracker.push(update);

            //tracker2seq
            if (!q_tracker2seq.empty()) q_tracker2seq.pop();
            Tracker2seq new_tracker2seq;
            //frameIndex
            new_tracker2seq.frameIndex = frameIndex;
            //classIndex
            new_tracker2seq.classIndex = updatedClassesTM;
            //bbox
            new_tracker2seq.bbox = updatedBboxesTM;
            //push
            q_tracker2seq.push(new_tracker2seq);
        }
        else //failed
        {
            Tracker2tracker update;
            //classIndex
            update.classIndex = updatedClassesTM;
            classSaver.push_back(updatedClassesTM); // save current class to the saver
            //previousImg
            update.previousImg = img;
            //push data
            q_tracker2tracker.push(update);
        }
        detectedFrame.push_back(frameIndex);
        detectedFrameClass.push_back(frameIndex);
    }
    else // no template or bbox -> nothing to do
    {
        if (!classIndexTM.empty())
        {
            Tracker2tracker update;
            //classIndex
            update.classIndex = classIndexTM;
            classSaver.push_back(classIndexTM); // save current class to the saver
            //previousImg
            update.previousImg = img;
            //push data
            q_tracker2tracker.push(update);
        }
    }
}
void TemplateMatching::getData(cv::Mat1b& frame, bool& boolTrackerTM, std::vector<int>& classIndexTM, std::vector<cv::Rect2d>& bboxesTM, std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& trackers_mosse,
    std::vector<cv::Mat1b>& templatesTM, std::vector<bool>& boolScalesTM, std::vector<std::vector<int>>& previousMove,
    int& numTrackersTM, cv::Mat1b& previousImg, std::vector<int>& num_notMove, std::queue<Tracker2tracker>& q_tracker2tracker, std::queue<std::vector<std::vector<double>>>& q_seq2tracker)
{
    if (!q_tracker2tracker.empty())
    {
        //make a instance
        Tracker2tracker newData = q_tracker2tracker.front();
        q_tracker2tracker.pop();
        //classIndex
        if (!newData.classIndex.empty())
        {
            //classIndex
            classIndexTM = newData.classIndex;
            numTrackersTM = classIndexTM.size();
            //previousImg
            previousImg = newData.previousImg;
        }
        //bbox
        if (!newData.bbox.empty()) {
            boolTrackerTM = true;
            //bbox
            bboxesTM = newData.bbox;
            //tracker
            trackers_mosse = newData.tracker;
            //template
            templatesTM = newData.templateImg;
            //scale
            boolScalesTM = newData.scale;
            //previous velocity
            previousMove = newData.vel;
            //num not move
            num_notMove = newData.num_notMove;
            //std::cout << "number of notMove ";
            //for (int i = 0; i < num_notMove.size(); i++)
            //    std::cout << num_notMove[i] << " ";
            //std::cout << std::endl;
            //std::cout <<"num of classes="<<classIndexTM<< ", num of notMove=" << num_notMove.size() << ", bboxesTM.size()=" << bboxesTM.size() << std::endl;
        }
        else
            boolTrackerTM = false;
    }
    //Kalman filter data compensation
    if (bool_kf)
    {
        if (!q_seq2tracker.empty())
        {
            std::vector<std::vector<double>> kf_predictions = q_seq2tracker.front();
            q_seq2tracker.pop();
            int counter_label = 0;
            int counter_tracker = 0;
            for (std::vector<double>& kf_predict : kf_predictions)
            {
                if (!kf_predict.empty() && classIndexTM[counter_label] < 0 && classIndexTM[counter_label] != -2) //revival
                {
                    classIndexTM[counter_label] = (int)kf_predict[0]; //update label
                    cv::Rect2d newRoi((double)std::min(std::max((int)kf_predict[1], 0), (frame.cols - (int)kf_predict[3] - 1)), (double)std::min(std::max((int)kf_predict[2], 0), (frame.rows - (int)kf_predict[4] - 1)), (double)kf_predict[3], (double)kf_predict[4]);
                    bboxesTM.insert(bboxesTM.begin() + counter_tracker, newRoi);
                    cv::Ptr<cv::mytracker::TrackerMOSSE> tracker = cv::mytracker::TrackerMOSSE::create();
                    tracker->init(frame, newRoi);//update tracker with current frame
                    trackers_mosse.insert(trackers_mosse.begin() + counter_tracker, tracker);
                    templatesTM.insert(templatesTM.begin() + counter_tracker, frame(newRoi)); //template
                    boolScalesTM.insert(boolScalesTM.begin() + counter_tracker, false); //scale
                    previousMove.insert(previousMove.begin() + counter_tracker, defaultMove); //previous move velocity
                    num_notMove.insert(num_notMove.begin() + counter_tracker, 0); //number of not moving times
                    boolTrackerTM = true;
                    //std::cout << "compensate with KF data" << std::endl;
                }
                if (classIndexTM[counter_label] >= 0) counter_tracker++;
                counter_label++;
            }
        }
    }
}

void TemplateMatching::organizeData(cv::Mat1b& frame, std::vector<int>& classIndexTM, std::vector<cv::Rect2d>& bboxesTM,
    std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& trackers_mosse, std::vector<cv::Mat1b>& templatesTM,
    std::vector<bool>& boolScalesTM, std::vector<std::vector<int>>& previousMove, cv::Mat1b& previousImg,
    bool& boolTrackerYolo,
    std::vector<int>& updatedClasses, std::vector<cv::Rect2d>& updatedBboxes,
    std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers, std::vector<cv::Mat1b>& updatedTemplates,
    std::vector<std::vector<int>>& updatedMove, int& numTrackersTM,std::vector<int>& num_notMove, std::vector<int>& updated_num_notMove,
    std::queue<Yolo2tracker>& q_yolo2tracker)
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
    getYoloData(trackersYolo, templatesYolo, bboxesYolo, classIndexesYolo, q_yolo2tracker); // get new frame
    // combine Yolo and TM data, and update latest data
    combineYoloTMData(frame, classIndexesYolo, classIndexTM, bboxesYolo, bboxesTM, trackersYolo, trackers_mosse, templatesYolo, templatesTM, previousMove, previousImg,
        updatedClasses, updatedBboxes, updatedTrackers, updatedTemplates, boolScalesTM, updatedMove, numTrackersTM,num_notMove,updated_num_notMove);
}

void TemplateMatching::getYoloData(std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& newTrackers, std::vector<cv::Mat1b>& newTemplates, std::vector<cv::Rect2d>& newBboxes, std::vector<int>& newClassIndexes,
    std::queue<Yolo2tracker>& q_yolo2tracker)
{
    Yolo2tracker newData;
    newData = q_yolo2tracker.front();
    q_yolo2tracker.pop();
    //classIndex
    newClassIndexes = newData.classIndex;
    //bbox
    if (!newData.bbox.empty()) newBboxes = newData.bbox;
    //tracker
    if (!newData.tracker.empty()) newTrackers = newData.tracker;
    //template
    if (!newData.templateImg.empty()) newTemplates = newData.templateImg;
}

float TemplateMatching::calculateIoU_Rect2d(const cv::Rect2d& box1, const cv::Rect2d& box2)
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

void TemplateMatching::combineYoloTMData(cv::Mat1b& frame, std::vector<int>& classIndexesYolo, std::vector<int>& classIndexTM,
    std::vector<cv::Rect2d>& bboxesYolo, std::vector<cv::Rect2d>& bboxesTM,
    std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& trackersYolo, std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& trackers_mosse,
    std::vector<cv::Mat1b>& templatesYolo, std::vector<cv::Mat1b>& templatesTM,
    std::vector<std::vector<int>>& previousMove, cv::Mat1b& previousImg,
    std::vector<int>& updatedClasses, std::vector<cv::Rect2d>& updatedBboxes,
    std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers, std::vector<cv::Mat1b>& updatedTemplates,
    std::vector<bool>& boolScalesTM, std::vector<std::vector<int>>& updatedMove, const int& numTrackersTM,std::vector<int>& num_notMove, std::vector<int>& updated_num_notMove)
{
    int counterYolo = 0;
    int counterTM = 0;      // for counting TM adaptations
    int counterClassTM = 0; // for counting TM class counter
    int counter_notMove = 0; //number of notMove -> if previous tracker exist or add new tracker -> add 1
    // organize current situation : determine if tracker is updated with Yolo or TM, and is deleted
    // think about tracker continuity : tracker survival : (not Yolo Tracker) and (not TM tracker)
    int numPastLabels = classIndexTM.size();
    /* should check carefully -> compare num of detection */
    for (const int& classIndex : classIndexesYolo)
    {
        ///if (!classIndexTM.empty()) //when comment out comment out this line, too!
        //   std::cout <<"numPastLabels="<<numPastLabels<<", classIndexTM.size()="<<classIndexTM.size()<<", bboxesTM.size()="<<bboxesTM.size()<<"classIndexTM="<<classIndexTM[counterClassTM]<< ",classIndexesYolo.size()=" << classIndexesYolo.size() << ", classYolo:" << classIndex << ", bboxesYolo.size()=" << bboxesYolo.size() << "counterClassTM=" << counterClassTM << ", counterTM=" << counterTM << std::endl;
        //std::cout << "num_notMove=" << updated_num_notMove.size() << ", updatedBboxes.size()=" << updatedBboxes.size() << std::endl;
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
                            if (classIndex == 0) //circular objects
                            {
                                if (bool_comparePSR) //compare PSR
                                {
                                    double psr_yolo = check_tracker(previousImg, bboxesTM[counterTM], trackersYolo[counterYolo]); //calculate PSR of yolo tracker
                                    //adopt current tracker
                                    if ((trackers_mosse[counterTM]->previous_psr) > psr_yolo && (trackers_mosse[counterTM]->counter_skip <= 0)) {
                                        //std::cout << "keep rameined tracker" << std::endl;
                                        updatedTrackers.push_back(trackers_mosse[counterTM]); // update template to YOLO's one
                                        updatedTemplates.push_back(templatesTM[counterTM]); // update template to YOLO's one
                                    }
                                    //adopt yolo tracker
                                    else {
                                        updatedTrackers.push_back(trackersYolo[counterYolo]); // update template to YOLO's one
                                        updatedTemplates.push_back(templatesYolo[counterYolo]); // update template to YOLO's one
                                    }
                                }
                                else {
                                    if ((trackers_mosse[counterTM]->previous_psr) >= min_keep_psr)
                                    {
                                        //std::cout << "keep ramained tracker" << std::endl;
                                        updatedTrackers.push_back(trackers_mosse[counterTM]); // update template to YOLO's one
                                        updatedTemplates.push_back(templatesTM[counterTM]); // update template to YOLO's one
                                    }
                                    else
                                    {
                                        updatedTrackers.push_back(trackersYolo[counterYolo]); // update template to YOLO's one
                                        updatedTemplates.push_back(templatesYolo[counterYolo]); // update template to YOLO's one
                                    }
                                }
                            }
                            else if (classIndex == 1) //non-circular objects -> update every time
                            {
                                //std::cout << "keep rameined tracker" << std::endl;
                                updatedTrackers.push_back(trackers_mosse[counterYolo]); // update template to YOLO's one
                                updatedTemplates.push_back(templatesTM[counterYolo]); // update template to YOLO's one
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
                        updated_num_notMove.push_back(num_notMove[counterTM]);
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
                        updated_num_notMove.push_back(0); //add not_move count
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
                        updated_num_notMove.push_back(num_notMove[counterTM]);
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
                    updated_num_notMove.push_back(0);
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
                updated_num_notMove.push_back(0);
                counterYolo++;
            }
            /* tracker was not found in YOLO */
            else
            {
                updatedClasses.push_back(classIndex);
            }
        }
    }
    std::cout << "num_notMove=" << updated_num_notMove.size() << ", updatedBboxes.size()=" << updatedBboxes.size() << std::endl;
    //IoU check -> delete duplicated trackers
    if (bool_iouCheck)
    {
        if (updatedBboxes.size() >= 2)
        {
            std::cout << " /////////////////////////// check Duplicated trackers" << std::endl;
            //std::cout << "num_notMove=" << num_notMove.size() << ", trackers="<<updatedTrackers.size()<<", bboxes=" << updatedBboxes.size() << std::endl;
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
                                    updatedTemplates[counter_template] = previousImg(newRoi); //change to previousFrame(newRoi);
                                    cv::Ptr<cv::mytracker::TrackerMOSSE> tracker = cv::mytracker::TrackerMOSSE::create();
                                    tracker->init(previousImg, newRoi); //change to tracker->init(previousFrame,newRoi);
                                    updatedTrackers[counter_template] = tracker;
                                    updatedMove[counter_template][0] = (int)((updatedMove[counter_template][0] + updatedMove[i][0]) / 2);
                                    updatedMove[counter_template][1] = (int)((updatedMove[counter_template][1] + updatedMove[i][1]) / 2);
                                    updated_num_notMove[counter_template] = std::min(updated_num_notMove[counter_template], updated_num_notMove[i]);
                                    //delete tracker
                                    updatedTrackers.erase(updatedTrackers.begin() + i);
                                    updatedTemplates.erase(updatedTemplates.begin() + i);
                                    updatedBboxes.erase(updatedBboxes.begin() + i);
                                    boolScalesTM.erase(boolScalesTM.begin() + i);
                                    updatedClasses[labels_on[i]] = -2;
                                    updatedMove.erase(updatedMove.begin() + i);
                                    updated_num_notMove.erase(updated_num_notMove.begin() + i);
                                    labels_on.erase(labels_on.begin() + i);
                                }
                                else
                                {
                                    if ((updatedTrackers[i]->previous_psr) >= (updatedTrackers[counter_template]->previous_psr))
                                    {
                                        //exchange tracker
                                        updatedTrackers[counter_template] = updatedTrackers[i];
                                        updatedTemplates[counter_template] = updatedTemplates[i];
                                        updatedBboxes[counter_template] = updatedBboxes[i];
                                        updatedMove[counter_template] = updatedMove[i];
                                        updated_num_notMove[counter_template] = std::min(updated_num_notMove[counter_template], updated_num_notMove[i]);
                                    }
                                    updatedTrackers.erase(updatedTrackers.begin() + i);
                                    updatedTemplates.erase(updatedTemplates.begin() + i);
                                    updatedBboxes.erase(updatedBboxes.begin() + i);
                                    boolScalesTM.erase(boolScalesTM.begin() + i);
                                    updatedClasses[labels_on[i]] = -2;
                                    updatedMove.erase(updatedMove.begin() + i);
                                    updated_num_notMove.erase(updated_num_notMove.begin() + i);
                                    labels_on.erase(labels_on.begin() + i);
                                }
                            }
                            else if (bool_checkVel)
                            {
                                int vx_base = updatedMove[counter_template][0]; int vy_base = updatedMove[counter_template][1];
                                int vx_cand = updatedMove[i][0]; int vy_cand = updatedMove[i][1];
                                float norm_base = std::pow((std::pow(vx_base, 2) + std::pow(vy_base, 2)), 0.5);
                                float norm_cand = std::pow((std::pow(vx_cand, 2) + std::pow(vy_cand, 2)), 0.5);
                                float cos = ((vx_base * vx_cand) + (vy_base * vy_cand)) / (norm_base * norm_cand);//check direction
                                if (cos <= thresh_cos_dup) //judge as another objects -> move ROI of another things
                                {
                                    int dx_cand = (int)(vx_cand / norm_cand * delta_move);
                                    int dy_cand = (int)(vy_cand / norm_cand * delta_move);
                                    //move duplicated bboxes position
                                    updatedBboxes[i].x += dx_cand;
                                    updatedBboxes[i].y += dy_cand;
                                }
                                else //same objects -> delete duplicated one
                                {
                                    //augment tracker and delete new tracker
                                    double left = std::min(updatedBboxes[i].x, updatedBboxes[counter_template].x);
                                    double right = std::max((updatedBboxes[i].x + updatedBboxes[i].width), (updatedBboxes[counter_template].x + updatedBboxes[counter_template].width));
                                    double top = std::min(updatedBboxes[i].y, updatedBboxes[counter_template].y);
                                    double bottom = std::max((updatedBboxes[i].y + updatedBboxes[i].height), (updatedBboxes[counter_template].y + updatedBboxes[counter_template].height));
                                    if ((0 < left && left < right && right < frame.cols && (right - left)>10) && (0 < top && top < bottom && bottom < frame.rows && (bottom - top)>10))
                                    {
                                        cv::Rect2d newRoi(left, top, (right - left), (bottom - top));
                                        updatedBboxes[counter_template] = newRoi;
                                        updatedTemplates[counter_template] = previousImg(newRoi);//change to previousFrame(newRoi);
                                        cv::Ptr<cv::mytracker::TrackerMOSSE> tracker = cv::mytracker::TrackerMOSSE::create();
                                        tracker->init(previousImg, newRoi); //change to tracker->init(previousFrame,newRoi);
                                        updatedTrackers[counter_template] = tracker;
                                        updatedMove[counter_template][0] = (int)((updatedMove[counter_template][0] + updatedMove[i][0]) / 2);
                                        updatedMove[counter_template][1] = (int)((updatedMove[counter_template][1] + updatedMove[i][1]) / 2);
                                        updated_num_notMove[counter_template] = std::min(updated_num_notMove[counter_template], updated_num_notMove[i]);
                                        //delete tracker
                                        updatedTrackers.erase(updatedTrackers.begin() + i);
                                        updatedTemplates.erase(updatedTemplates.begin() + i);
                                        updatedBboxes.erase(updatedBboxes.begin() + i);
                                        boolScalesTM.erase(boolScalesTM.begin() + i);
                                        updatedClasses[labels_on[i]] = -2;
                                        updatedMove.erase(updatedMove.begin() + i);
                                        updated_num_notMove.erase(updated_num_notMove.begin() + i);
                                        labels_on.erase(labels_on.begin() + i);
                                    }
                                    else
                                    {
                                        if ((updatedTrackers[i]->previous_psr) > (updatedTrackers[counter_template]->previous_psr))
                                        {
                                            //exchange trackere
                                            updatedTrackers[counter_template] = updatedTrackers[i];
                                            updatedTemplates[counter_template] = updatedTemplates[i];
                                            updatedBboxes[counter_template] = updatedBboxes[i];
                                            updatedMove[counter_template] = updatedMove[i];
                                            updated_num_notMove[counter_template] = std::min(updated_num_notMove[counter_template], updated_num_notMove[i]);
                                        }
                                        updatedTrackers.erase(updatedTrackers.begin() + i);
                                        updatedTemplates.erase(updatedTemplates.begin() + i);
                                        updatedBboxes.erase(updatedBboxes.begin() + i);
                                        boolScalesTM.erase(boolScalesTM.begin() + i);
                                        updatedClasses[labels_on[i]] = -2;
                                        updatedMove.erase(updatedMove.begin() + i);
                                        updated_num_notMove.erase(updated_num_notMove.begin() + i);
                                        labels_on.erase(labels_on.begin() + i);
                                    }
                                }
                            }
                            else
                            {
                                if ((updatedTrackers[i]->previous_psr) > (updatedTrackers[counter_template]->previous_psr))
                                {
                                    //exchange trackere
                                    updatedTrackers[counter_template] = updatedTrackers[i];
                                    updatedTemplates[counter_template] = updatedTemplates[i];
                                    updatedBboxes[counter_template] = updatedBboxes[i];
                                    updatedMove[counter_template] = updatedMove[i];
                                    updated_num_notMove[counter_template] = std::min(updated_num_notMove[counter_template], updated_num_notMove[i]);
                                }
                                updatedTrackers.erase(updatedTrackers.begin() + i);
                                updatedTemplates.erase(updatedTemplates.begin() + i);
                                updatedBboxes.erase(updatedBboxes.begin() + i);
                                boolScalesTM.erase(boolScalesTM.begin() + i);
                                updatedClasses[labels_on[i]] = -2;
                                updatedMove.erase(updatedMove.begin() + i);
                                updated_num_notMove.erase(updated_num_notMove.begin() + i);
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

double TemplateMatching::check_tracker(cv::Mat1b& previousImg, cv::Rect2d& roi, cv::Ptr<cv::mytracker::TrackerMOSSE>& tracker)
{
    double scale_x = scaleXTM;
    double scale_y = scaleYTM;
    int leftSearch = std::min(previousImg.cols, std::max(0, static_cast<int>(roi.x - (scale_x - 1) * roi.width / 2)));
    int topSearch = std::min(previousImg.rows, std::max(0, static_cast<int>(roi.y - (scale_y - 1) * roi.height / 2)));
    int rightSearch = std::max(0, std::min(previousImg.cols, static_cast<int>(roi.x + (scale_x + 1) * roi.width / 2)));
    int bottomSearch = std::max(0, std::min(previousImg.rows, static_cast<int>(roi.y + (scale_y + 1) * roi.height / 2)));
    if ((rightSearch - leftSearch) > 0 && (bottomSearch - topSearch) > 0)
    {
        cv::Rect2d searchArea(leftSearch, topSearch, (rightSearch - leftSearch), (bottomSearch - topSearch));
        cv::Mat1b croppedImg = previousImg.clone();
        croppedImg = croppedImg(searchArea); // crop img
        //convert roi from image coordinate to local search area coordinate
        cv::Rect2d croppedRoi;
        croppedRoi.x = roi.x - searchArea.x;
        croppedRoi.y = roi.y - searchArea.y;
        croppedRoi.width = roi.width;
        croppedRoi.height = roi.height;
        // MOSSE Tracker
        double psr = tracker->check_quality(croppedImg, croppedRoi, true);
        return psr;
    }
    else
        return 0;
}

void TemplateMatching::process(std::vector<int>& updatedClasses, std::vector<cv::Rect2d>& updatedBboxes,
    std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers, std::vector<cv::Mat1b>& updatedTemplates,
    std::vector<bool>& boolScalesTM, std::vector<std::vector<int>>& updatedMove, std::vector<int>& updated_num_notMove,
    cv::Mat1b& img,
    std::vector<int>& updatedClassesTM, std::vector<cv::Rect2d>& updatedBboxesTM,
    std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers_mosse, std::vector<cv::Mat1b>& updatedTemplatesTM,
    std::vector<bool>& updatedSearchScales, std::vector<std::vector<int>>& updatedMoveTM,std::vector<int>& updated_num_notMove_tm)
{
    int counterTracker = 0; // counter for number of tracker
    int counter = 0;        // counter for all classes
    if (bool_multithread)
    {
        std::vector<std::thread> threadTracking; // prepare threads
        for (const int& classIndexTM : updatedClasses)
        {
            if (classIndexTM >= 0)
            {
                //template matching
                threadTracking.emplace_back(&TemplateMatching::matchingTemplate, this, classIndexTM, counterTracker, counter,
                    std::ref(updatedTemplates), std::ref(boolScalesTM), std::ref(updatedBboxes), std::ref(updatedMove), std::ref(updated_num_notMove),std::ref(img),
                    std::ref(updatedTrackers_mosse), std::ref(updatedTemplatesTM), std::ref(updatedBboxesTM), std::ref(updatedClassesTM), std::ref(updatedSearchScales), std::ref(updatedMoveTM),std::ref(updated_num_notMove_tm));
                //MOSSE
                threadTracking.emplace_back(&TemplateMatching::track_mosse, this, classIndexTM, counterTracker, counter,
                    std::ref(updatedTrackers), std::ref(updatedTemplates), std::ref(boolScalesTM), std::ref(updatedBboxes), std::ref(updatedMove), std::ref(updated_num_notMove),std::ref(img),
                    std::ref(updatedTrackers_mosse), std::ref(updatedTemplatesTM), std::ref(updatedBboxesTM), std::ref(updatedClassesTM), std::ref(updatedSearchScales), std::ref(updatedMoveTM),std::ref(updated_num_notMove_tm));
                // std::cout << counter << "-th thread has started" << std::endl;
                counterTracker++;
            }
            else if (classIndexTM == -2)
                updatedClassesTM[counter] = -2;
            counter++;
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
            //std::cout << "process ongoin :: " << classIndexTM << std::endl;
            if (classIndexTM >= 0)
            {
                //template matching
                std::thread threadTM(&TemplateMatching::matchingTemplate, this, classIndexTM, counterTracker, counter, std::ref(updatedTemplates), std::ref(boolScalesTM),
                    std::ref(updatedBboxes), std::ref(updatedMove), std::ref(updated_num_notMove),std::ref(img), std::ref(updatedTrackers_mosse), std::ref(updatedTemplatesTM), std::ref(updatedBboxesTM), std::ref(updatedClassesTM), std::ref(updatedSearchScales), std::ref(updatedMoveTM),std::ref(updated_num_notMove_tm));
                //mosse
                track_mosse(classIndexTM, counterTracker, counter, updatedTrackers, updatedTemplates, boolScalesTM, updatedBboxes, updatedMove,updated_num_notMove, img, updatedTrackers_mosse, updatedTemplatesTM, updatedBboxesTM, updatedClassesTM, updatedSearchScales, updatedMoveTM,updated_num_notMove_tm);
                threadTM.join();
                counterTracker++;
            }
            else if (classIndexTM == -2)
                updatedClassesTM[counter] = -2;
            counter++;
        }
    }
    // organize data -> if templates not updated -> erase data]
    int counter_tracker = 0;

    int counterCheck = 0;
    int counter_label = 0;
    //std::cout << "Before deleting : : updatedBboxesTM.size()=" << updatedBboxesTM.size() << std::endl;
    while (true)
    {
        if (counterCheck >= updatedBboxesTM.size() && counter_label>=updatedClassesTM.size()) break;
        if (counterCheck >= updatedBboxesTM.size() && counter_label < updatedClassesTM.size()) {
            while (true) {
                if (counter_label >= updatedClassesTM.size()) break;
                if (updatedClassesTM[counter_label] >= 0) updatedClassesTM[counter_label] = -1;
                counter_label++;
            }
            break;
        }
        // not updated
        //std::cout << "bboxes width=" << updatedBboxesTM[counterCheck].width << ", label=" << updatedClassesTM[counter_label] << std::endl;
        if (updatedBboxesTM[counterCheck].width <= 0 || updated_num_notMove_tm[counterCheck]>=MAX_NOTMOVE)
        {
            //std::cout << "bboxes size=" << updatedBboxesTM.size() << ", num_notMove.size()=" << num_notMove.size() << std::endl;
            //std::cout << "delete data" << std::endl;
            updatedTrackers_mosse.erase(updatedTrackers_mosse.begin() + counterCheck);
            updatedTemplatesTM.erase(updatedTemplatesTM.begin() + counterCheck);
            updatedBboxesTM.erase(updatedBboxesTM.begin() + counterCheck);
            updatedSearchScales.erase(updatedSearchScales.begin() + counterCheck);
            updatedMoveTM.erase(updatedMoveTM.begin() + counterCheck);
            updated_num_notMove_tm.erase(updated_num_notMove_tm.begin() + counterCheck);
            if (updatedClassesTM[counter_label] >=0) updatedClassesTM[counter_label] = -1;
            //std::cout << "finish deleting" << std::endl;
        }
        else {
            counterCheck++;
        }
        counter_label++;
    }
    //std::cout << "after deleting : : updatedBboxesTM.size()=" << updatedBboxesTM.size() << std::endl;
}

void TemplateMatching::matchingTemplate(const int classIndexTM, int counterTracker, int counter,
    std::vector<cv::Mat1b>& updatedTemplates, std::vector<bool>& boolScalesTM, std::vector<cv::Rect2d>& updatedBboxes, std::vector<std::vector<int>>& updatedMove, std::vector<int>& updated_num_notMove,cv::Mat1b& img,
    std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers_mosse, std::vector<cv::Mat1b>& updatedTemplatesTM,
    std::vector<cv::Rect2d>& updatedBboxesTM, std::vector<int>& updatedClassesTM, std::vector<bool>& updatedSearchScales, std::vector<std::vector<int>>& updatedMoveTM,std::vector<int>& updated_num_notMove_tm)
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
            scale_x = static_cast<double>(std::max(std::min((1.3 + (std::abs(deltaX_past) / previousRoi.width)), scaleXYolo), scaleXTM));
            scale_y = static_cast<double>(std::max(std::min((1.3 + (std::abs(deltaY_past) / previousRoi.height)), scaleYYolo), scaleYTM));
        }
        leftSearch = std::min(std::max(0, static_cast<int>(previousRoi.x + deltaX_past - (scale_x - 1) * previousRoi.width / 2)), img.cols);
        topSearch = std::min(std::max(0, static_cast<int>(previousRoi.y + deltaY_past - (scale_y - 1) * previousRoi.height / 2)), img.rows);
        rightSearch = std::max(0, std::min(img.cols, static_cast<int>(previousRoi.x + deltaX_past + (scale_x + 1) * previousRoi.width / 2)));
        bottomSearch = std::max(0, std::min(img.rows, static_cast<int>(previousRoi.y + deltaY_past + (scale_y + 1) * previousRoi.height / 2)));
    }
    else // scale is set to YOLO : larger search area
    {
        leftSearch = std::min(img.cols, std::max(0, static_cast<int>(previousRoi.x + deltaX_past - (scaleXYolo - 1) * previousRoi.width / 2)));
        topSearch = std::min(img.rows, std::max(0, static_cast<int>(previousRoi.y + deltaY_past - (scaleYYolo - 1) * previousRoi.height / 2)));
        rightSearch = std::max(0, std::min(img.cols, static_cast<int>(previousRoi.x + deltaX_past + (scaleXYolo + 1) * previousRoi.width / 2)));
        bottomSearch = std::max(0, std::min(img.rows, static_cast<int>(previousRoi.y + deltaY_past + (scaleYYolo + 1) * previousRoi.height / 2)));
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
                cv::Rect2d roi, roi_cropped;
                leftRoi = std::min(img.cols, std::max(0, static_cast<int>(matchLoc.x + leftSearch)));
                topRoi = std::min(img.rows, std::max(0, static_cast<int>(matchLoc.y + topSearch)));
                rightRoi = std::max(0, std::min(img.cols, static_cast<int>(leftRoi + templateImg.cols)));
                bottomRoi = std::max(0, std::min(img.rows, static_cast<int>(topRoi + templateImg.rows)));
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
                    
                    // update information
                    if (updatedBboxesTM.at(counterTracker).x == -1) //default data
                    {
                        //update roi with template matching result
                        cv::Ptr<cv::mytracker::TrackerMOSSE> tracker = cv::mytracker::TrackerMOSSE::create();
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
                        if (updatedBboxesTM.at(counterTracker).x == -1) {
                            updatedBboxesTM.at(counterTracker) = roi;
                            if (std::pow(deltaX_current, 2) + std::pow(deltaY_current, 2) >= MoveThreshold)
                                updated_num_notMove_tm.at(counterTracker) = 0;
                            else
                                updated_num_notMove_tm[counterTracker] = updated_num_notMove.at(counterTracker) + 1;
                        }
                        //std::cout << "roi.x=" << roi.x << ", roi.y=" << roi.y << std::endl;
                        //std::cout << "  ------- succeed in tracking with TM ------------------ " << std::endl;
                    }
                }
            }
        }
    }
}

void TemplateMatching::track_mosse(const int classIndexTM, int counterTracker, int counter,
    std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers, std::vector<cv::Mat1b>& updatedTemplates, std::vector<bool>& boolScalesTM, std::vector<cv::Rect2d>& updatedBboxes,
    std::vector<std::vector<int>>& updatedMove,std::vector<int>& updated_num_notMove,cv::Mat1b& img,
    std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers_mosse, std::vector<cv::Mat1b>& updatedTemplatesTM,
    std::vector<cv::Rect2d>& updatedBboxesTM, std::vector<int>& updatedClassesTM, std::vector<bool>& updatedSearchScales, std::vector<std::vector<int>>& updatedMoveTM,std::vector<int>& updated_num_notMove_tm)
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
            scale_x = static_cast<double>(std::max(std::min((1.3 + (deltaX_past / previousRoi.width)), scaleXYolo), scaleXTM));
            scale_y = static_cast<double>(std::max(std::min((1.3 + (deltaY_past / previousRoi.height)), scaleYYolo), scaleYTM));
        }
        leftSearch = std::min(img.cols, std::max(0, static_cast<int>(previousRoi.x + deltaX_past - (scale_x - 1) * previousRoi.width / 2)));
        topSearch = std::min(img.rows, std::max(0, static_cast<int>(previousRoi.y + deltaY_past - (scale_y - 1) * previousRoi.height / 2)));
        rightSearch = std::max(0, std::min(img.cols, static_cast<int>(previousRoi.x + deltaX_past + (scale_x + 1) * previousRoi.width / 2)));
        bottomSearch = std::max(0, std::min(img.rows, static_cast<int>(previousRoi.y + deltaY_past + (scale_y + 1) * previousRoi.height / 2)));
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
        if (psr >= threshold_mosse)
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
                    updated_num_notMove_tm.at(counterTracker) = 0;
                else
                    updated_num_notMove_tm[counterTracker] = updated_num_notMove.at(counterTracker) + 1;

                // update information
                updatedBboxesTM.at(counterTracker) = newRoi;
                updatedTrackers_mosse.at(counterTracker) = tracker;
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