#pragma once

#ifndef SEQUENCE_H
#define SEQUENCE_H

#include "stdafx.h"
#include "utility.h"

//Kalman filter setting
extern const int COUNTER_VALID; //frames by official tracker
extern const int COUNTER_LOST; //frames by deleting tracker
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

//for sharing data with triangulation
extern std::vector<std::vector<std::vector<double>>> seqData_left, seqData_right;

extern std::queue<bool> queueYolo_tracker2seq_left, queueYolo_tracker2seq_right;
extern std::queue<bool> queueYolo_seq2tri_left, queueYolo_seq2tri_right;

// 3D positioning ~ trajectory prediction
extern std::queue<int> queueTargetFrameIndex_left;                      // TM estimation frame
extern std::queue<int> queueTargetFrameIndex_right;
extern std::queue<std::vector<cv::Rect2d>> queueTargetBboxesLeft;  // bboxes from template matching for predict objects' trajectory
extern std::queue<std::vector<cv::Rect2d>> queueTargetBboxesRight; // bboxes from template matching for predict objects' trajectory
extern std::queue<std::vector<int>> queueTargetClassIndexesLeft;   // class from template matching for maintain consistency
extern std::queue<std::vector<int>> queueTargetClassIndexesRight;  // class from template matching for maintain consistency

//latest labels for matching data in both images
extern std::queue<std::vector<int>> queueUpdateLabels_left;
extern std::queue<std::vector<int>> queueUpdateLabels_right;

//from seq : kalman prediction
extern std::queue<std::vector<std::vector<double>>> queueKfPredictLeft; //{label, left,top,width,height}
extern std::queue<std::vector<std::vector<double>>> queueKfPredictRight;

class Sequence
{
private:
    //deault vector
    std::vector<std::vector<double>> defaultVector{ {} }; //default vector for kalman filter initialization
    std::vector<std::vector<double>> fault{ {-1,-1,-1,-1,-1,-1} }; //fault vector for seqData if first tracking was failed : {frameIndex, label, left, top, width, height}
    //obesrvation vector for kalman filter
    Eigen::Vector2d observation;
    float t_elapsed = 0;
    int counter_both = 0;
    int counter_left = 0;
    int counter_right = 0;
    const bool bool_addKF = false;
public:
    //storage
    std::vector<std::vector<std::vector<double>>> kfData_left, kfData_right; //{num of objects, num of sequence, unit vector}
    std::vector<KalmanFilter2D> kalmanVector_left, kalmanVector_right; //kalman filter instances
    //seqData : {frameIndex, label, left, top, width, height}, kfData : {frameIndex, label,left, top,width,height}
    Eigen::Vector<double, 6> kf_predict; //for kalmanfilter prediction result 
    //storage for new data
    std::vector<cv::Rect2d> newRoi_left, newRoi_right;
    std::vector<int> newLabels_left, newLabels_right;
    int frameIndex_left, frameIndex_right;

    Sequence()
    {
        std::cout << "construct Sequence class" << std::endl;
    }

    void main()
    {
        Utility utSeq;

        while (true)
        {
            if (!queueTargetBboxesLeft.empty() || !queueTargetBboxesRight.empty()) break;
            //std::cout << "wait for target data" << std::endl;
        }
        std::cout << "start saving sequential data" << std::endl;
        int counterIteration = 0;
        int counterFinish = 0;
        int counterNextIteration = 0;
        while (true) // continue until finish
        {
            counterIteration++;
            //std::cout << "get imgs" << std::endl;
            if (queueFrame.empty())
            {
                if (counterFinish == 10) break;
                counterFinish++;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                std::cout << "Sequence :: By finish : remain count is " << (10 - counterFinish) << std::endl;
                continue;
            }
            else
            {
                counterFinish = 0;
                //std::cout << "queueTargetBboxesLeft.empty():" << queueTargetBboxesLeft.empty() << "queueTargetBboxesRight.empty():" << queueTargetBboxesRight.empty() << "queueTargetClassIndexLeft.empty():" << queueTargetClassIndexesLeft.empty() << "queueTargetClassIndexRight.empty():" << queueTargetClassIndexesRight.empty() << std::endl;
                /* new detection data available */
                if (!queueTargetBboxesLeft.empty() && !queueTargetBboxesRight.empty())
                {
                    counterNextIteration = 0;
                    auto start = std::chrono::high_resolution_clock::now();
                    //left 
                    //std::cout << "start saving both" << std::endl;
                    std::thread thread_left(&Sequence::organize, this, std::ref(newRoi_left), std::ref(newLabels_left), std::ref(frameIndex_left),
                        std::ref(queueTargetBboxesLeft), std::ref(queueTargetClassIndexesLeft), std::ref(queueTargetFrameIndex_left),
                        std::ref(seqData_left), std::ref(kfData_left), std::ref(kalmanVector_left),
                        std::ref(queueKfPredictLeft), std::ref(queueUpdateLabels_left));
                    //right
                    organize(newRoi_right, newLabels_right, frameIndex_right, queueTargetBboxesRight, queueTargetClassIndexesRight, queueTargetFrameIndex_right, seqData_right, kfData_right, kalmanVector_right, queueKfPredictRight, queueUpdateLabels_right);
                    thread_left.join();
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    t_elapsed += t_elapsed + static_cast<float>(duration.count());
                    counter_both++;
                    //std::cout << " Time taken by save both sequential data : " << duration.count() << " microseconds" << std::endl;
                }
                else if (!queueTargetBboxesLeft.empty() && queueTargetBboxesRight.empty()) //left data is available
                {
                    //std::cout << "start saving left" << std::endl;
                    auto start = std::chrono::high_resolution_clock::now();
                    organize(newRoi_left, newLabels_left, frameIndex_left, queueTargetBboxesLeft, queueTargetClassIndexesLeft, queueTargetFrameIndex_left, seqData_left, kfData_left, kalmanVector_left, queueKfPredictLeft, queueUpdateLabels_left);
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    t_elapsed += t_elapsed + static_cast<float>(duration.count());
                    counter_left++;
                    //std::cout << " Time taken by save left sequential data : " << duration.count() << " microseconds" << std::endl;
                }
                else if (queueTargetBboxesLeft.empty() && !queueTargetBboxesRight.empty()) //right data is available
                {
                    //std::cout << "start saving right" << std::endl;
                    auto start = std::chrono::high_resolution_clock::now();
                    organize(newRoi_right, newLabels_right, frameIndex_right, queueTargetBboxesRight, queueTargetClassIndexesRight, queueTargetFrameIndex_right, seqData_right, kfData_right, kalmanVector_right, queueKfPredictRight, queueUpdateLabels_right);
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    t_elapsed += t_elapsed + static_cast<float>(duration.count());
                    counter_right++;
                    //std::cout << " Time taken by save right sequential data : " << duration.count() << " microseconds" << std::endl;
                }
                /* no data available */
                else
                {
                    //nothing to do
                }
            }
        }
        std::cout << "sequential data" << std::endl;
        if ((counter_both + counter_left + counter_right) > 0) std::cout << "process speed :: " << static_cast<int>((counter_both + counter_left + counter_right) / t_elapsed * 1000000) << " Hz for both:" << counter_both << ", left=" << counter_left << ", right=" << counter_right << std::endl;
        std::cout << "LEFT " << std::endl;
        utSeq.checkSeqData(seqData_left, file_seq_left);
        std::cout << "RIGHT " << std::endl;
        utSeq.checkSeqData(seqData_right, file_seq_right);
        std::cout << "Kalman Prediction data" << std::endl;
        std::cout << "LEFT " << std::endl;
        utSeq.checkSeqData(kfData_left, file_kf_left);
        std::cout << "RIGHT " << std::endl;
        utSeq.checkSeqData(kfData_right, file_kf_right);
    }

    void organize(std::vector<cv::Rect2d>& newRoi_left, std::vector<int>& newLabels_left, int& frameIndex_left,
        std::queue<std::vector<cv::Rect2d>>& queueSeqBboxLeft, std::queue<std::vector<int>>& queueSeqClassIndexLeft, std::queue<int>& queueSeqFrameLeft,
        std::vector<std::vector<std::vector<double>>>& seqData_left, std::vector<std::vector<std::vector<double>>>& kfData_left, std::vector<KalmanFilter2D>& kalmanVector_left,
        std::queue<std::vector<std::vector<double>>>& queueuKfPredictLeft, std::queue<std::vector<int>>& queueUpdateLabels)
    {
        //get latest data
        getData(newRoi_left, newLabels_left, frameIndex_left, queueSeqBboxLeft, queueSeqClassIndexLeft, queueSeqFrameLeft);
        //std::cout << "Sequence.h :: finish getting data" << std::endl;
        //update Data
        bool boolKalmanPredict_left = false; //whether kalman prediction is adopted -> if true pusj kalman
        std::vector<std::vector<double>> kf_predictions_left(newLabels_left.size()); //prepare list for kalman prediction
        updateData(newRoi_left, newLabels_left, frameIndex_left, seqData_left, kfData_left, kalmanVector_left, kf_predictions_left, boolKalmanPredict_left);
        //push data
        if (boolKalmanPredict_left)
        {
            if (!queueuKfPredictLeft.empty()) queueuKfPredictLeft.pop();
            queueuKfPredictLeft.push(kf_predictions_left);
        }
        //push data to triangulation
        if (!queueUpdateLabels.empty()) queueUpdateLabels.pop();
        if (!newRoi_left.empty()) queueUpdateLabels.push(newLabels_left);
    }

    void getData(std::vector<cv::Rect2d>& newRoi, std::vector<int>& newLabels, int& frameIndex, std::queue<std::vector<cv::Rect2d>>& queueSeqBbox, std::queue<std::vector<int>>& queueSeqClassIndex, std::queue<int>& queueSeqFrame)
    {
        //get
        if (!queueSeqBbox.empty()) newRoi = queueSeqBbox.front();
        if (!queueSeqClassIndex.empty()) newLabels = queueSeqClassIndex.front();
        if (!queueSeqFrame.empty()) frameIndex = queueSeqFrame.front();
        //pop
        if (!queueSeqBbox.empty()) queueSeqBbox.pop();
        if (!queueSeqClassIndex.empty()) queueSeqClassIndex.pop();
        if (!queueSeqFrame.empty()) queueSeqFrame.pop();
    }

    void updateData(std::vector<cv::Rect2d>& newRoi, std::vector<int>& newLabels, int& frameIndex,
        std::vector<std::vector<std::vector<double>>>& seqData, std::vector<std::vector<std::vector<double>>>& kfData, std::vector<KalmanFilter2D>& kalmanVector,
        std::vector<std::vector<double>>& kf_predictions, bool& boolKalmanPredict)
    {
        int counter_objects = 0; // counter for new detections
        int counter_label = 0; //counter for new labels
        if (!seqData.empty()) //seqData exists
        {
            int num_past_objects = seqData.size();
            //std::cout << "add data to existed seqData :: size=" << num_past_objects << std::endl;
            for (int& label : newLabels) //for each label
            {
                double dframe = 3.0; //frame interval between each detection
                if (counter_label < num_past_objects) // within existed objects
                {
                    //std::cout << "Sequence.h :: seqData.size()="<<num_past_objects<<",counter_label = " << counter_label << "newRoi.size()="<<newRoi.size()<<", counter_objects = " << counter_objects << ", label="<<label<<std::endl;
                    //std::cout << "newRoi.x=" << newRoi[counter_objects].x << std::endl;
                    if (label >= 0) //new sequential data
                    {
                        //add data
                        seqData[counter_label].push_back({ (double)frameIndex,(double)label, newRoi[counter_objects].x,newRoi[counter_objects].y,newRoi[counter_objects].width,newRoi[counter_objects].height });
                        dframe = (double)frameIndex - kfData[counter_label].back()[0];
                        observation << (newRoi[counter_objects].x + newRoi[counter_objects].width / 2), (newRoi[counter_objects].y + newRoi[counter_objects].height / 2);
                        kalmanVector[counter_label].predict(kf_predict, dframe, seqData[counter_label]);
                        kalmanVector[counter_label].update(observation);
                        //kalmanVector[counter_label].predict(kf_predict, dframe, seqData[counter_label]);
                        kfData[counter_label].push_back({ (double)frameIndex,(double)label,(kf_predict[0] - newRoi[counter_objects].width / 2),(kf_predict[1] - newRoi[counter_objects].height / 2),newRoi[counter_objects].width,newRoi[counter_objects].height });
                        //std::cout << "kalmanVector update = " << kalmanVector[counter_label].counter_update << std::endl;
                        if (kalmanVector[counter_label].counter_update >= COUNTER_VALID) //official tracker
                        {
                            kf_predictions[counter_label] = { (double)label,(kf_predict[0] - newRoi[counter_objects].width / 2),(kf_predict[1] - newRoi[counter_objects].height / 2),newRoi[counter_objects].width,newRoi[counter_objects].height }; //add center position
                            boolKalmanPredict = true;
                        }
                        counter_objects++;
                    } //end (if label>=0)
                    else //new data isn't available, label is minus value
                    {
                        if (seqData[counter_label].back()[0] > 0) //still alive tracker -> kalman predict
                        {
                            if (label == -2) //duplicated label
                            {
                                std::cout << "delete duplicated tracker" << std::endl;
                                seqData[counter_label].push_back({ -1,-1,-1,-1,-1,-1 });
                                kalmanVector[counter_label].counter_update = 0; //delete kalman filter model
                                kfData[counter_label].push_back({ -1,-1,-1,-1,-1,-1 });
                            }
                            else //lost data
                            {
                                dframe = (double)frameIndex - kfData[counter_label].back()[0];
                                kalmanVector[counter_label].predict(kf_predict, dframe, seqData[counter_label]);
                                std::vector<double> latest = kfData[counter_label].back(); //{frameIndex,label,left,top,width,height}
                                kfData[counter_label].push_back({ (double)frameIndex,latest[1],(kf_predict[0] - latest[4] / 2),(kf_predict[1] - latest[5] / 2),latest[4],latest[5] });
                                if (bool_addKF) seqData[counter_label].push_back({ (double)frameIndex,(double)latest[1], (kf_predict[0] - latest[4] / 2),(kf_predict[1] - latest[5] / 2),latest[4],latest[5] });
                                if (kalmanVector[counter_label].counter_update >= COUNTER_VALID) //official tracker
                                {
                                    kf_predictions[counter_label] = { latest[1],(kf_predict[0] - latest[4] / 2),(kf_predict[1] - latest[5] / 2),latest[4],latest[5] }; //add center position
                                    boolKalmanPredict = true;
                                }
                            }
                        } //end (if kalman predict)
                    }
                    counter_label++;
                } //end (if within existed objects)

                else //new trackers
                {
                    if (label >= 0) //new tracker available
                    {
                        seqData.push_back({ { (double)frameIndex,(double)label, newRoi[counter_objects].x,newRoi[counter_objects].y,newRoi[counter_objects].width,newRoi[counter_objects].height } }); //add new objects]
                        //make an instance of KalmanFilter2D
                        kalmanVector.push_back(KalmanFilter2D(INIT_X, INIT_Y, INIT_VX, INIT_VY, INIT_AX, INIT_AY, NOISE_POS, NOISE_VEL, NOISE_ACC, NOISE_SENSOR));
                        kalmanVector.back().predict(kf_predict, dframe, defaultVector);
                        observation << (newRoi[counter_objects].x + newRoi[counter_objects].width / 2), (newRoi[counter_objects].y + newRoi[counter_objects].height / 2);
                        //kalmanVector.back().update(observation);
                        kalmanVector.back().predict(kf_predict, dframe, defaultVector);
                        kfData.push_back({ { (double)frameIndex,(double)label,(kf_predict[0] - newRoi[counter_objects].width / 2),(kf_predict[1] - newRoi[counter_objects].height / 2),newRoi[counter_objects].width,newRoi[counter_objects].height} });

                        counter_objects++;
                    }
                    else //tracking was failed
                    {
                        seqData.push_back(fault);
                        kfData.push_back(fault);
                        kalmanVector.push_back(KalmanFilter2D(INIT_X, INIT_Y, INIT_VX, INIT_VY, INIT_AX, INIT_AY, NOISE_POS, NOISE_VEL, NOISE_ACC, NOISE_SENSOR));
                    }
                    counter_label++;
                } //finish else (new trackers)
            }// end for
        }//end if (!seqData.empty())
        else //first detection
        {
            //std::cout << "first detection :: size=" << newLabels.size() <<", newRoi.size="<<newRoi.size() << std::endl;
            for (int& label : newLabels)
            {
                //std::cout << "label = " << label << std::endl;


                double dframe = 10.0;
                if (label >= 0) //new tracker available
                {
                    seqData.push_back({ { (double)frameIndex,(double)label, newRoi[counter_objects].x,newRoi[counter_objects].y,newRoi[counter_objects].width,newRoi[counter_objects].height } }); //add new objects]
                    //make an instance of KalmanFilter2D
                    kalmanVector.push_back(KalmanFilter2D(INIT_X, INIT_Y, INIT_VX, INIT_VY, INIT_AX, INIT_AY, NOISE_POS, NOISE_VEL, NOISE_ACC, NOISE_SENSOR));
                    kalmanVector.back().predict(kf_predict, dframe, defaultVector);
                    observation << (newRoi[counter_objects].x + newRoi[counter_objects].width / 2), (newRoi[counter_objects].y + newRoi[counter_objects].height / 2);
                    kalmanVector.back().update(observation);
                    kfData.push_back({ { (double)frameIndex,(double)label,(kf_predict[0] - newRoi[counter_objects].width / 2),(kf_predict[1] - newRoi[counter_objects].height / 2),newRoi[counter_objects].width,newRoi[counter_objects].height } });
                    counter_objects++;
                }
                else //tracking was failed
                {
                    seqData.push_back(fault);
                    kfData.push_back(fault);
                    kalmanVector.push_back(KalmanFilter2D(INIT_X, INIT_Y, INIT_VX, INIT_VY, INIT_AX, INIT_AY, NOISE_POS, NOISE_VEL, NOISE_ACC, NOISE_SENSOR));
                }
                counter_label++;
            }
        }
    }

};

#endif