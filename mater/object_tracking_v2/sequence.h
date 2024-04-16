#pragma once

#ifndef SEQUENCE_H
#define SEQUENCE_H

#include "stdafx.h"
#include "utility.h"
#include "kalmanfilter.h"
#include "global_parameters.h"


//for sharing data with triangulation
extern std::vector<std::vector<std::vector<double>>> seqData_left, seqData_right;

//tracker2seq
extern std::queue<Tracker2seq> q_tracker2seq_left, q_tracker2seq_right;
//seq2tracker
extern std::queue<std::vector<std::vector<double>>> q_seq2tracker_left, q_seq2tracker_right;


//latest labels for matching data in both images
extern std::queue<std::vector<int>> queueUpdateLabels_left;
extern std::queue<std::vector<int>> queueUpdateLabels_right;

class Sequence
{
private:
    //deault vector
    std::vector<std::vector<double>> defaultVector{ {} }; //default vector for kalman filter initialization
    std::vector<std::vector<double>> fault{ {-1,-1,-1,-1,-1,-1} }; //fault vector for seqData if first tracking was failed : {frameIndex, label, left, top, width, height}
    //obesrvation vector for kalman filter
    Eigen::Vector2d observation;
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
    };

    ~Sequence() {};

    //main
    void main();

    //organize data
    void organize(int& frameIndex, std::vector<int>& newLabels, std::vector<cv::Rect2d>& newRoi,
        std::queue<Tracker2seq>& q_tracker2seq,
        std::vector<std::vector<std::vector<double>>>& seqData, std::vector<std::vector<std::vector<double>>>& kfData, std::vector<KalmanFilter2D>& kalmanVector,
        std::queue<std::vector<std::vector<double>>>& q_seq2tracker, std::queue<std::vector<int>>& queueUpdateLabels);

    //get data from Tracker2seq
    void getData(int& frameIndex, std::vector<int>& newLabels, std::vector<cv::Rect2d>& newRoi, std::queue<Tracker2seq>& q_tracker2seq);
    
    //update storage
    void updateData(std::vector<cv::Rect2d>& newRoi, std::vector<int>& newLabels, int& frameIndex,
        std::vector<std::vector<std::vector<double>>>& seqData, std::vector<std::vector<std::vector<double>>>& kfData, std::vector<KalmanFilter2D>& kalmanVector,
        std::vector<std::vector<double>>& kf_predictions, bool& boolKalmanPredict);
};

#endif
