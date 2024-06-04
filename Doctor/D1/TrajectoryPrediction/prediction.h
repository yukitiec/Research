#pragma once

#ifndef PREDICTION_H
#define PREDICTION_H

#include "stdafx.h"
#include "utility.h"


class Prediction
{
private:
    const int numObjects = 100;
    const cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 754.66874569, 0, 255.393104, // fx: focal length in x, cx: principal point x
        0, 754.64708568, 335.6848201,                           // fy: focal length in y, cy: principal point y
        0, 0, 1                                // 1: scaling factor
        );
    const cv::Mat cameraMatrix_right = (cv::Mat_<double>(3, 3) << 802.62616415, 0, 286.48516862, // fx: focal length in x, cx: principal point x
        0, 802.15806832, 293.54957668,                           // fy: focal length in y, cy: principal point y
        0, 0, 1                                // 1: scaling factor
        );
    const cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << -0.00661832, -0.19633213, 0.00759942, -0.01391234, 0.73355661);
    const cv::Mat distCoeffs_right = (cv::Mat_<double>(1, 5) << 0.00586444, -0.18180071, 0.00489287, -0.00392576, 1.20394993);
    const cv::Mat projectMatrix = (cv::Mat_<double>(3, 4) << 375.5, 0, 249.76, 0, // fx: focal length in x, cx: principal point x
        0, 375.5, 231.0285, 0,                           // fy: focal length in y, cy: principal point y
        0, 0, 1, 0                                // 1: scaling factor
        );
    const cv::Mat projectMatrix_right = (cv::Mat_<double>(3, 4) << 375.5, 0, 249.76, 280, // fx: focal length in x, cx: principal point x
        0, 375.5, 231.028, 42,                           // fy: focal length in y, cy: principal point y
        0, 0, 1, 0                               // 1: scaling factor
        );
    const double BASELINE = 208;
    //number of points for predicting trajectory
    const double N_POINTS_PREDICT = 15.0;
public:
    std::vector<double> coefX, coefY, coefZ; //predicted trajectory
    Prediction()
    {
        std::cout << "construct Prediction class" << std::endl;
    }

    void main();

    void predictTargets(int& index, double& depth_target, std::vector<std::vector<double>>& data, std::vector<std::vector<std::vector<double>>>& targets3D);

    void linearRegression(std::vector<std::vector<double>>& data, std::vector<double>& result_x);

    void linearRegressionZ(std::vector<std::vector<double>>& data, std::vector<double>& result_z);

    void curveFitting(std::vector<std::vector<double>>& data, std::vector<double>& result);

    //from here, for matching objects in 2 cameras
    void trajectoryPredict2D(std::vector<std::vector<std::vector<double>>>& dataLeft, std::vector<std::vector<double>>& coefficientsX, std::vector<std::vector<double>>& coefficientsY, std::vector<int>& classesLatest);

    double calculateME(std::vector<double>& coefXLeft, std::vector<double>& coefYLeft, std::vector<double>& coefXRight, std::vector<double>& coefYRight);

    void dataMatching(std::vector<std::vector<double>>& coefficientsXLeft, std::vector<std::vector<double>>& coefficientsXRight,
        std::vector<std::vector<double>>& coefficientsYLeft, std::vector<std::vector<double>>& coefficientsYRight,
        std::vector<int>& classesLatestLeft, std::vector<int>& classesLatestRight,
        std::vector<std::vector<std::vector<int>>>& dataLeft, std::vector<std::vector<std::vector<int>>>& dataRight,
        std::vector<std::vector<std::vector<std::vector<int>>>>& dataFor3D);

    void predict3DTargets(double& depth_target, std::vector<std::vector<std::vector<std::vector<double>>>>& datasFor3D, std::vector<std::vector<double>>& targets3D);

};

#endif

