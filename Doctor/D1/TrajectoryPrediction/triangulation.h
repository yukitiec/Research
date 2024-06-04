#pragma once

#ifndef TRIANGULATION_H
#define TRIANGULATION_H

#include "stdafx.h"
#include "matching.h"
#include "utility.h"


class Triangulation
{
private:
    const cv::Mat cameraMatrix_left = (cv::Mat_<double>(3, 3) << 754.66874569, 0, 255.393104, // fx: focal length in x, cx: principal point x
        0, 754.64708568, 335.6848201,                           // fy: focal length in y, cy: principal point y
        0, 0, 1                                // 1: scaling factor
        );
    const cv::Mat cameraMatrix_right = (cv::Mat_<double>(3, 3) << 802.62616415, 0, 286.48516862, // fx: focal length in x, cx: principal point x
        0, 802.15806832, 293.54957668,                           // fy: focal length in y, cy: principal point y
        0, 0, 1                                // 1: scaling factor
        );
    const cv::Mat distCoeffs_left = (cv::Mat_<double>(1, 5) << -0.00661832, - 0.19633213,  0.00759942, - 0.01391234,  0.73355661);
    const cv::Mat distCoeffs_right = (cv::Mat_<double>(1, 5) << 0.00586444, - 0.18180071,  0.00489287, - 0.00392576,  1.20394993);
    const cv::Mat projectMatrix_left = (cv::Mat_<double>(3, 4) << 375.5, 0, 249.76, 0, // fx: focal length in x, cx: principal point x
        0, 375.5, 231.0285, 0,                           // fy: focal length in y, cy: principal point y
        0, 0, 1, 0                                // 1: scaling factor
        );
    const cv::Mat projectMatrix_right = (cv::Mat_<double>(3, 4) << 375.5, 0, 249.76, -280, // fx: focal length in x, cx: principal point x
        0, 375.5, 231.028, 0,                           // fy: focal length in y, cy: principal point y
        0, 0, 1, 0                               // 1: scaling factor
        );
    const cv::Mat transform_cam2base = (cv::Mat_<double>(4, 4) << 1.0, 0.0, 0.0, 0.0, // fx: focal length in x, cx: principal point x
        0.0, 1.0, 0.0, 0.0,                           // fy: focal length in y, cy: principal point y
        0.0, 0.0, 1.0, 0.0 ,                              // 1: scaling factor
        0.0, 0.0, 0.0,1.0
        );
    const double fX = (cameraMatrix_left.at<double>(0, 0)+cameraMatrix_right.at<double>(0,0))/2;
    const double fY = (cameraMatrix_left.at<double>(1, 1) + cameraMatrix_right.at<double>(1, 1)) / 2;
    const double fSkew = (cameraMatrix_left.at<double>(0, 1) + cameraMatrix_right.at<double>(0, 1)) / 2;
    const double oX_left = cameraMatrix_left.at<double>(0,2);
    const double oX_right = cameraMatrix_right.at<double>(0, 2);
    const double oY_left = cameraMatrix_left.at<double>(1, 2);
    const double oY_right = cameraMatrix_right.at<double>(1, 2);
    const double BASELINE = 280.0;
    const int numJoint = 6; //number of joints
    //which method to triangulate 3D points
    const int method_triangulate = 1; //0 : DLT, 1: stereo method
    const int numObjects = 100;
    const double threshold_difference_perFrame = 300;//max speed per frame is 300 mm/frame
    const int counter_valid = 3; //num of minimum counter for valid data

public:
    int num_obj_left = 0;
    int num_obj_right = 0;

    Triangulation()
    {
        std::cout << "construct Triangulation class" << std::endl;
    }

    void main();

    bool compareVectors(const std::vector<int>& a, const std::vector<int>& b);

    void sortData(std::vector<std::vector<int>>& data);

    void triangulation(std::vector<std::vector<std::vector<double>>>& data_left, std::vector<std::vector<std::vector<double>>>& data_right, std::vector<std::vector<int>>& matchingIndexes, std::vector<std::vector<std::vector<double>>>& data_3d);

    void calulate3Ds(int& index, std::vector<std::vector<double>>& left, std::vector<std::vector<double>>& right, std::vector<std::vector<std::vector<double>>>& data_3d);

    void dlt(std::vector<cv::Point2d>& points_left, std::vector<cv::Point2d>& points_right, std::vector<cv::Point3d>& results);

    void stereo3D(std::vector<cv::Point2d>& left, std::vector<cv::Point2d>& right, std::vector<cv::Point3d>& results);
};

#endif