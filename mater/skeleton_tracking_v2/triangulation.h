#pragma once

#ifndef TRIANGULATION_H
#define TRIANGULATION_H

#include "stdafx.h"
#include "global_parameters.h"
#include "utility.h"

/*3D position*/
extern std::queue<std::vector<std::vector<std::vector<int>>>> queueTriangulation_left;
extern std::queue<std::vector<std::vector<std::vector<int>>>> queueTriangulation_right;

/* from joints to robot control */
extern std::queue<std::vector<std::vector<std::vector<int>>>> queueJointsPositions;

/* 3D triangulation */
extern const int BASELINE; // distance between 2 cameras
// std::vector<std::vector<float>> cameraMatrix{ {179,0,160},{0,179,160},{0,0,1} }; //camera matrix from camera calibration

/* revise here based on camera calibration */
extern const cv::Mat cameraMatrix;
extern const cv::Mat distCoeffs;
/* transformation matrix from camera coordinate to robot base coordinate */
extern const std::vector<std::vector<float>> transform_cam2base;
/* save file*/
extern const std::string file_3d;

class Triangulation
{
private:
    const float fX = cameraMatrix.at<float>(0, 0);
    const float fY = cameraMatrix.at<float>(1, 1);
    const float fSkew = cameraMatrix.at<float>(0, 1);
    const float oX = cameraMatrix.at<float>(0, 2);
    const float oY = cameraMatrix.at<float>(1, 2);
    const int numJoint = 6; //number of joints
    const float epsiron = 0.001;
public:
    Triangulation()
    {
        std::cout << "construct Triangulation class" << std::endl;
    }

    void main();

    void getData(std::vector<std::vector<std::vector<int>>>& data_left, std::vector<std::vector<std::vector<int>>>& data_right);

    void triangulation(std::vector<std::vector<std::vector<int>>>& data_left, std::vector<std::vector<std::vector<int>>>& data_right,
        std::vector<std::vector<std::vector<int>>>& data_3d);

    void cal3D(std::vector<int>& left, std::vector<int>& right, std::vector<int>& result);

    void arrangeData(std::vector<std::vector<std::vector<int>>>& data_3d, std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver);
};

#endif