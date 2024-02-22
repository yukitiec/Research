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
    const cv::Mat cameraMatrix_left = (cv::Mat_<float>(3, 3) << 375.1248, 0, 220.745, // fx: focal length in x, cx: principal point x
        0, 374.53, 223.9,                           // fy: focal length in y, cy: principal point y
        0, 0, 1                                // 1: scaling factor
        );
    const cv::Mat cameraMatrix_right = (cv::Mat_<float>(3, 3) << 376.18, 0, 236.08, // fx: focal length in x, cx: principal point x
        0, 376.48, 237.6,                           // fy: focal length in y, cy: principal point y
        0, 0, 1                                // 1: scaling factor
        );
    const cv::Mat distCoeffs_left = (cv::Mat_<float>(1, 5) << -0.04646561, 0.07635738, 0.0005208, -0.00482826, -0.0215558);
    const cv::Mat distCoeffs_right = (cv::Mat_<float>(1, 5) << -0.0688505, 0.13593516, 0.00094933, 0.0013044, -0.09112549);
    const cv::Mat projectMatrix_left = (cv::Mat_<float>(3, 4) << 375.5, 0, 249.76, 0, // fx: focal length in x, cx: principal point x
        0, 375.5, 231.0285, 0,                           // fy: focal length in y, cy: principal point y
        0, 0, 1, 0                                // 1: scaling factor
        );
    const cv::Mat projectMatrix_right = (cv::Mat_<float>(3, 4) << 375.5, 0, 249.76, -280, // fx: focal length in x, cx: principal point x
        0, 375.5, 231.028, 0,                           // fy: focal length in y, cy: principal point y
        0, 0, 1, 0                               // 1: scaling factor
        );
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

    void main()
    {
        Utility utTri;//constructor
        while (true)
        {
            if (!queueTriangulation_left.empty() && !queueTriangulation_right.empty()) break;
            //std::cout << "wait for target data" << std::endl;
        }
        std::cout << "start calculating 3D position" << std::endl;


        std::vector<std::vector<std::vector<std::vector<int>>>> posSaver_3d; //{num of sequence, num of human, joints, {frameIndex, X,Y,Z}}
        int counterIteration = 0;
        int counterFinish = 0;
        int counterNextIteration = 0;
        int frameIndex;
        while (true) // continue until finish
        {
            counterIteration++;
            if (queueFrame.empty() && queueTriangulation_left.empty() && queueTriangulation_right.empty())
            {
                if (counterFinish == 10) break;
                counterFinish++;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                std::cout << "By finish : remain count is " << (10 - counterFinish) << std::endl;
                continue;
            }
            else
            {
                /* new detection data available */
                if (!queueTriangulation_left.empty() && !queueTriangulation_right.empty())
                {
                    counterNextIteration = 0;
                    auto start = std::chrono::high_resolution_clock::now();
                    std::vector<std::vector<std::vector<int>>> data_left, data_right; //[num of human, joints, [frameIndex, x,y]]
                    //std::cout << "start 3d positioning" << std::endl;
                    getData(data_left, data_right);
                    //std::cout << "start" << std::endl;
                    std::vector<std::vector<cv::Point3f>> data_3d; //{num of human, joints, {frameIndex, X,Y,Z}}
                    std::vector<std::vector<bool>> boolDetects;
                    triangulation(data_left, data_right, data_3d, boolDetects, frameIndex);
                    //std::cout << "calculate 3d position" << std::endl;
                    /* arrange posSaver -> sequence data */
                    if (!data_3d.empty())
                    {
                        arrangeData(data_3d, boolDetects, frameIndex, posSaver_3d);
                        queueJointsPositions.push(posSaver_3d.back());
                        //std::cout << "arrange data" << std::endl;
                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                        std::cout << "!!!! time taken by 3d positioning=" << duration.count() << " microseconds !!!!" << std::endl;
                    }
                }
                /* at least one data can't be available -> delete data */
                else
                {
                    //std::cout << "both data can't be availble :: left " << !queueTriangulation_left.empty() << ", right=" << queueTriangulation_right.empty() << std::endl;
                    if (!queueTriangulation_left.empty() || !queueTriangulation_right.empty())
                    {
                        if (counterNextIteration == 10)
                        {
                            if (!queueTriangulation_left.empty()) queueTriangulation_left.pop();
                            if (!queueTriangulation_right.empty()) queueTriangulation_right.pop();
                            counterNextIteration = 0;
                        }
                        else
                        {
                            std::this_thread::sleep_for(std::chrono::milliseconds(2));
                            counterNextIteration++;
                        }

                    }
                    counterFinish = 0;
                }
            }
        }
        std::cout << "***triangulation data***" << std::endl;
        utTri.save3d(posSaver_3d, file_3d);
    }

    void getData(std::vector<std::vector<std::vector<int>>>& data_left, std::vector<std::vector<std::vector<int>>>& data_right)
    {
        if (!queueTriangulation_left.empty() && !queueTriangulation_right.empty())
        {
            data_left = queueTriangulation_left.front();
            data_right = queueTriangulation_right.front();
            queueTriangulation_left.pop();
            queueTriangulation_right.pop();
        }
    }

    void dlt(std::vector<cv::Point2f>& points_left, std::vector<cv::Point2f>& points_right, std::vector<cv::Point3f>& results)
    {
        cv::Mat points_left_mat(points_left);
        cv::Mat undistorted_points_left_mat;
        cv::Mat points_right_mat(points_right);
        cv::Mat undistorted_points_right_mat;

        // Undistort the points
        cv::undistortPoints(points_left_mat, undistorted_points_left_mat, cameraMatrix_left, distCoeffs_left);
        cv::undistortPoints(points_right_mat, undistorted_points_right_mat, cameraMatrix_right, distCoeffs_right);

        // Output matrix for the 3D points
        cv::Mat triangulated_points_mat;

        // Triangulate points
        cv::triangulatePoints(projectMatrix_left, projectMatrix_right, undistorted_points_left_mat, undistorted_points_right_mat, triangulated_points_mat);

        // Convert homogeneous coordinates to 3D points
        cv::convertPointsFromHomogeneous(triangulated_points_mat.t(), triangulated_points_mat);
        // Access triangulated 3D points
        results = triangulated_points_mat;
    }

    void triangulation(std::vector<std::vector<std::vector<int>>>& data_left, std::vector<std::vector<std::vector<int>>>& data_right,
        std::vector<std::vector<cv::Point3f>>& data_3d, std::vector<std::vector<bool>>& boolDetects, int& frameIndex)
    {
        int numHuman = std::min(data_left.size(), data_right.size());
        // for each human
        for (int i = 0; i < numHuman; i++)
        {
            std::vector<cv::Point3f> temp;
            std::vector<cv::Point2f> left, right;
            // for each joint
            if (data_left[i][0][0] == data_right[i][0][0])
            {
                frameIndex = data_left[i][0][0];
                //std::cout << "data left:" << data_left[i].size() << ", right: " << data_right[i].size() << std::endl;
                for (int j = 0; j < numJoint; j++)
                {
                    //std::cout << "cal3D" << std::endl;
                    //std::cout << "left size="<<data_left[i][j].size()<<"right size="<<data_right[i][j].size() << std::endl;
                    //std::cout << "num human:" << i << ", num joints:" << j << std::endl;
                    left.emplace_back(((float)data_left[i][j][1], (float)data_left[i][j][2]));
                    right.emplace_back(((float)data_right[i][j][1], (float)data_right[i][j][2]));
                    //std::cout << "push data" << std::endl;
                    if (data_left[i][j][1] > 0 && data_right[i][j][1] > 0)
                    {
                        if (j == 0)
                            boolDetects.push_back({ true });
                        else
                            boolDetects[i].push_back(true);
                    }
                    else
                    {
                        if (j == 0)
                            boolDetects.push_back({ false });
                        else
                            boolDetects[i].push_back(false);
                    }
                }
                dlt(left, right, temp);
                data_3d.push_back(temp);
            }
        }
    }

    void cal3D(std::vector<int>& left, std::vector<int>& right, std::vector<int>& result)
    {
        //both joints is detected
        if (left[1] != -1 && right[1] != -1)
        {
            // frameIndex is same
            if (left[0] == right[0])
            {
                int xl = left[1]; int xr = right[1]; int yl = left[2]; int yr = right[2];
                float disparity = xl - xr + epsiron;
                if (disparity < 0.1) disparity += epsiron;
                int X = (int)(BASELINE / disparity) * (xl - oX - (fSkew / fY) * ((yl + yr) / 2 - oY));
                int Y = (int)(BASELINE * (fX / fY) * ((yl + yr) / 2 - oY) / disparity);
                int Z = (int)(fX * BASELINE / disparity);
                /* convert Camera coordinate to robot base coordinate */
                X = static_cast<int>(transform_cam2base[0][0] * X + transform_cam2base[0][1] * Y + transform_cam2base[0][2] * Z + transform_cam2base[0][3]);
                Y = static_cast<int>(transform_cam2base[1][0] * X + transform_cam2base[1][1] * Y + transform_cam2base[1][2] * Z + transform_cam2base[1][3]);
                Z = static_cast<int>(transform_cam2base[2][0] * X + transform_cam2base[2][1] * Y + transform_cam2base[2][2] * Z + transform_cam2base[2][3]);
                result = std::vector<int>{ left[0],X,Y,Z };
            }
            else
            {
                std::cout << "frameIndex is different" << std::endl;
                int X = -1; int Y = -1; int Z = -1;
                result = std::vector<int>{ -1,X,Y,Z };
            }
        }
        // at least one isn't detected
        else
        {
            std::cout << "frameIndex is different" << std::endl;
            int X = -1; int Y = -1; int Z = -1;
            result = std::vector<int>{ -1,X,Y,Z };
        }

    }

    void arrangeData(std::vector<std::vector<cv::Point3f>>& data_3d, std::vector<std::vector<bool>>& boolDetects, int& frameIndex, std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver)
    {
        // already first 3d calculation done
        //std::cout << "posSaver.size()=" << posSaver.size() << std::endl;
        std::vector<std::vector<std::vector<int>>> all; //all human data
        std::vector<int> pastData;
        if (!posSaver.empty())
        {
            //std::cout << "data3d.size()=" << data_3d.size() << std::endl;
            // for each human
            for (int i = 0; i < data_3d.size(); i++)
            {
                std::vector<std::vector<int>> tempHuman;
                /* same human */
                if (posSaver.back().size() > i)
                {
                    // for each joint
                    for (int j = 0; j < data_3d[i].size(); j++)
                    {
                        // detected
                        if (boolDetects[i][j]) //detected
                        {
                            tempHuman.push_back({ frameIndex, (int)data_3d[i][j].x,(int)data_3d[i][j].y,(int)data_3d[i][j].z });
                        }
                        // not detected
                        else
                        {
                            // already detected
                            if (posSaver.back()[i][j][0] > 0)
                            {
                                pastData = posSaver.back()[i][j];
                                pastData[0] = frameIndex;
                                tempHuman.push_back(pastData); //adapt last detection
                            }
                            // not detected yet
                            else
                            {
                                tempHuman.push_back({ -1,-1,-1,-1 }); //-1
                            }
                        }
                    }
                }
                //new human
                else
                {
                    // for each joint
                    for (int j = 0; j < data_3d[i].size(); j++)
                    {
                        // detected
                        if (boolDetects[i][j]) //detected
                        {
                            tempHuman.push_back({ frameIndex, (int)data_3d[i][j].x,(int)data_3d[i][j].y,(int)data_3d[i][j].z });
                        }
                        // not detected
                        else
                        {
                            tempHuman.push_back({ -1,-1,-1,-1 }); //-1
                        }
                    }
                }
                all.push_back(tempHuman); //push human data
            }
            posSaver.push_back(all);
        }
        // first detection
        else
        {
            for (int i = 0; i < data_3d.size(); i++) //for each human
            {
                std::vector<std::vector<int>> tempHuman;
                // for each joint
                for (int j = 0; j < data_3d[i].size(); j++) //for each joint
                {
                    // detected
                    if (boolDetects[i][j]) //detected
                    {
                        tempHuman.push_back({ frameIndex, (int)data_3d[i][j].x,(int)data_3d[i][j].y,(int)data_3d[i][j].z });
                    }
                    // not detected
                    else
                    {
                        tempHuman.push_back({ -1,-1,-1,-1 }); //-1
                    }
                }
                all.push_back(tempHuman); //push human data
            }
            posSaver.push_back(all);
        }
    }

};

#endif