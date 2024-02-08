#pragma once

#ifndef TRIANGULATION_H
#define TRIANGULATION_H

#include "stdafx.h"
#include "global_parameters.h"
#include "utility.h"
#include "matching.h"

//3d storage
extern const int numObjects;

extern const std::string file_3d;

//Yolo detect signal
extern std::queue<bool> queueYolo_seq2tri_left, queueYolo_seq2tri_right;
extern std::queue<bool> queue_tri2predict;
std::queue<std::vector<std::vector<int>>> queueMatchingIndexes; //for saving matching indexes


/*3D position*/
extern std::vector<std::vector<std::vector<double>>> seqData_left, seqData_right; //storage for sequential data to share with triangulation.h
extern std::queue<std::vector<int>> queueUpdateLabels_left;
extern std::queue<std::vector<int>> queueUpdateLabels_right;
/* for predict */
extern std::queue< std::vector<std::vector<std::vector<int>>>> queue3DData;
/* 3D triangulation */
extern const int BASELINE; // distance between 2 cameras
extern const std::vector<std::vector<float>> transform_cam2base;
// std::vector<std::vector<float>> cameraMatrix{ {179,0,160},{0,179,160},{0,0,1} }; //camera matrix from camera calibration

/* revise here based on camera calibration */
extern const cv::Mat cameraMatrix;
extern const cv::Mat distCoeffs;
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
    const float oY_left = cameraMatrix_left.at<float>(1, 2);
    const float oY_right = cameraMatrix_right.at<float>(1, 2);
    const float fX = cameraMatrix.at<double>(0, 0);
    const float fY = cameraMatrix.at<double>(1, 1);
    const float fSkew = cameraMatrix.at<double>(0, 1);
    const float oX = cameraMatrix.at<double>(0, 2);
    const float oY = cameraMatrix.at<double>(1, 2);
    const int numJoint = 6; //number of joints
public:
    int num_past_class_left = 0;
    int num_past_class_right = 0;
    int num_new_class_left, num_new_class_right;

    Triangulation()
    {
        std::cout << "construct Triangulation class" << std::endl;
    }

    void main()
    {
        Utility utTri;//constructor
        Matching match; //matching algorithm
        std::vector<std::vector<std::vector<int>>> data_3d(numObjects); //{num of objects, sequential, { frameIndex, X,Y,Z }}
        while (true)
        {
            if (!queueUpdateLabels_left.empty() || !queueUpdateLabels_right.empty()) break;
            //std::cout << "wait for target data" << std::endl;
        }
        std::cout << "start calculating 3D position" << std::endl;

        int counterIteration = 0;
        int counterFinish = 0;
        int counter = 0;//counter before delete new data
        while (true) // continue until finish
        {
            counterIteration++;
            if (queueFrame.empty() && queueUpdateLabels_left.empty() && queueUpdateLabels_right.empty())
            {
                if (counterFinish == 10) break;
                counterFinish++;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                std::cout << "By finish : remain count is " << (10 - counterFinish) << std::endl;
                continue;
            }
            else
            {
                counterFinish = 0;
                /* new detection data available */
                if (!queueUpdateLabels_left.empty() && !queueUpdateLabels_right.empty())
                {
                    counter = 0; //reset
                    auto start = std::chrono::high_resolution_clock::now();
                    std::cout << "start 3d positioning" << std::endl;
                    //get latest classes
                    std::vector<int> labels_left = queueUpdateLabels_left.front(); queueUpdateLabels_left.pop();
                    std::vector<int> labels_right = queueUpdateLabels_right.front(); queueUpdateLabels_right.pop();
                    std::vector<std::vector<int>> matchingIndexes; //list for matching indexes
                    //matching when number of labels increase
                    if (labels_left.size() > num_past_class_left || labels_right.size() > num_past_class_right)
                    {
                        std::cout << "update matching" << std::endl;;
                        if (!queueMatchingIndexes.empty()) queueMatchingIndexes.pop();
                        match.main(seqData_left, seqData_right, labels_left, labels_right, matchingIndexes, oY_left, oY_right); //matching objects in 2 images
                        if (!matchingIndexes.empty()) sortData(matchingIndexes); //sort data in ascending way, 0,1,2,3,4,,,
                    }
                    //use previous matching indexes
                    else if (!queueMatchingIndexes.empty())
                    {
                        matchingIndexes = queueMatchingIndexes.front();
                        queueMatchingIndexes.pop();
                    }
                    //calculate 3d positions based on matchingIndexes
                    if (!matchingIndexes.empty())
                    {
                        std::cout << "start 3D positioning" << std::endl;
                        triangulation(seqData_left, seqData_right, matchingIndexes, data_3d);
                        queue_tri2predict.push(true);
                        queue3DData.push(data_3d);
                        queueMatchingIndexes.push(matchingIndexes);
                        std::cout << "calculate 3d position" << std::endl;
                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                        std::cout << "time taken by 3d positioning=" << duration.count() << " microseconds" << std::endl;
                    }
                }
                /* at least one data can't be available -> delete data */
                else
                {
                    if (!queueUpdateLabels_left.empty() || !queueUpdateLabels_right.empty())
                    {
                        //std::cout << "both data can't be availble :: left " << !queueTriangulation_left.empty() << ", right=" << queueTriangulation_right.empty() << std::endl;
                        if ((!queueUpdateLabels_left.empty() || !queueUpdateLabels_right.empty()) && counter == 10)
                        {
                            if (!queueUpdateLabels_left.empty()) queueUpdateLabels_left.pop();
                            if (!queueUpdateLabels_right.empty()) queueUpdateLabels_right.pop();
                            counter = 0;
                        }
                        else
                        {
                            std::this_thread::sleep_for(std::chrono::microseconds(200));
                            counter++;
                        }
                    }
                }
            }
        }
        std::cout << "***triangulation data***" << std::endl;
        utTri.save3d(data_3d, file_3d);
    }

    bool compareVectors(const std::vector<int>& a, const std::vector<int>& b)
    {
        // Compare based on the first element of each vector
        return a[0] < b[0];
    }

    void sortData(std::vector<std::vector<int>>& data)
    {
        std::sort(data.begin(), data.end(), [this](const std::vector<int>& a, const std::vector<int>& b) {
            return compareVectors(a, b);
            });
    }

    void triangulation(std::vector<std::vector<std::vector<double>>>& data_left, std::vector<std::vector<std::vector<double>>>& data_right, std::vector<std::vector<int>>& matchingIndexes, std::vector<std::vector<std::vector<int>>>& data_3d)
    {
        //for all matching data
        for (std::vector<int>& matchIndex : matchingIndexes)
        {
            int index = matchIndex[0]; //left object's index
            //calculate objects
            std::vector<std::vector<double>> left = data_left[matchIndex[0]]; //{num frames, {frameIndex, classLabel,left,top,width,height}}
            std::vector<std::vector<double>> right = data_right[matchIndex[1]]; //{num frames, {frameIndex, classLabel,left,top,width,height}}
            calulate3Ds(index, left, right, data_3d); //{num of objects, sequential, {frameIndex, X,Y,Z}}
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

    void calulate3Ds(int& index, std::vector<std::vector<double>>& left, std::vector<std::vector<double>>& right, std::vector<std::vector<std::vector<int>>>& data_3d)
    {
        std::vector<std::vector<int>> temp;
        int left_frame_start = left[0][0];
        int right_frame_start = right[0][0];
        int num_frames_left = left.size();
        int num_frames_right = right.size();
        int it_left = 0; int it_right = 0;
        //check whether previous data is in data_3d -> if exist, check last frame Index
        std::vector<cv::Point3f> results;
        std::vector<cv::Point2f> pts_left, pts_right;
        std::vector<double> frames;
        if (!data_3d[index].empty())
        {
            int last_frameIndex = (data_3d[index].back())[0]; //get last frame index
            //calculate 3d position for all left data
            //gather frame-matched data 
            while (it_left < num_frames_left && it_right < num_frames_right)
            {
                if ((left[it_left][0] == -1) && (left[it_left][0] == right[it_right][0]) && left[it_left][0] > last_frameIndex)
                {
                    std::vector<int> result;
                    pts_left.emplace_back(((float)(left[it_left][2] + left[it_left][4] / 2), (float)(left[it_left][3] + left[it_left][5] / 2)));
                    pts_right.emplace_back(((float)(left[it_right][2] + left[it_right][4] / 2), (float)(left[it_right][3] + left[it_right][5] / 2)));
                    frames.push_back(left[it_left][0]);
                    it_left++;
                    it_right++;
                }
                else if (left[it_left][0] > right[it_right][0]) it_right++;
                else if (left[it_left][0] < right[it_right][0]) it_left++;
            }
            //triangulate 
            if (!pts_left.empty())
            {
                dlt(pts_left, pts_right, results);
                //push results to data_3d :: {frameIndex, x,y,z}
                if (!results.empty())
                {
                    int counter = 0;
                    std::vector<int> newData;
                    for (cv::Point3f& result : results)
                    {
                        newData.push_back((int)frames[counter]);
                        newData.push_back((int)result.x);
                        newData.push_back((int)result.y);
                        newData.push_back((int)result.z);
                        data_3d.at(index).push_back(newData);
                    }
                }
            }
        }
        //no previous data
        else
        {
            //calculate 3d position for all left data
            std::vector<std::vector<int>> temp_3d;
            while (it_left < num_frames_left && it_right < num_frames_right)
            {
                if ((left[it_left][0] == -1) && (left[it_left][0] == right[it_right][0]))
                {
                    std::vector<int> result;
                    pts_left.emplace_back(((float)(left[it_left][2] + left[it_left][4] / 2), (float)(left[it_left][3] + left[it_left][5] / 2)));
                    pts_right.emplace_back(((float)(left[it_right][2] + left[it_right][4] / 2), (float)(left[it_right][3] + left[it_right][5] / 2)));
                    frames.push_back(left[it_left][0]);
                    it_left++;
                    it_right++;
                }
                else if (left[it_left][0] > right[it_right][0]) it_right++;
                else if (left[it_left][0] < right[it_right][0]) it_left++;
            }
            //triangulate 
            if (!pts_left.empty())
            {
                dlt(pts_left, pts_right, results);
                //push results to data_3d :: {frameIndex, x,y,z}
                if (!results.empty())
                {
                    int counter = 0;
                    std::vector<int> newData;
                    for (cv::Point3f& result : results)
                    {
                        newData.push_back((int)frames[counter]);
                        newData.push_back((int)result.x);
                        newData.push_back((int)result.y);
                        newData.push_back((int)result.z);
                        data_3d.at(index).push_back(newData);
                    }
                }
            }
        }
    }

    void cal3D(std::vector<double>& left, std::vector<double>& right, std::vector<int>& result)
    {
        double xl = left[2]; int xr = right[2]; int yl = left[3]; int yr = right[3];
        int disparity = (int)(xl - xr);
        int X = (int)(BASELINE / disparity) * (xl - oX - (fSkew / fY) * (yl - oY));
        int Y = (int)(BASELINE * (fX / fY) * (yl - oY) / disparity);
        int Z = (int)(fX * BASELINE / disparity);
        /* convert Camera coordinate to robot base coordinate */
        X = static_cast<int>(transform_cam2base[0][0] * X + transform_cam2base[0][1] * Y + transform_cam2base[0][2] * Z + transform_cam2base[0][3]);
        Y = static_cast<int>(transform_cam2base[1][0] * X + transform_cam2base[1][1] * Y + transform_cam2base[1][2] * Z + transform_cam2base[1][3]);
        Z = static_cast<int>(transform_cam2base[2][0] * X + transform_cam2base[2][1] * Y + transform_cam2base[2][2] * Z + transform_cam2base[2][3]);
        result = std::vector<int>{ (int)left[0],X,Y,Z };
    }

};

#endif