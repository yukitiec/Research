#pragma once

#ifndef MATCHING_H
#define MATCHING_H

#include "stdafx.h"


class Matching
{
private:
    const bool debug = true;
    const int dif_threshold = 15; //difference between 2 cams
    const float MAX_ROI_DIF = 2.0; //max roi difference
    const float MIN_ROI_DIF = 0.5;//minimum roi difference

public:
    Matching()
    {
        std::cout << "construct Matching class" << std::endl;
    }

    void main(std::vector<std::vector<std::vector<double>>>& vec_left, std::vector<std::vector<std::vector<double>>>& vec_right, std::vector<int>& classes_left, std::vector<int>& classes_right, std::vector<std::vector<int>>& matching,
        const float& oY_left, const float& oY_right)
    {
        std::vector<std::vector<double>> ball_left, ball_right, box_left, box_right; //storage for data
        std::vector<int> ball_index_left, box_index_left, ball_index_right, box_index_right; //storage for index
        //extract position and index, and then sort -> 300 microseconds for 2 
        arrangeData(vec_left, classes_left, ball_left, box_left, ball_index_left, box_index_left); //sort data
        arrangeData(vec_right, classes_right, ball_right, box_right, ball_index_right, box_index_right); //sort data, retain indexes 
        //matching data in y value
        //ball
        int num_left = ball_index_left.size();
        int num_right = ball_index_right.size();
        auto start_time2 = std::chrono::high_resolution_clock::now();
        // more objects detected in left camera -> 79 microseconds for 2
        matchingObj(ball_left, ball_right, ball_index_left, ball_index_right, oY_left, oY_right, matching);
        //box
        matchingObj(box_left, box_right, box_index_left, box_index_right, oY_left, oY_right, matching);
        if (debug)
        {
            for (int i = 0; i < matching.size(); i++)
                std::cout << i << "-th matching :: left : " << matching[i][0] << ", right: " << matching[i][1] << std::endl;
        }
    }

    void arrangeData(std::vector<std::vector<std::vector<double>>> vec_left, std::vector<int> classes_left, std::vector<std::vector<double>>& ball_left, std::vector<std::vector<double>>& box_left,
        std::vector<int>& ball_index_left, std::vector<int>& box_index_left)
    {
        for (int i = 0; i < classes_left.size(); i++)
        {
            // ball
            if (classes_left[i] == 0)
            {
                ball_left.push_back(vec_left[i].back());
                ball_index_left.push_back(i);
            }
            // box
            else if (classes_left[i] == 1)
            {
                box_left.push_back(vec_left[i].back());
                box_index_left.push_back(i);
            }
        }
        // sort data 
        sortData(ball_left, ball_index_left);
        sortData(box_left, box_index_left);
    }

    void sortData(std::vector<std::vector<double>>& data, std::vector<int>& classes)
    {
        // Create an index array to remember the original order
        std::vector<size_t> index(data.size());
        for (size_t i = 0; i < index.size(); ++i)
        {
            index[i] = i;
        }
        // Sort data1 based on centerX values and apply the same order to data2 : {frameIndex,label,left,top,width,height}
        std::sort(index.begin(), index.end(), [&](size_t a, size_t b)
            { return (data[a][3] + data[a][5] / 2) >= (data[b][3] + data[b][5] / 2); });

        std::vector<std::vector<double>> sortedData(data.size());
        std::vector<int> sortedClasses(classes.size());

        for (size_t i = 0; i < index.size(); ++i)
        {
            sortedData[i] = data[index[i]];
            sortedClasses[i] = classes[index[i]];
        }

        data = sortedData;
        classes = sortedClasses;
    }

    void matchingObj(std::vector<std::vector<double>>& ball_left, std::vector<std::vector<double>>& ball_right, std::vector<int>& ball_index_left, std::vector<int>& ball_index_right,
        const float& oY_left, const float& oY_right, std::vector<std::vector<int>>& matching)
    {
        /**
        * matching objects in both images with label, bbox info. and y-position
        * Args:
        *   ball_left, ball_riht : {frameIndex, label,left,top,width,height}
        * Return:
        *   matching : {{index_left, index_right}}
        */
        int dif_min = dif_threshold;
        int matchIndex_right;
        int i = 0;
        int startIndex = 0; //from which index to start comparison
        //for each object
        while (i < ball_left.size() && startIndex < ball_right.size())
        {
            int j = 0;
            bool boolMatch = false;
            dif_min = dif_threshold;
            std::cout << "startIndex = " << startIndex << std::endl;
            //continue right object y-value is under threshold
            while (true)
            {
                //if right detection is too low -> stop searching
                if (((ball_left[i][3] - ball_right[startIndex + j][3]) > dif_threshold) || (startIndex + j < ball_right.size())) break;
                //bbox info. criteria
                if (((float)ball_left[i][4] * MAX_ROI_DIF >= (float)ball_right[startIndex + j][4]) && ((float)ball_left[i][5] * MAX_ROI_DIF >= (float)ball_right[startIndex + j][5]) &&
                    ((float)ball_left[i][4] * MIN_ROI_DIF <= (float)ball_right[startIndex + j][4]) && ((float)ball_left[i][5] * MIN_ROI_DIF <= (float)ball_right[startIndex + j][5]))
                {
                    std::cout << "dif_in_2imgs=" << std::abs(((float)(ball_left[i][3] + ball_left[i][5] / 2) - oY_left) - ((float)(ball_right[startIndex + j][3] + ball_right[startIndex + j][5] / 2) - oY_right)) << std::endl;
                    int dif = std::abs(((float)(ball_left[i][3] + ball_left[i][5] / 2) - oY_left) - ((float)(ball_right[startIndex + j][3] + ball_right[startIndex + j][5] / 2) - oY_right));
                    if (dif < dif_min)
                    {
                        dif_min = dif;
                        matchIndex_right = startIndex + j;
                        boolMatch = true;
                    }
                }
                j++;
            }
            std::cout << "matching objects found? : " << boolMatch << std::endl;
            //match index is the last value
            if (boolMatch && (matchIndex_right == (startIndex + j - 1))) startIndex += j;
            else startIndex += std::max(j - 1, 0);
            /* matching successful*/
            if (boolMatch)
            {
                matching.push_back({ ball_index_left[i],ball_index_right[matchIndex_right] }); //save matching pair
                //delete selected data
                ball_index_left.erase(ball_index_left.begin() + i);
                ball_left.erase(ball_left.begin() + i);
                //ball_index_right.erase(ball_index_right.begin() + matchIndex_right);
                //ball_right.erase(ball_right.begin() + matchIndex_right);
            }
            // can't find matching object
            else
                i++;
        }
    }

};

#endif 