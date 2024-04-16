#pragma once

#ifndef UTILITY_H
#define UTILITY_H

#include "stdafx.h"


extern std::queue<std::vector<std::vector<cv::Rect2i>>> queueYoloSearchRoi_left;        // queue for search roi for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Rect2i>>> queueYoloSearchRoi_right;        // queue for search roi for optical flow. vector size is [num human,6]

class Utility
{
public:
    Utility()
    {
        std::cout << "construct Utility class" << std::endl;
    }

    void pushImg(std::array<cv::Mat1b, 2>& frame, int& frameIndex);

    void removeFrame();

    void getImages(std::array<cv::Mat1b, 2>& frame, int& frameIndex);

    void saveYolo(std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver, const std::string& file);

    void save(std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver, const std::string& file);

    void save3d(std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver, const std::string& file);
};

#endif