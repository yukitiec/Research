#pragma once

#ifndef UTILITY_H
#define UTILITY_H

#include "stdafx.h"
#include "global_parameters.h"

/* frame queue */
extern std::queue<std::array<cv::Mat1b, 2>> queueFrame;
extern std::queue<int> queueFrameIndex;

class Utility
{
public:
    Utility()
    {
        std::cout << "construct Utility class" << std::endl;
    }

    void pushImg(std::array<cv::Mat1b, 2>& frame, int& frameIndex);

    void getImages(std::array<cv::Mat1b, 2>& frame, int& frameIndex);

    void saveYolo(std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver, const std::string& file);

    void save(std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver, const std::string& file);

    void save3d(std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver, const std::string& file);
};

#endif