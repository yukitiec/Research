#pragma once

#ifndef UTILITY_H
#define UTILITY_H

#include "stdafx.h"

// queue definition
//extern std::queue<std::array<cv::Mat1b, 2>> queueFrame; // queue for frame
//extern std::queue<int> queueFrameIndex;  // queue for frame index


class Utility
{
public:
    std::queue<std::array<cv::Mat1b, 2>> queueFrame; // queue for frame
    std::queue<int> queueFrameIndex;  // queue for frame index

    Utility()
    {
        std::cout << "construct Utility" << std::endl;
    }

    bool getImagesFromQueueYolo(std::array<cv::Mat1b, 2>& imgs, int& frameIndex);

    void checkStorage(std::vector<std::vector<cv::Rect2d>>& posSaverYolo, std::vector<int>& detectedFrame, std::string fileName);

    void checkClassStorage(std::vector<std::vector<int>>& classSaverYolo, std::vector<int>& detectedFrame, std::string fileName);


    void checkStorage_v2(std::vector<std::vector<cv::Rect2d>>& posSaverYolo, std::vector<int>& detectedFrame, std::string fileName);

    void checkClassStorage_V2(std::vector<std::vector<int>>& classSaverYolo, std::vector<int>& detectedFrame, std::string fileName);

    void checkStorageTM(std::vector<std::vector<cv::Rect2d>>& posSaverYolo, std::vector<int>& detectedFrame, std::string fileName);

    void checkClassStorageTM(std::vector<std::vector<int>>& classSaverYolo, std::vector<int>& detectedFrame, std::string fileName);

    void checkSeqData(std::vector<std::vector<std::vector<double>>>& dataLeft, std::string fileName);

    void checkKfData(std::vector<std::vector<std::vector<double>>>& dataLeft, std::string fileName);

    void checkSeqData_v2(std::vector<std::vector<std::vector<int>>>& dataLeft, std::vector<std::vector<int>>& classesLeft, std::string fileName_bbox, std::string fileName_class);

    void save3d(std::vector<std::vector<std::vector<double>>>& posSaver, const std::string file);

    void saveTarget(std::vector<std::vector<std::vector<double>>>& posSaver, const std::string file);

    bool getImagesFromQueueTM(std::array<cv::Mat1b, 2>& imgs, int& frameIndex);

    /* read imgs */
    void pushFrame(std::array<cv::Mat1b, 2>& src, const int frameIndex);
};

#endif
