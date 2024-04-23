#pragma once

#include "stdafx.h"
namespace UR = ur_rtde;

extern const std::string URIP;

extern std::unique_ptr<UR::RTDEControlInterface> urCtrl;
extern std::unique_ptr<UR::RTDEIOInterface> urDO;
extern std::unique_ptr<UR::RTDEReceiveInterface> urDI;

/* from joints to robot control */
extern std::queue<std::vector<std::vector<std::vector<int>>>> queueJointsPositions;
extern std::queue<bool> q_endTracking;

