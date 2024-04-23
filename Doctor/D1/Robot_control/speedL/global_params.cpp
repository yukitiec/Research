#include "global_params.h"

extern const std::string URIP="169.254.52.209";

std::unique_ptr<UR::RTDEControlInterface> urCtrl = std::make_unique<UR::RTDEControlInterface>(URIP);
std::unique_ptr<UR::RTDEIOInterface> urDO = std::make_unique<UR::RTDEIOInterface>(URIP);
std::unique_ptr<UR::RTDEReceiveInterface> urDI = std::make_unique<UR::RTDEReceiveInterface>(URIP);


std::queue<std::vector<std::vector<std::vector<int>>>> queueJointsPositions;
std::queue<bool> q_endTracking;