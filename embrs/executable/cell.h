#pragma once  // Prevent multiple inclusions
#pragma pack(push, 1)  // Pack the structures to avoid padding
#include <utility> 
#include "util.h"

enum CellState {
    BURNT = 0,
    FUEL = 1,
    WILDFIRE = 2,
    PRESCRIBED_FIRE = 3
};

struct Cell {
    int id;
    CellState state;
    int fuelType;
    float fuelContent;
    float moisture;
    Position position;
    std::pair<int, int> indices;
    bool changed;

};



#pragma pack(pop)