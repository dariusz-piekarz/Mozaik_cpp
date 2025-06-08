#define _CRT_SECURE_NO_WARNINGS // switches off warnings in glob.hpp

#include <ctime>

#include <spdlog/spdlog.h>

#include "core.hpp"
#include "utils.hpp"


/**
* The main function of the Mozaik project.
* The project aims to reconstruct an image by replacing each its pixel by other images selected
* by some strategy, images are additionally filtrated to the pixel color to ensure better overlook.
* 
* @return void
**/
int main() {
    const clock_t start = std::clock();
    mozaik2();
    const clock_t end = std::clock();
    spdlog::info("Mozaik total time execution: " + elapseTime(start, end));
    return 0;
}
