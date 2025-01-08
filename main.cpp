#define _CRT_SECURE_NO_WARNINGS // switches off warnings in glob.hpp


#include <ctime>


#include <spdlog/spdlog.h>


#include "core.hpp"



int main()
{
    clock_t start = std::clock();
    mozaik2();
    clock_t end = std::clock();
    spdlog::info("Mozaik total time execution: " + elapseTime(start, end));
    return 0;
}
