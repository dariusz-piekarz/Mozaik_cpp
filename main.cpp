#include "core.hpp"
#include <ctime>
#include <spdlog/spdlog.h>


int main()
{
    clock_t start = std::clock();
    mozaik();
    clock_t end = std::clock();
    spdlog::info("Mozaik total time execution: " + elapse_time(start, end) + " s.");
    return 0;
}
