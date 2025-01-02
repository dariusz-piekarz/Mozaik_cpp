#include "core.hpp"
#include <ctime>
#include <spdlog/spdlog.h>


int main()
{
    clock_t start = std::clock();
    mozaik2();
    clock_t end = std::clock();
    spdlog::info("Mozaik total time execution: " + elapse_time(start, end));
    return 0;
}
