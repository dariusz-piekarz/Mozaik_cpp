#include <ctime>
#include <exception>
#include <iostream>

#include <spdlog/spdlog.h>

#include "core.hpp"
#include "utils.hpp"


/**
* The main function of the Mozaik project.
* The project aims to reconstruct an image by replacing each its pixel by other images selected
* by some strategy, images are additionally filtrated to the pixel color to ensure better overlook.
*
* @return 0 on success, nonzero on error
**/
int main(int argc, char *argv[]) {
    try {
        if (argc > 2) {
            std::cerr << "Usage: " << argv[0] << " [config.json]\n";
            return 2;
        }

        const clock_t start = std::clock();
        if (argc == 2) {
            mozaikCoreApp(fs::path(argv[1]));
        } else {
            mozaik2();
        }
        const clock_t end = std::clock();
        spdlog::info("Mozaik total time execution: " + elapseTime(start, end));
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Mozaik error: " << error.what() << '\n';
        return 1;
    }
}
