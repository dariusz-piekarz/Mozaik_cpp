#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "utils.hpp"

namespace fs = std::filesystem;

int main() {
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path directory = fs::temp_directory_path() /
                               ("mozaik_paths_test_" + std::to_string(suffix));
    try {
        fs::create_directory(directory);
        for (const char *name : {"a.JpG", "b.JPEG", "c.PnG", "d.gif"}) {
            std::ofstream(directory / name) << "test";
        }
        fs::create_directory(directory / "e.jpg");

        const std::vector<fs::path> expected = {
            directory / "a.JpG", directory / "b.JPEG", directory / "c.PnG"
        };
        if (allImagePaths(directory.string()) != expected ||
            findPaths(directory.string(), ".jpg") != std::vector<fs::path>{directory / "a.JpG"}) {
            throw std::runtime_error("Image discovery ignored case or changed path order");
        }

        fs::remove_all(directory);
        std::cout << "PASS image discovery\n";
        return 0;
    } catch (const std::exception &error) {
        std::error_code ignored;
        fs::remove_all(directory, ignored);
        std::cerr << "FAIL image discovery: " << error.what() << '\n';
        return 1;
    }
}
