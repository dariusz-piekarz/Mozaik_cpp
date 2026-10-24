#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "../core.hpp"

namespace fs = std::filesystem;

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() / ("mozaik_loader_test_" + std::to_string(suffix));
        if (!fs::create_directory(path)) {
            throw std::runtime_error("Could not create a temporary test directory");
        }
    }

    ~TemporaryDirectory() {
        std::error_code error;
        fs::remove_all(path, error);
    }

    fs::path path;
};

int main() {
    try {
        const TemporaryDirectory directory;
        const cv::Size tile_size(4, 3);
        std::vector<fs::path> paths;
        std::vector<cv::Vec3b> expected_colors;

        for (int i = 0; i < 64; ++i) {
            const fs::path path = directory.path / ("tile_" + std::to_string(i) + ".bmp");
            paths.push_back(path);

            if (i == 12) {
                std::ofstream(path) << "not an image";
                continue;
            }
            if (i == 37) {
                continue;
            }

            const cv::Vec3b color(i * 3, 255 - i * 2, i * 5 % 256);
            const cv::Mat tile(tile_size, CV_8UC3, cv::Scalar(color[0], color[1], color[2]));
            if (!cv::imwrite(path.string(), tile)) {
                throw std::runtime_error("Could not write a test image");
            }
            expected_colors.push_back(color);
        }

        for (int run = 0; run < 4; ++run) {
            const auto images = readImages(paths, tile_size);
            if (images.size() != expected_colors.size()) {
                throw std::runtime_error("Unreadable images were not skipped correctly");
            }
            for (std::size_t i = 0; i < images.size(); ++i) {
                if (images[i].size() != tile_size || images[i].type() != CV_8UC3) {
                    throw std::runtime_error("An image has incorrect dimensions or type");
                }
                const auto actual = images[i].at<cv::Vec3b>(0, 0);
                for (int channel = 0; channel < 3; ++channel) {
                    if (actual[channel] != expected_colors[i][channel]) {
                        throw std::runtime_error("Images are not in input path order at index " + std::to_string(i));
                    }
                }
            }
        }

        bool resize_failed = false;
        try {
            readImages(paths, cv::Size(0, 0));
        } catch (const cv::Exception &) {
            resize_failed = true;
        }
        if (!resize_failed) {
            throw std::runtime_error("A worker resize error was not propagated");
        }

        std::cout << "Parallel image loader test passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Parallel image loader test failed: " << error.what() << '\n';
        return 1;
    }
}
