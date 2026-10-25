#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <thread>
#include <optional>
#include <stdexcept>
#include <utility>

#include <spdlog/spdlog.h>

#include <opencv2/opencv.hpp>

#include "utils.hpp"
#include "core.hpp"
#include "exceptions.hpp"


namespace fs = std::filesystem;


std::vector<cv::Mat> _readImages(const std::vector<fs::path> &paths, const cv::Size &subsize) {
    std::vector<cv::Mat> images;
    int iter = 0;
    const auto TOTAL = static_cast<int>(paths.size());
    for (const auto &path: paths) {
        cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
        if (!image.empty()) {
            cv::resize(image, image, subsize, 0, 0, cv::INTER_AREA);
            images.push_back(std::move(image));
        }

        iter++;

        if (iter % 10 == 0 || iter == TOTAL) {
            displayProgressBar(iter, TOTAL, 100);
        }
    }
    return images;
}


std::vector<cv::Mat> _readImages2(const std::vector<fs::path> &paths, const cv::Size &subsize) {
    if (paths.empty()) {
        return {};
    }

    const std::size_t thread_count = std::min<std::size_t>(12, paths.size());
    const int total = static_cast<int>(paths.size());
    std::atomic<int> iter(0);
    std::mutex mtx;
    std::vector<std::vector<cv::Mat>> batches(thread_count);
    std::vector<std::exception_ptr> failures(thread_count);
    std::vector<std::jthread> threads;
    threads.reserve(thread_count);

    for (std::size_t i = 0; i < thread_count; ++i) {
        const auto begin = paths.begin() + static_cast<std::ptrdiff_t>(paths.size() * i / thread_count);
        const auto end = paths.begin() + static_cast<std::ptrdiff_t>(paths.size() * (i + 1) / thread_count);
        std::vector<fs::path> subrange(begin, end);
        threads.emplace_back([&, i, subrange = std::move(subrange)] {
            try {
                threadReadAndRescale(subrange, batches[i], subsize, mtx, iter, total);
            } catch (...) {
                failures[i] = std::current_exception();
            }
        });
    }

    for (auto &thread: threads) {
        thread.join();
    }
    for (const auto &failure: failures) {
        if (failure) {
            std::rethrow_exception(failure);
        }
    }

    std::vector<cv::Mat> images;
    images.reserve(paths.size());
    for (auto &batch: batches) {
        images.insert(images.end(), std::make_move_iterator(batch.begin()), std::make_move_iterator(batch.end()));
    }
    return images;
}


std::vector<cv::Mat> readImages(const std::vector<fs::path> &paths, const cv::Size &subsize) {
    if (static_cast<int>(paths.size()) > 50) {
        return _readImages2(paths, subsize);
    }
    return _readImages(paths, subsize);
}


void mainImageResize(cv::Mat &image, const std::tuple<int, int> &new_size,
                     const std::string_view aspect_ratio) {
    const int width = std::get<0>(new_size);
    const int height = std::get<1>(new_size);
    if (image.empty() || width <= 0 || height <= 0) {
        throw std::invalid_argument("Image and requested dimensions must be nonempty and positive.");
    }

    const auto proportionalSize = [](int fixed_size, int original_other, int original_fixed) -> int {
        const auto numerator = static_cast<std::int64_t>(fixed_size) * original_other;
        const auto rounded = (numerator + original_fixed / 2) / original_fixed;
        if (rounded > std::numeric_limits<int>::max()) {
            throw std::overflow_error("Proportional image dimension exceeds the supported range.");
        }
        return static_cast<int>(rounded > 0 ? rounded : 1);
    };

    if (aspect_ratio != "interactive" && aspect_ratio != "preserve_width" &&
        aspect_ratio != "preserve_height" && aspect_ratio != "stretch") {
        throw std::invalid_argument("Unsupported aspect ratio mode: " + std::string(aspect_ratio));
    }

    const auto resizePreservingWidth = [&]() {
        const int suggested_height = proportionalSize(width, image.rows, image.cols);
        cv::resize(image, image, cv::Size(width, suggested_height), 0, 0, cv::INTER_AREA);
    };
    const auto resizePreservingHeight = [&]() {
        const int suggested_width = proportionalSize(height, image.cols, image.rows);
        cv::resize(image, image, cv::Size(suggested_width, height), 0, 0, cv::INTER_AREA);
    };

    if (static_cast<std::int64_t>(image.cols) * height == static_cast<std::int64_t>(image.rows) * width ||
        aspect_ratio == "stretch") {
        cv::resize(image, image, cv::Size(width, height), 0, 0, cv::INTER_AREA);
    } else if (aspect_ratio == "preserve_width") {
        resizePreservingWidth();
    } else if (aspect_ratio == "preserve_height") {
        resizePreservingHeight();
    } else {
        std::cout << "Original proportion is not preserved with the provided new size. " <<
                "If you want to rescale with preserving proportion press Y / y, otherwise press N / n.\n";

        const auto read_decision = []() {
            std::string value;
            if (!(std::cin >> value)) {
                throw std::runtime_error("Input ended while selecting image resize options.");
            }
            return value;
        };

        std::string decision = read_decision();

        while (decision != "Y" && decision != "y" && decision != "N" && decision != "n") {
            std::cout <<
                    "Invalid input. Please press Y/y to rescale with preserving proportion or N/n to continue without preserving proportion.\n";
            decision = read_decision();
        }

        if (decision == "Y" || decision == "y") {
            std::cout << "Proposed are two solutions: keep width and adjust height or keep height and adjust width." <<
                    " To select the first strategy press W/w, to select the second press H/h\n";
            decision = read_decision();

            while (decision != "W" && decision != "w" && decision != "H" && decision != "h") {
                std::cout << "Invalid input. Please press W/w to keep width or H/h to keep height.\n";
                decision = read_decision();
            }

            if (decision == "W" || decision == "w") {
                resizePreservingWidth();
            } else if (decision == "H" || decision == "h") {
                resizePreservingHeight();
            }
        } else {
            cv::resize(image, image, cv::Size(width, height), 0, 0, cv::INTER_AREA);
        }
    }
}


std::vector<std::vector<cv::Mat> > restructure(const std::vector<cv::Mat> &subimages,
                                               const cv::Mat &image,
                                               const std::string_view &strategy) {
    const int width = image.cols;
    const int height = image.rows;
    const auto subimage_len = static_cast<int>(subimages.size());

    std::vector ret_image(height, std::vector<cv::Mat>(width));

    if (strategy == "duplication") {
#pragma omp parallel for collapse(2)
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                ret_image[i][j] = subimages[(i * width + j) % subimage_len];
            }
        }
    } else if (strategy == "pixel_mean") {
        const std::vector<cv::Vec3b> MEANS = calculateMeans(subimages);

#pragma omp parallel for collapse(2)
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                ret_image[i][j] = subimages[closestImage(MEANS, image.at<cv::Vec3b>(i, j))];
            }
        }
    } else if (strategy == "pixel_mean_random") {
        const std::vector<cv::Vec3b> MEANS = calculateMeans(subimages);
        const auto RANK = static_cast<int>(std::ceil(24. / 1117. * static_cast<double>(subimage_len) + 2.));

#pragma omp parallel for collapse(2)
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                ret_image[i][j] = subimages[selectClosestPictRandom(MEANS, image.at<cv::Vec3b>(i, j), RANK)];
            }
        }
    } else {
        throw std::invalid_argument("Unsupported strategy!");
    }
    return ret_image;
}


std::vector<std::vector<cv::Mat> > project(const std::vector<std::vector<cv::Mat> > &decomposition,
                                           const cv::Mat &image) {
    std::vector output(image.rows, std::vector<cv::Mat>(image.cols));
#pragma omp parallel for collapse(2)
    for (int i = 0; i < image.rows; i++)
        for (int j = 0; j < image.cols; j++)
            output[i][j] = projectImageToColor(decomposition[i][j], image.at<cv::Vec3b>(i, j));
    return output;
}


cv::Mat renderMosaic(const std::vector<cv::Mat> &subimages, const cv::Mat &image,
                     const std::string_view &strategy, bool filtration) {
    if (image.empty() || image.type() != CV_8UC3 || subimages.empty()) {
        throw std::invalid_argument("A color source image and at least one tile are required.");
    }
    const cv::Size tile_size = subimages.front().size();
    if (tile_size.width <= 0 || tile_size.height <= 0) {
        throw std::invalid_argument("Tiles must have positive dimensions.");
    }
    for (const auto &tile : subimages) {
        if (tile.size() != tile_size || tile.type() != CV_8UC3) {
            throw std::invalid_argument("All tiles must be color images of the same size.");
        }
    }
    if (strategy != "duplication" && strategy != "pixel_mean" && strategy != "pixel_mean_random") {
        throw std::invalid_argument("Unsupported strategy!");
    }

    const auto output_width = static_cast<std::int64_t>(image.cols) * tile_size.width;
    const auto output_height = static_cast<std::int64_t>(image.rows) * tile_size.height;
    if (output_width > std::numeric_limits<int>::max() ||
        output_height > std::numeric_limits<int>::max()) {
        throw std::overflow_error("Mosaic dimensions exceed OpenCV's supported range.");
    }
    cv::Mat output(static_cast<int>(output_height), static_cast<int>(output_width), CV_8UC3);

    const auto means = strategy == "duplication" ? std::vector<cv::Vec3b>{} : calculateMeans(subimages);
    const auto rank = static_cast<int>(std::ceil(24. / 1117. * static_cast<double>(subimages.size()) + 2.));
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            const auto color = image.at<cv::Vec3b>(row, col);
            std::size_t index;
            if (strategy == "duplication") {
                index = (static_cast<std::size_t>(row) * image.cols + col) % subimages.size();
            } else if (strategy == "pixel_mean") {
                index = static_cast<std::size_t>(closestImage(means, color));
            } else {
                index = static_cast<std::size_t>(selectClosestPictRandom(means, color, rank));
            }

            const cv::Rect target(col * tile_size.width, row * tile_size.height,
                                  tile_size.width, tile_size.height);
            cv::Mat destination = output(target);
            if (filtration) {
                const double scale[] = {
                    color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
                };
                for (int tile_row = 0; tile_row < tile_size.height; ++tile_row) {
                    const auto *source_pixels = subimages[index].ptr<cv::Vec3b>(tile_row);
                    auto *destination_pixels = destination.ptr<cv::Vec3b>(tile_row);
                    for (int tile_col = 0; tile_col < tile_size.width; ++tile_col) {
                        for (int channel = 0; channel < 3; ++channel) {
                            destination_pixels[tile_col][channel] = cv::saturate_cast<uchar>(
                                source_pixels[tile_col][channel] * scale[channel]);
                        }
                    }
                }
            } else {
                subimages[index].copyTo(destination);
            }
        }
    }
    return output;
}


void glueImages(const std::vector<std::vector<cv::Mat> > &decomposition,
                const bool &show,
                const std::optional<std::string> &save_to) {
    const cv::Mat merged_image = mergeImages(decomposition);

    if (show) {
        showImage(merged_image, "Merged image");
    }

    if (save_to.has_value()) {
        try {
            if (!cv::imwrite(save_to.value(), merged_image)) {
                throw std::runtime_error("Failed to save output image: " + save_to.value());
            }
        } catch (const cv::Exception &error) {
            throw std::runtime_error("Failed to save output image '" + save_to.value() +
                                     "': " + error.what());
        }
    }
}


void mozaikCoreApp(const fs::path &config_path) {
    Config config;

    config = parseConfig(config_path);

    if (!fs::exists(config.image_path)) {
        throw fs::filesystem_error("Error: The main image file does not exist: ", config.image_path,
                                   std::error_code(std::make_error_code(std::errc::no_such_file_or_directory)));
    }

    if (!fs::is_directory(config.filler_images_dir_path)) {
        throw fs::filesystem_error("Error: The directory where sub images should be stored does not exist: ",
                                   config.filler_images_dir_path,
                                   std::error_code(std::make_error_code(std::errc::no_such_file_or_directory)));
    }

    spdlog::info("Collecting paths.");
    clock_t start0 = std::clock();

    std::vector<fs::path> images_paths = allImagePaths(config.filler_images_dir_path);
    clock_t end0 = std::clock();

    if (images_paths.empty()) {
        throw ExtensionError(
            "The path: " + config.filler_images_dir_path +
            ", contains no images with extensions [jpg, JPG, JPEG, jpeg, png, PNG].");
    }

    spdlog::info("Collecting paths completed in time: " + elapseTime(start0, end0));
    spdlog::info("Reading main image started.");

    cv::Mat image;
    try {
        image = cv::imread(config.image_path, cv::IMREAD_COLOR);
    } catch (const cv::Exception &error) {
        throw std::runtime_error("Cannot read main image '" + config.image_path +
                                 "': " + error.what());
    }
    if (image.empty()) {
        throw std::runtime_error("Cannot read main image: " + config.image_path);
    }

    spdlog::info("Reading main image completed.");
    spdlog::info("Reading sub images started.");
    clock_t start1 = std::clock();

    std::vector<cv::Mat> images = readImages(images_paths,
                                             cv::Size(std::get<0>(config.sub_images_size),
                                                      std::get<1>(config.sub_images_size)));
    clock_t end1 = std::clock();

    if (images.empty()) {
        throw std::out_of_range("Set of images is empty.");
    }

    spdlog::info("Reading sub images completed in time: " + elapseTime(start1, end1));
    spdlog::info("Rescaling main image started.");

    mainImageResize(image, config.image_size, config.aspect_ratio);

    spdlog::info("Rescaling main image completed.");
    spdlog::info("Rendering mosaic started.");
    clock_t start2 = std::clock();

    const cv::Mat result = renderMosaic(images, image, config.strategy, config.filtration);
    clock_t end2 = std::clock();

    spdlog::info("Rendering mosaic completed in time: " + elapseTime(start2, end2));
    if (config.show) {
        showImage(result, "Merged image");
    }
    if (!config.output_image_path.empty()) {
        try {
            if (!cv::imwrite(config.output_image_path, result)) {
                throw std::runtime_error("Failed to save output image: " + config.output_image_path);
            }
        } catch (const cv::Exception &error) {
            throw std::runtime_error("Failed to save output image '" + config.output_image_path +
                                     "': " + error.what());
        }
    }
    spdlog::info("Program successfully run.");
}


void mozaik() {
    const fs::path local_config = fs::current_path() / "config.json";
    if (fs::is_regular_file(local_config)) {
        mozaikCoreApp(local_config);
        return;
    }

    const fs::path example_config = fs::current_path() / "examples" / "config.json";
    if (fs::is_regular_file(example_config)) {
        mozaikCoreApp(example_config);
        return;
    }

    throw fs::filesystem_error("No config.json found in the current directory or examples:",
                               example_config,
                               std::make_error_code(std::errc::no_such_file_or_directory));
}

void mozaik2() {
    std::string config_path;
    while (true) {
        std::cout << "Provide a path to the config file:\n";
        if (!std::getline(std::cin, config_path)) {
            throw std::runtime_error("Input ended before a config path was provided.");
        }
        if (fs::is_regular_file(config_path)) {
            mozaikCoreApp(config_path);
            return;
        }
        std::cout << "Provide a valid path to the config file.\n";
    }
}
