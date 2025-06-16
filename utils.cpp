#include <atomic>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>
#include <ranges>

#include <json/json.h>
#include <spdlog/spdlog.h>

#include <opencv2/opencv.hpp>
#include <glob.hpp>

#include "utils.hpp"


namespace fs = std::filesystem;


Config parseConfig(const fs::path &config_file_path) {
    std::ifstream file(config_file_path);
    Json::Reader reader;
    Json::Value root;
    reader.parse(file, root);

    Config config;

    config.image_path = root["image_path"].asString();
    config.filler_images_dir_path = root["filler_images_dir_path"].asString();
    config.image_size = std::make_tuple(root["image_size"][0].asInt(), root["image_size"][1].asInt());
    config.sub_images_size = std::make_tuple(root["sub_images_size"][0].asInt(), root["sub_images_size"][1].asInt());
    config.strategy = root["strategy"].asString();
    config.filtration = root["filtration"].asBool();
    config.show = root["show"].asBool();
    config.output_image_path = root["output_image_path"].asString();

    return config;
}


std::vector<fs::path> findPaths(const std::string &dirname, const std::string &extension = ".jpg") {
    const std::string pattern = std::format("{}\\*{}", dirname, extension);
    return glob::glob(pattern);
}


std::vector<fs::path> allImagePaths(const std::string &dirname) {
    std::vector<fs::path> all_paths;

    for (const auto &ext: {".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"}) {
        auto temp_paths = findPaths(dirname, ext);

        if (!temp_paths.empty()) {
            all_paths.insert(all_paths.end(), temp_paths.begin(), temp_paths.end());
        }
    }
    return all_paths;
}


void displayProgressBar(const int &progress, const int &total, const int &bar_width = 100) {
    const double percentage = static_cast<double>(progress) / static_cast<double>(total);
    const auto completed = static_cast<int>(percentage * static_cast<double>(bar_width));

    std::cout << "\r[";

    for (int i = 0; i < bar_width; ++i) {
        if (i < completed) {
            std::cout << "#";
        } else {
            std::cout << "-";
        }
    }

    std::cout << "] " << static_cast<int>(percentage * 100) << "%";
    std::cout.flush();
    if (percentage == 1.0) {
        std::cout << "" << std::endl;
    }
}


std::string elapseTime(const double start, const double end) {
    const double elapsed_seconds = (end - start) / CLOCKS_PER_SEC;
    int minutes = static_cast<int>(elapsed_seconds) / 60;
    const double seconds = elapsed_seconds - (minutes * 60);
    return std::format("{} min {} s.", minutes, static_cast<int>(seconds));
}


void freeMemory(std::vector<std::vector<cv::Mat> > &decomposition) {
    for (auto &row: decomposition) {
        for (auto &mat: row) {
            mat.release();
        }
        row.clear();
    }
    decomposition.clear();
    decomposition.shrink_to_fit();
}


cv::Mat projectImageToColor(const cv::Mat &image, const cv::Vec3b &color_BGR) {
    cv::Mat new_image;
    std::vector<cv::Mat> channels(3);
    const std::vector ratios = {color_BGR[0] / 255.0, color_BGR[1] / 255.0, color_BGR[2] / 255.0};

    cv::split(image, channels);
    for (int i = 0; i < 3; ++i) {
        channels[i] *= ratios[i];
    }
    cv::merge(channels, new_image);
    new_image.convertTo(new_image, CV_8UC3);
    return new_image;
}


inline std::vector<cv::Vec3b> calculateMeans(const std::vector<cv::Mat> &images) {
    std::vector<cv::Vec3b> means;
    means.reserve(images.size());
    cv::Scalar mean_scalar;

    for (const auto &image: images) {
        mean_scalar = cv::mean(image);

        cv::Vec3b mean_color(static_cast<uchar>(mean_scalar[0]), // B
                             static_cast<uchar>(mean_scalar[1]), // G
                             static_cast<uchar>(mean_scalar[2])); // R

        means.emplace_back(mean_color);
    }
    return means;
}


inline double l2Norm(cv::Vec3b v, cv::Vec3b w) {
    return std::sqrt(
        std::pow(static_cast<double>(v[0] - w[0]), 2) +
        std::pow(static_cast<double>(v[1] - w[1]), 2) +
        std::pow(static_cast<double>(v[2] - w[2]), 2)
    );
}


inline int closestImage(const std::vector<cv::Vec3b> &means, const cv::Vec3b &color_BGR) {
    int final_index = 0;
    double final_min_dist = std::numeric_limits<double>::max();

    for (int i = 0; i < means.size(); i++) {
        if (const double dist = l2Norm(means[i], color_BGR); dist < final_min_dist) {
            final_min_dist = dist;
            final_index = i;
        }
    }

    return final_index;
}

inline int selectClosestPictRandom(const std::vector<cv::Vec3b> &pictures_means_BGR,
                                   const cv::Vec3b &color_BGR,
                                   const int rank = 30) {
    std::vector<std::pair<double, int> > dists;

#pragma omp parallel
    {
        std::vector<std::pair<double, int> > local_dists;

#pragma omp for nowait
        for (int i = 0; i < pictures_means_BGR.size(); i++) {
            double distance = l2Norm(pictures_means_BGR[i], color_BGR);
            local_dists.emplace_back(distance, i);
        }
#pragma omp critical
        dists.insert(dists.end(), local_dists.begin(), local_dists.end());
    }

    std::ranges::sort(dists);

    std::vector<int> closest_indices;
    for (int i = 0; i < std::min(rank, static_cast<int>(dists.size())); ++i) {
        closest_indices.push_back(dists[i].second);
    }

    if (!closest_indices.empty()) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution dis(0, static_cast<int>(closest_indices.size() - 1));
        return closest_indices[dis(gen)];
    }
    return -1;
};


cv::Mat mergeImages(const std::vector<std::vector<cv::Mat> > &image_grid) {
    std::vector<cv::Mat> rows;
    rows.reserve(image_grid.size());

    for (const auto &row: image_grid) {
        cv::Mat row_image;
        cv::hconcat(row, row_image);
        rows.emplace_back(row_image);
    }

    cv::Mat final_image;
    cv::vconcat(rows, final_image);

    return final_image;
};


void showImage(const cv::Mat &image, const std::string &title) {
    cv::imshow(title, image);
    cv::waitKey(0);
    cv::destroyAllWindows();
};


void threadReadAndRescale(const std::vector<fs::path> &subpaths,
                          std::vector<cv::Mat> &im_subset,
                          const cv::Size &subsize,
                          std::mutex &mtx,
                          std::atomic<int> &iter,
                          const int &total) {
    int progress = 0;
    std::vector<cv::Mat> buffer;
    buffer.reserve(subpaths.size());

    for (const auto &path: subpaths) {
        cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            continue;
        }

        cv::resize(image, image, subsize, 0, 0, cv::INTER_AREA);
        buffer.push_back(std::move(image));
        iter += 1;
        progress = iter;
        if (progress % 10 == 0 || progress == total) {
            displayProgressBar(progress, total, 100);
        }
    } {
        std::scoped_lock guard(mtx);
        im_subset.insert(im_subset.end(), std::make_move_iterator(buffer.begin()),
                         std::make_move_iterator(buffer.end()));
    }
};
