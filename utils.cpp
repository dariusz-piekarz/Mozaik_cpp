#include <algorithm>
#include <atomic>
#include <array>
#include <cctype>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <json/json.h>
#include <spdlog/spdlog.h>

#include <opencv2/opencv.hpp>

#include "utils.hpp"


namespace fs = std::filesystem;


Config parseConfig(const fs::path &config_file_path) {
    std::ifstream file(config_file_path);
    if (!file) {
        throw std::runtime_error("Cannot open config file: " + config_file_path.string());
    }

    Json::CharReaderBuilder reader;
    reader["allowComments"] = false;
    reader["allowTrailingCommas"] = false;
    reader["rejectDupKeys"] = true;
    reader["failIfExtra"] = true;
    Json::Value root;
    std::string parse_errors;
    if (!Json::parseFromStream(reader, file, &root, &parse_errors)) {
        throw std::invalid_argument("Invalid JSON in config file '" + config_file_path.string() +
                                    "': " + parse_errors);
    }
    if (!root.isObject()) {
        throw std::invalid_argument("Config must contain a JSON object: " + config_file_path.string());
    }

    constexpr std::array<std::string_view, 8> allowed_fields = {
        "image_path", "filler_images_dir_path", "image_size", "sub_images_size",
        "strategy", "filtration", "show", "output_image_path"
    };
    for (const auto &name : root.getMemberNames()) {
        if (std::find(allowed_fields.begin(), allowed_fields.end(), std::string_view(name)) ==
            allowed_fields.end()) {
            throw std::invalid_argument("Unknown config field: " + name);
        }
    }

    const auto require_string = [&root](const char *name, bool allow_empty = false) {
        const Json::Value &value = root[name];
        if (!value.isString() || (!allow_empty && value.asString().empty())) {
            throw std::invalid_argument(std::string("Config field '") + name +
                                        (allow_empty ? "' must be a string" :
                                                       "' must be a nonempty string"));
        }
        return value.asString();
    };
    const auto require_size = [&root](const char *name) {
        const Json::Value &value = root[name];
        if (!value.isArray() || value.size() != 2 || !value[0].isInt() || !value[1].isInt() ||
            value[0].asInt() <= 0 || value[1].asInt() <= 0) {
            throw std::invalid_argument(std::string("Config field '") + name +
                                        "' must contain two positive integer dimensions");
        }
        return std::make_tuple(value[0].asInt(), value[1].asInt());
    };
    const auto require_bool = [&root](const char *name) {
        const Json::Value &value = root[name];
        if (!value.isBool()) {
            throw std::invalid_argument(std::string("Config field '") + name + "' must be a boolean");
        }
        return value.asBool();
    };

    const fs::path config_dir = fs::absolute(config_file_path).parent_path();
    const auto resolve_path = [&config_dir](const std::string &value) {
        fs::path path(value);
        if (path.is_relative()) {
            path = config_dir / path;
        }
        return path.lexically_normal().string();
    };

    Config config;

    config.image_path = resolve_path(require_string("image_path"));
    config.filler_images_dir_path = resolve_path(require_string("filler_images_dir_path"));
    config.image_size = require_size("image_size");
    config.sub_images_size = require_size("sub_images_size");
    config.strategy = require_string("strategy");
    if (config.strategy != "duplication" && config.strategy != "pixel_mean" &&
        config.strategy != "pixel_mean_random") {
        throw std::invalid_argument("Unsupported config strategy: " + config.strategy);
    }
    config.filtration = require_bool("filtration");
    config.show = require_bool("show");
    const std::string output_path = require_string("output_image_path", true);
    config.output_image_path = output_path.empty() ? "" : resolve_path(output_path);

    return config;
}


namespace {
std::string lowercaseExtension(const fs::path &path) {
    std::string extension = path.extension().string();
    std::ranges::transform(extension, extension.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return extension;
}
}


std::vector<fs::path> findPaths(const std::string &dirname, const std::string &extension) {
    std::string wanted = extension;
    std::ranges::transform(wanted, wanted.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });

    std::vector<fs::path> paths;
    for (const auto &entry : fs::directory_iterator(dirname)) {
        if (entry.is_regular_file() && lowercaseExtension(entry.path()) == wanted) {
            paths.push_back(entry.path());
        }
    }
    std::ranges::sort(paths);
    return paths;
}


std::vector<fs::path> allImagePaths(const std::string &dirname) {
    std::vector<fs::path> paths;
    for (const auto &entry : fs::directory_iterator(dirname)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string extension = lowercaseExtension(entry.path());
        if (extension == ".jpg" || extension == ".jpeg" || extension == ".png") {
            paths.push_back(entry.path());
        }
    }
    std::ranges::sort(paths);
    return paths;
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


std::vector<cv::Vec3b> calculateMeans(const std::vector<cv::Mat> &images) {
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


double l2Norm(cv::Vec3b v, cv::Vec3b w) {
    return std::sqrt(
        std::pow(static_cast<double>(v[0] - w[0]), 2) +
        std::pow(static_cast<double>(v[1] - w[1]), 2) +
        std::pow(static_cast<double>(v[2] - w[2]), 2)
    );
}


int closestImage(const std::vector<cv::Vec3b> &means, const cv::Vec3b &color_BGR) {
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

int selectClosestPictRandom(const std::vector<cv::Vec3b> &pictures_means_BGR,
                                   const cv::Vec3b &color_BGR,
                                   const int rank = 30) {
    if (rank <= 0 || pictures_means_BGR.empty()) {
        return -1;
    }

    using Candidate = std::pair<int, int>; // Squared distance and original index.
    const auto candidate_count = std::min(static_cast<std::size_t>(rank), pictures_means_BGR.size());
    // Keep the worst of the selected candidates at the front of the max heap.
    thread_local std::vector<Candidate> nearest;
    nearest.clear();
    nearest.reserve(candidate_count);

    for (std::size_t i = 0; i < pictures_means_BGR.size(); ++i) {
        int squared_distance = 0;
        for (int channel = 0; channel < 3; ++channel) {
            const int difference = static_cast<int>(pictures_means_BGR[i][channel]) - color_BGR[channel];
            squared_distance += difference * difference;
        }

        const Candidate candidate(squared_distance, static_cast<int>(i));
        if (nearest.size() < candidate_count) {
            nearest.push_back(candidate);
            std::push_heap(nearest.begin(), nearest.end());
        } else if (candidate < nearest.front()) {
            std::pop_heap(nearest.begin(), nearest.end());
            nearest.back() = candidate;
            std::push_heap(nearest.begin(), nearest.end());
        }
    }

    thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<std::size_t> distribution(0, nearest.size() - 1);
    return nearest[distribution(generator)].second;
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
    std::vector<cv::Mat> buffer;
    buffer.reserve(subpaths.size());

    for (const auto &path: subpaths) {
        cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
        if (!image.empty()) {
            cv::resize(image, image, subsize, 0, 0, cv::INTER_AREA);
            buffer.push_back(std::move(image));
        }

        std::scoped_lock guard(mtx);
        const int progress = ++iter;
        if (progress % 10 == 0 || progress == total) {
            displayProgressBar(progress, total, 100);
        }
    } {
        std::scoped_lock guard(mtx);
        im_subset.insert(im_subset.end(), std::make_move_iterator(buffer.begin()),
                         std::make_move_iterator(buffer.end()));
    }
};
