#pragma once

#define _CRT_SECURE_NO_WARNINGS // switches off warnings in glob.hpp

#include <fstream>
#include <atomic>
#include <mutex>
#include <vector>

#include <opencv2/opencv.hpp>
#include <glob.hpp>

namespace fs = std::filesystem;


struct Config {
    std::string image_path;
    std::string filler_images_dir_path;
    std::tuple<int, int> image_size;
    std::tuple<int, int> sub_images_size;
    std::string strategy;
    bool filtration = false;
    bool show = false;
    std::string output_image_path;
};

extern Config parseConfig(const fs::path &);

extern std::vector<fs::path> findPaths(const std::string &, const std::string &);

extern std::vector<fs::path> allImagePaths(const std::string &);

extern void displayProgressBar(const int &, const int &, const int &);

extern std::string elapseTime(double, double);

extern void freeMemory(std::vector<std::vector<cv::Mat> > &);

extern cv::Mat projectImageToColor(const cv::Mat &, const cv::Vec3b &);

extern inline std::vector<cv::Vec3b> calculateMeans(const std::vector<cv::Mat> &);

extern inline double l2Norm(cv::Vec3b, cv::Vec3b);

extern inline int closestImage(const std::vector<cv::Vec3b> &, const cv::Vec3b &);

extern inline int selectClosestPictRandom(const std::vector<cv::Vec3b> &, const cv::Vec3b &, int);

extern cv::Mat mergeImages(const std::vector<std::vector<cv::Mat> > &);

extern void showImage(const cv::Mat &image, const std::string &);

extern void threadReadAndRescale(const std::vector<fs::path>&, std::vector<cv::Mat> &, const cv::Size &, std::mutex &,
                                 std::atomic<int> &, const int &);
