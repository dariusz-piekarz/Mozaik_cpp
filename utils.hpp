#pragma once


#define _CRT_SECURE_NO_WARNINGS // switches off warnings in glob.hpp


#include <iostream>
#include <vector>
#include <tuple>
#include <map>
#include <format>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <cmath>
#include <ctime>
#include <ranges>
#include <random>
#include <omp.h>


#include <json/json.h>
#include <spdlog/spdlog.h>


#include <opencv2/opencv.hpp>
#include <glob/glob.hpp>


namespace fs = std::filesystem;


struct Config
{
	std::string image_path;
	std::string filler_images_dir_path;
	std::tuple<int, int> image_size;
	std::tuple<int, int> sub_images_size;
	std::string strategy;
	bool filtration;
	bool show;
	std::string output_image_path;
};


Config parse_config(const std::string& config_file_path)
{
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


inline std::vector<fs::path> find_paths(const std::string& dirname, const std::string& extension = ".jpg")
{
    std::string pattern = std::format("{}\\*{}", dirname, extension);
    return glob::glob(pattern);
}


void display_progress_bar(int& progress, int total, int bar_width = 100)
{
	float percentage = static_cast<float>(progress) / total;
	int completed = static_cast<int>(percentage * bar_width);

	std::cout << "\r[";

	for (int i = 0; i < bar_width; ++i)
	{
		if (i < completed)
			std::cout << "#";
		else
			std::cout << "-";
	}

	std::cout << "] " << static_cast<int>(percentage * 100) << "%";
	std::cout.flush();
}


std::vector<fs::path> all_image_paths(const std::string& dirname)
{
	std::vector<fs::path> all_paths;

	const std::vector<std::string> extensions({ ".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG" });
	for (const auto& ext : extensions)
	{
		auto temp_paths = find_paths(dirname, ext);

		if (!temp_paths.empty())
			all_paths.insert(all_paths.end(), temp_paths.begin(), temp_paths.end());
	}
	return all_paths;
}


std::string elapse_time(double start, double end)
{
	double elapsed_seconds = (end - start) / CLOCKS_PER_SEC;
	int minutes = static_cast<int>(elapsed_seconds) / 60;
	double seconds = elapsed_seconds - (minutes * 60);
	return std::format("{}.{}", minutes, seconds);
}


void freeMemory(std::vector<std::vector<cv::Mat>>& decomposition)
{
	for (auto& row : decomposition)
	{
		for (auto& mat : row)
			mat.release();
		row.clear();
	}
	decomposition.clear();
	decomposition.shrink_to_fit();
}


cv::Mat project_image_to_color(const cv::Mat& image, const cv::Vec3b& color_BGR)
{
	cv::Mat new_image;
	std::vector<cv::Mat> channels(3);
	std::vector<double> ratios = { color_BGR[0] / 255.0, color_BGR[1] / 255.0, color_BGR[2] / 255.0 };

	cv::split(image, channels);
	for (int i = 0; i < 3; ++i)
		channels[i] *= ratios[i];
	cv::merge(channels, new_image);
	new_image.convertTo(new_image, CV_8UC3);
	return new_image;
}


inline std::vector<cv::Vec3b> calculate_means(const std::vector<cv::Mat>& images)
{
	std::vector<cv::Vec3b> means;
	cv::Scalar mean_scalar;

	#pragma omp parallel for
	for (const auto& image : images)
	{
		mean_scalar = cv::mean(image);

		cv::Vec3b mean_color(static_cast<uchar>(mean_scalar[0]), // B
			static_cast<uchar>(mean_scalar[1]), // G
			static_cast<uchar>(mean_scalar[2])); // R

		means.push_back(mean_color);
	}
	return means;
}


inline int closest_image(const std::vector<cv::Vec3b>& means, const cv::Vec3b& color_BGR)
{
	int index = 0;
	double dist;
	double min_dist;

	#pragma omp parallel for
	for (int i = 0; i < means.size(); i++)
	{
		dist = cv::norm(means[i] - color_BGR, cv::NORM_L2);
		if (i == 0)
			min_dist = dist;
		else if (dist < min_dist)
		{
			min_dist = dist;
			index = i;
		}
	}
	return index;
}


inline int select_closest_pict_random(const std::vector<cv::Vec3b>& picures_means_BGR, const cv::Vec3b& color_BGR, int rank = 30)
{
	std::vector<std::pair<double, int>> dists;

	#pragma omp parallel
	{
		std::vector<std::pair<double, int>> local_dists;

		#pragma omp for nowait
		for (int i = 0; i < picures_means_BGR.size(); i++)
		{
			double distance = cv::norm(picures_means_BGR[i] - color_BGR, cv::NORM_L2);
			local_dists.emplace_back(distance, i);
		}
		#pragma omp critical
		dists.insert(dists.end(), local_dists.begin(), local_dists.end());
	}

	std::sort(dists.begin(), dists.end());

	std::vector<int> closest_indices;
	for (int i = 0; i < std::min(rank, static_cast<int>(dists.size())); ++i)
		closest_indices.push_back(dists[i].second);
	
	if (!closest_indices.empty())
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<int> dis(0, static_cast<int>(closest_indices.size() - 1));
		return closest_indices[dis(gen)];
	}
	return -1;
}


cv::Mat merge_images(const std::vector<std::vector<cv::Mat>>& image_grid)
{
	std::vector<cv::Mat> rows;

	for (const auto& row : image_grid)
	{
		cv::Mat row_image;
		cv::hconcat(row, row_image);
		rows.push_back(row_image);
	}

	cv::Mat final_image;
	cv::vconcat(rows, final_image);

	return final_image;
}


void show_image(const cv::Mat& image, const std::string& title)
{
	cv::imshow(title, image);
	cv::waitKey(0);
	cv::destroyAllWindows();
}