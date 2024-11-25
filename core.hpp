#pragma once


#include "utils.hpp"


#include <iostream>
#include <ctime>
#include <cmath>
#include <ranges>
#include <thread>
#include <mutex>
#include <omp.h>


#include <spdlog/spdlog.h>


#include <opencv2/opencv.hpp>


namespace fs = std::filesystem;


std::vector<cv::Mat> read_images2(const std::vector<fs::path>& paths, const cv::Size& subsize)
{
	std::vector<cv::Mat> images;
	std::mutex mtx;
	std::vector<std::thread> threads;
	const int thread_count = 8;
	const int total = static_cast<int>(paths.size());

	std::vector<int> partition;
	partition.reserve(thread_count + 1);

	for (int i = 0; i <= thread_count; i++)
		partition.push_back(static_cast<int>(std::floor(total * i / thread_count)));

	for (int i = 0; i < thread_count; i++)
	{
		std::vector<fs::path> subrange(paths.begin() + partition[i], paths.begin() + partition[i + 1]);
		threads.emplace_back(thread_read_and_rescale, subrange, std::ref(images), std::ref(subsize), std::ref(mtx));
	}
	for (auto& thread : threads) 
		if (thread.joinable()) 
			thread.join();

	return images;
}


std::vector<cv::Mat> read_images(const std::vector<fs::path>& paths, const cv::Size& subsize)
{
	std::vector<cv::Mat> images;
	int iter = 0; 
	const int total = static_cast<int>(paths.size()); 

	for (const auto& path : paths) 
	{
		cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR); 
		if (image.empty()) continue;
		
		cv::resize(image, image, subsize, 0, 0, cv::INTER_AREA); 
		images.push_back(image); 
		if (iter % 10 == 0) 
			display_progress_bar(iter, total);
		
		iter++;
		
	}
	std::cout << std::endl;
	return images;
}


void main_image_resize(cv::Mat& image, std::tuple<int, int> new_size)
{
	int width = std::get<0>(new_size);
	int height = std::get<1>(new_size);
	double ratio = static_cast<double>(width) / static_cast<double>(height);
	double new_ratio = static_cast<double>(image.cols) / static_cast<double>(image.rows);

	if (ratio == new_ratio)
		cv::resize(image, image, cv::Size(width, height), 0, 0, cv::INTER_AREA);
	else
	{
		
		int suggested_new_size_h = std::ceil(width / ratio);
		int suggested_new_size_w = std::ceil(height * ratio);
		std::cout << "Original proportion is not preserved with the provided new size. " <<
			"If you want to rescale with preserving proportion press Y / y, otherwise press N / n.\n";
		
		std::string decision;
		std::cin >> decision;

		while (decision != "Y" && decision != "y" && decision != "N" && decision != "n")
		{
			std::cout << "Invalid input. Please press Y/y to rescale with preserving proportion or N/n to continue without preserving proportion.\n";
			std::cin >> decision;
		}

		if (decision == "Y" || decision == "y")
		{

			std::cout << "Proposed are two solutions: keep width and adjust height or keep height and adjust width." << 
				" To select the first strategy press W/w, to select the second press H/h\n";
			std::cin >> decision;

			while (decision != "W" && decision != "w" && decision != "H" && decision != "h") 
			{
				std::cout << "Invalid input. Please press W/w to keep width or H/h to keep height.\n";
				std::cin >> decision;
			}

			if (decision == "W" || decision == "w")
				cv::resize(image, image, cv::Size(width, suggested_new_size_h), 0, 0, cv::INTER_AREA);
			else if (decision == "H" || decision == "h")
				cv::resize(image, image, cv::Size(suggested_new_size_w, height), 0, 0, cv::INTER_AREA);
		}
		else
			cv::resize(image, image, cv::Size(width, height), 0, 0, cv::INTER_AREA);
	}
}


std::vector<std::vector<cv::Mat>> restructure(std::vector<cv::Mat>& subimages, const cv::Mat& image, const std::string& strategy = "pixel_mean_random")
{
	int width = image.cols;
	int height = image.rows;
	int subim_len = subimages.size();

	std::vector<std::vector<cv::Mat>> ret_image(height, std::vector<cv::Mat>(width));

	if (strategy == "duplication")
	{
		#pragma omp parallel for collapse(2)
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				ret_image[i][j] = subimages[(i * width + j) % subim_len];
	}
	else if (strategy == "pixel_mean")
	{
		const std::vector<cv::Vec3b> means = calculate_means(subimages);

		#pragma omp parallel for collapse(2)
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				ret_image[i][j] = subimages[closest_image(means, image.at<cv::Vec3b>(i, j))];
	}
	else if (strategy == "pixel_mean_random")
	{
		const std::vector<cv::Vec3b> means = calculate_means(subimages);
		const int rank = std::ceil(24. / 1117. * static_cast<double>(subim_len) + 5.);
		std::cout << rank << std::endl;
		#pragma omp parallel for collapse(2)
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				ret_image[i][j] = subimages[select_closest_pict_random(means, image.at<cv::Vec3b>(i, j), rank )];
	}
	else
		throw std::invalid_argument("Unsupported strategy!");
	return ret_image;
}


std::vector<std::vector<cv::Mat>> project(const std::vector<std::vector<cv::Mat>>& decomposition, const cv::Mat& image)
{
	std::vector<std::vector<cv::Mat>> output(image.rows, std::vector<cv::Mat>(image.cols));
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < image.rows; i++)
		for (int j = 0; j < image.cols; j++)
			output[i][j] = project_image_to_color(decomposition[i][j], image.at<cv::Vec3b>(i, j));
	return output; 
}


void glue_images(const std::vector<std::vector<cv::Mat>>& decomposition, bool show = true, std::optional<std::string> save_to = std::nullopt)
{
	cv::Mat merged_image = merge_images(decomposition);

	if (show)
		show_image(merged_image, "Merged image");

	if (save_to.has_value())
		cv::imwrite(save_to.value(), merged_image);
}


void mozaik_core_app(std::string config_path)
{
	Config config;
	
	config = parse_config(config_path);

	if (!fs::exists(config.image_path))
		throw std::runtime_error("Error: The main image file does not exist: " + config.image_path);

	if (!fs::exists(config.filler_images_dir_path))
		throw std::runtime_error("Error: The directory where sub images should be stored does not exist: " + config.filler_images_dir_path);

	spdlog::info("Collecting paths.");
	clock_t start0 = std::clock();

	std::vector<fs::path> images_paths = all_image_paths(config.filler_images_dir_path);
	clock_t end0 = std::clock();

	if (images_paths.empty())
		throw std::runtime_error("The path: " + config.filler_images_dir_path + ", contains no images with extensions [jpg, JPG, JPEG, jpeg, png, PNG].");

	spdlog::info("Collecting paths completed in time: " + elapse_time(start0, end0) + " s.");
	spdlog::info("Reading main image started.");

	cv::Mat image = cv::imread(config.image_path, cv::IMREAD_COLOR);

	spdlog::info("Reading main image completed.");
	spdlog::info("Reading sub images started.");
	clock_t start1 = std::clock();

	std::vector<cv::Mat> images = read_images2(images_paths, cv::Size(std::get<0>(config.sub_images_size), std::get<1>(config.sub_images_size)));
	clock_t end1 = std::clock();

	if (images.empty())
		throw std::runtime_error("Set of images is empty.");

	spdlog::info("Reading sub images completed in time: " + elapse_time(start1, end1) + " s.");
	spdlog::info("Rescaling main image started.");

	main_image_resize(image, config.image_size);

	spdlog::info("Rescaling main image completed.");
	spdlog::info("Reordering of sub images started.");
	clock_t start2 = std::clock();

	std::vector<std::vector<cv::Mat>> matrix_images = restructure(images, image, config.strategy);
	clock_t end2 = std::clock();

	spdlog::info("Reordering of sub images completed in time: " + elapse_time(start2, end2) + " s.");

	std::vector<std::vector<cv::Mat>> matrix_images_f;

	if (config.filtration)
	{
		spdlog::info("Filtration of sub images to main picture pixels started.");
		clock_t start3 = std::clock();

		matrix_images_f = project(matrix_images, image);
		clock_t end3 = std::clock();

		freeMemory(matrix_images);

		spdlog::info("Filtration of sub images to main picture pixels completed in time: " + elapse_time(start3, end3) + " s.");
	}

	spdlog::info("Combining pictures started.");

	std::optional<std::string> save_to = config.output_image_path != "" ? std::optional<std::string>{config.output_image_path} : std::nullopt;
	clock_t start4 = std::clock();
	if (config.filtration)
		glue_images(matrix_images_f, config.show, save_to);
	else
		glue_images(matrix_images, config.show, save_to);
	clock_t end4 = std::clock();

	spdlog::info("Combining pictures completed in time: " + elapse_time(start4, end4) + " s.");
	spdlog::info("Program successfully run.");
}


void mozaik(void)
{
	fs::path cfg_path = fs::path(__FILE__).parent_path();
	std::string config_path = cfg_path.string() + "\\config.json";


	if (!fs::exists(config_path))
		throw std::runtime_error("Error: The config file does not exist: " + config_path);

	mozaik_core_app(config_path);
}


void mozaik2(void)
{
	std::string config_path;
	std::cout << "Provide a path to the config file:\n";
	std::cin >> config_path;

	while (!fs::exists(config_path))
	{
		std::cout << "Provide a valid path to the config file:\n";
		std::cin >> config_path;
	}

	mozaik_core_app(config_path);
}
