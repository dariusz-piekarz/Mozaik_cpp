#pragma once

#include <atomic>
#include <cmath>
#include <ctime>
#include <iostream>
#include <mutex>
#include <optional>
#include <thread>
#include <omp.h>


#include <spdlog/spdlog.h>


#include <opencv2/opencv.hpp>


#include "utils.hpp"


namespace fs = std::filesystem;

/**
 * Function to read images from a given list of paths and resize them to a specified size.
 *
 * @param paths A vector of filesystem paths pointing to the images to be read.
 * @param subsize The target size to which each image should be resized.
 *
 * @return A vector of cv::Mat objects, each representing an image after resizing.
 *
 * This function reads images from the provided paths, resizes them to the specified subsize,
 * and returns a vector of the resized images. If an image cannot be read or is empty,
 * it is skipped and not included in the returned vector.
 *
 * The function also displays a progress bar to indicate the progress of image reading.
 */
std::vector<cv::Mat> _readImages(const std::vector<fs::path>& paths, const cv::Size& subsize)
{
	std::vector<cv::Mat> images;
	int iter = 0;
	auto TOTAL = static_cast<int>(paths.size());
	for (const auto& path : paths)
	{
		cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
		if (image.empty()) continue;

		cv::resize(image, image, subsize, 0, 0, cv::INTER_AREA);
		images.push_back(image);

		iter++;

		if (iter % 10 == 0 || iter == TOTAL)
		{
			displayProgressBar(iter, TOTAL);
		}
	}
	return images;
}


/**
 * Function to read images from a given list of paths and resize them to a specified size using multithreading approach.
 *
 * @param paths A vector of filesystem paths pointing to the images to be read.
 * @param subsize The target size to which each image should be resized.
 *
 * @return A vector of cv::Mat objects, each representing an image after resizing.
 *
 * This function reads images using multithreading approach from the provided paths, resizes them to the specified subsize,
 * and returns a vector of the resized images. If an image cannot be read or is empty,
 * it is skipped and not included in the returned vector.
 *
 * The function also displays a progress bar to indicate the progress of image reading.
 */
std::vector<cv::Mat> _readImages2(const std::vector<fs::path>& paths, const cv::Size& subsize)
{
	std::vector<cv::Mat> images;
	std::mutex mtx;
	std::vector<std::jthread> threads;
	
	const int THREAD_COUNT = 12;
	auto TOTAL = static_cast<int>(paths.size());
	std::atomic<int> iter(0);

	std::vector<int> partition;
	partition.reserve(THREAD_COUNT + 1);

	for (int i = 0; i <= THREAD_COUNT; i++)
	{
		partition.push_back(static_cast<int>(std::floor(TOTAL * i / THREAD_COUNT)));
	}

	for (int i = 0; i < THREAD_COUNT; i++)
	{
		std::vector<fs::path> subrange(paths.begin() + partition[i], paths.begin() + partition[i + 1]);
		threads.emplace_back(threadReadAndRescale, subrange, std::ref(images), std::ref(subsize), std::ref(mtx), std::ref(iter), std::ref(TOTAL));
	}
	return images;
}


/**
 * This function reads images using multithreading approach from the provided paths, resizes them to the specified subsize,
 * and returns a vector of the resized images. If an image cannot be read or is empty,
 * it is skipped and not included in the returned vector.
 *
 * The function also displays a progress bar to indicate the progress of image reading.
 * @param paths A vector of filesystem paths pointing to the images to be read.
 * @param subsize The target size to which each image should be resized.
 *
 * @return A vector of cv::Mat objects, each representing an image after resizing.
 
 */
std::vector<cv::Mat> readImages(const std::vector<fs::path>& paths, const cv::Size& subsize)
{
	std::vector<cv::Mat> images;
	auto TOTAL = static_cast<int>(paths.size());

	if (TOTAL > 50)
	{
		return _readImages2(paths, subsize);
	}
	else
	{
		return _readImages(paths, subsize);
	}
}


/**
 * Function to resize the main image: if proportion of the original image are not preserved,
 *	then the function ask if proceed, proposing alternative proportions based on user preferences.
 * 
 * @param image A cv::Mat - the main image as cv Matrix.
 * @param new_size A tuple of ints specifying the target size to which the image should be resized.
 *
 * @return A  cv::Mat object, each representing an image after resizing.
 */
void mainImageResize(cv::Mat& image, std::tuple<int, int> new_size)
{
	int width = std::get<0>(new_size);
	int height = std::get<1>(new_size);
	double ratio = static_cast<double>(width) / static_cast<double>(height);
	double new_ratio = static_cast<double>(image.cols) / static_cast<double>(image.rows);

	if (ratio == new_ratio)
	{
		cv::resize(image, image, cv::Size(width, height), 0, 0, cv::INTER_AREA);
	}
	else
	{
		
		auto suggested_new_size_h = static_cast<int>(std::ceil(width / ratio));
		auto suggested_new_size_w = static_cast<int>(std::ceil(height * ratio));
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
			{
				cv::resize(image, image, cv::Size(width, suggested_new_size_h), 0, 0, cv::INTER_AREA);
			}
			else if (decision == "H" || decision == "h")
			{
				cv::resize(image, image, cv::Size(suggested_new_size_w, height), 0, 0, cv::INTER_AREA);
			}
		}
		else
		{
			cv::resize(image, image, cv::Size(width, height), 0, 0, cv::INTER_AREA);
		}
	}
}


/**
  * This function take as arguments a list of cv::Mat which are images, to each pixel in the main image cv::Mat there is found a cv::Mat image in the list
 * of images based on the 3 strategies:
 * - 'pixel_mean' - there is found a mean pixel to each cv::Mat in the images vector and the image with the closest mean pixel to a pixel in the main cv::Mat is 
 *	 found based on l2 norm,
 * - 'pixel_mean_random' - similarly as in the case above, there is found a population of images with a smallest l2 distance between mean pixels and the image pixel,
 *   selection is based on the uniform distribution from the selected population of images with the smallest l2 distances,
 * - 'duplication' - if number of images is smaller than number of pixels in the image, to each row of pixels there are copied cv::Mat images, so that the entire picture
 *   is filled.
 *
 * @param subimages A std::vector<cv::Mat>& - images which replace pixels of the main image.
 * @param image A cv::Mat - the input image.
 * @param strategy A std::string& - 'pixel_mean', 'pixel_mean_random' or 'duplication' - way of fitting subimages to pixels of the main image.
 * 
 * @return std::vector<std::vector<cv::Mat>> A matrix of cv::Matrices build from the images based on pixels of the main picture and selected strategy
 * 

 */
std::vector<std::vector<cv::Mat>> restructure(std::vector<cv::Mat>& subimages, const cv::Mat& image, const std::string& strategy = "pixel_mean_random")
{
	int width = image.cols;
	int height = image.rows;
	auto subim_len = static_cast<int>(subimages.size());

	std::vector<std::vector<cv::Mat>> ret_image(height, std::vector<cv::Mat>(width));

	if (strategy == "duplication")
	{
		#pragma omp parallel for collapse(2)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				ret_image[i][j] = subimages[(i * width + j) % subim_len];
			}
		}
	}
	else if (strategy == "pixel_mean")
	{
		const std::vector<cv::Vec3b> MEANS = calculateMeans(subimages);

		#pragma omp parallel for collapse(2)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				ret_image[i][j] = subimages[closestImage(MEANS, image.at<cv::Vec3b>(i, j))];
			}
		}
	}
	else if (strategy == "pixel_mean_random")
	{
		const std::vector<cv::Vec3b> MEANS = calculateMeans(subimages);
		auto RANK = static_cast<int>(std::ceil(24. / 1117. * static_cast<double>(subim_len) + 2.));

		#pragma omp parallel for collapse(2)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				ret_image[i][j] = subimages[selectClosestPictRandom(MEANS, image.at<cv::Vec3b>(i, j), RANK)];
			}
		}
	}
	else
	{
		throw std::invalid_argument("Unsupported strategy!");
	}
	return ret_image;
}


/**
 * Function that filter an image at position (i,j) in a matrix of images to a corresponding color given by (i,j) pixel in the image.
 * The function performs the filtration to each image in the matrix decomposition.
 *
 * @param decomposition A const std::vector<std::vector<cv::Mat>>& - matrix of images.
 * @param image A cv::Mat - the input image.
 *
 * @return std::vector<std::vector<cv::Mat>> - A matrix of filtrated images
 */
std::vector<std::vector<cv::Mat>> project(const std::vector<std::vector<cv::Mat>>& decomposition, const cv::Mat& image)
{
	std::vector<std::vector<cv::Mat>> output(image.rows, std::vector<cv::Mat>(image.cols));
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < image.rows; i++)
		for (int j = 0; j < image.cols; j++)
			output[i][j] = projectImageToColor(decomposition[i][j], image.at<cv::Vec3b>(i, j));
	return output; 
}



/**
 * Matrix of images is merged to a single images, displayed and saved (if parameters show and save_to have right values).
 * 
 * @param decomposition A const std::vector<std::vector<cv::Mat>>&
 * @param show A bool - decide if the image should be displayed
 * @param save_to A std::optional<std::string> - string path which saves
 * 
 * @return void
 **/
void glueImages(const std::vector<std::vector<cv::Mat>>& decomposition, bool show = true, std::optional<std::string> save_to = std::nullopt)
{
	cv::Mat merged_image = mergeImages(decomposition);

	if (show)
	{
		showImage(merged_image, "Merged image");
	}

	if (save_to.has_value())
	{
		cv::imwrite(save_to.value(), merged_image);
	}
}

/**
 * Core function of the mozaik app.
 * It reads the configuration, loads the images, performs the mozaik, and optionally displays and/or saves the result.
 *
 * @param config_path A std::string - path to the configuration file
 *
 * @return void 
 **/
void mozaikCoreApp(std::string config_path)
{
	Config config;
	
	config = parseConfig(config_path);

	if (!fs::exists(config.image_path))
	{
		throw fs::filesystem_error("Error: The main image file does not exist: ", config.image_path, std::error_code(std::make_error_code(std::errc::no_such_file_or_directory)));
	}

	if (!fs::exists(config.filler_images_dir_path))
	{
		throw fs::filesystem_error("Error: The directory where sub images should be stored does not exist: ", config.filler_images_dir_path, std::error_code(std::make_error_code(std::errc::no_such_file_or_directory)));
	}

	spdlog::info("Collecting paths.");
	clock_t start0 = std::clock();

	std::vector<fs::path> images_paths = allImagePaths(config.filler_images_dir_path);
	clock_t end0 = std::clock();

	if (images_paths.empty())
	{
		throw std::runtime_error("The path: " + config.filler_images_dir_path + ", contains no images with extensions [jpg, JPG, JPEG, jpeg, png, PNG].");
	}

	spdlog::info("Collecting paths completed in time: " + elapseTime(start0, end0));
	spdlog::info("Reading main image started.");

	cv::Mat image = cv::imread(config.image_path, cv::IMREAD_COLOR);

	spdlog::info("Reading main image completed.");
	spdlog::info("Reading sub images started.");
	clock_t start1 = std::clock();

	std::vector<cv::Mat> images = readImages(images_paths, cv::Size(std::get<0>(config.sub_images_size), std::get<1>(config.sub_images_size)));
	clock_t end1 = std::clock();

	if (images.empty())
	{
		throw std::out_of_range("Set of images is empty.");
	}

	spdlog::info("Reading sub images completed in time: " + elapseTime(start1, end1));
	spdlog::info("Rescaling main image started.");

	mainImageResize(image, config.image_size);

	spdlog::info("Rescaling main image completed.");
	spdlog::info("Reordering of sub images started.");
	clock_t start2 = std::clock();

	std::vector<std::vector<cv::Mat>> matrix_images = restructure(images, image, config.strategy);
	clock_t end2 = std::clock();

	spdlog::info("Reordering of sub images completed in time: " + elapseTime(start2, end2));

	std::vector<std::vector<cv::Mat>> matrix_images_f;

	if (config.filtration)
	{
		spdlog::info("Filtration of sub images to main picture pixels started.");
		clock_t start3 = std::clock();

		matrix_images_f = project(matrix_images, image);
		clock_t end3 = std::clock();

		freeMemory(matrix_images);

		spdlog::info("Filtration of sub images to main picture pixels completed in time: " + elapseTime(start3, end3));
	}

	spdlog::info("Combining pictures started.");

	std::optional<std::string> save_to = config.output_image_path != "" ? std::optional<std::string>{config.output_image_path} : std::nullopt;
	clock_t start4 = std::clock();
	if (config.filtration)
	{
		glueImages(matrix_images_f, config.show, save_to);
	}
	else
	{
		glueImages(matrix_images, config.show, save_to);
	}
	clock_t end4 = std::clock();

	spdlog::info("Combining pictures completed in time: " + elapseTime(start4, end4));
	spdlog::info("Program successfully run.");
}

/**
 * Main function of the mozaik app.
 * It reads the configuration file and runs the mozaik app.
 The config file is read from he location of the core.hpp file.
 *
 * @return void
 **/
void mozaik(void)
{
	fs::path cfg_path = fs::path(__FILE__).parent_path();
	std::string config_path = cfg_path.string() + "\\config.json";


	if (!fs::exists(config_path))
	{
		throw fs::filesystem_error("Error: The config file does not exist:", config_path, std::error_code(std::make_error_code(std::errc::no_such_file_or_directory)));
	}

	mozaikCoreApp(config_path);
}


/**
 * Main function of the mozaik app.
 * It allows user to input the path to the configuration file.
 *
 * @return  void
 **/
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

	mozaikCoreApp(config_path);
}
