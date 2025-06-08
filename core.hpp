#pragma once

#include <iostream>
#include <vector>
#include <filesystem>

#include <spdlog/spdlog.h>

#include <opencv2/opencv.hpp>


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
extern std::vector<cv::Mat> _readImages(const std::vector<fs::path> &paths, const cv::Size &subsize);


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
extern std::vector<cv::Mat> _readImages2(const std::vector<fs::path> &paths, const cv::Size &subsize);


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
extern std::vector<cv::Mat> readImages(const std::vector<fs::path> &paths, const cv::Size &subsize);


/**
 * Function to resize the main image: if proportion of the original image are not preserved,
 *	then the function ask if it should proceed, proposing alternative proportions based on user preferences.
 * 
 * @param image A cv::Mat - the main image as cv Matrix.
 * @param new_size A tuple of ints specifying the target size to which the image should be resized.
 *
 * @return A  cv::Mat object, each representing an image after resizing.
 */
extern void mainImageResize(cv::Mat &image, const std::tuple<int, int> &new_size);


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
extern std::vector<std::vector<cv::Mat> > restructure(const std::vector<cv::Mat> &subimages,
                                                      const cv::Mat &image,
                                                      const std::string_view &strategy = "pixel_mean_random");


/**
 * Function that filter an image at position (i,j) in a matrix of images to a corresponding color given by (i,j) pixel in the image.
 * The function performs the filtration to each image in the matrix decomposition.
 *
 * @param decomposition A const std::vector<std::vector<cv::Mat>>& - matrix of images.
 * @param image A cv::Mat - the input image.
 *
 * @return std::vector<std::vector<cv::Mat>> - A matrix of filtrated images
 */
extern std::vector<std::vector<cv::Mat> > project(const std::vector<std::vector<cv::Mat> > &decomposition,
                                                  const cv::Mat &image);


/**
 * Matrix of images is merged to a single images, displayed and saved (if parameters show and save_to have right values).
 * 
 * @param decomposition A const std::vector<std::vector<cv::Mat>>&
 * @param show A bool - decide if the image should be displayed
 * @param save_to A std::optional<std::string> - string path which saves
 * 
 * @return void
 **/
extern void glueImages(const std::vector<std::vector<cv::Mat> > &decomposition,
                       const bool &show = true,
                       const std::optional<std::string> &save_to = std::nullopt);


/**
 * Core function of the mozaik app.
 * It reads the configuration, loads the images, performs the mozaik, and optionally displays and/or saves the result.
 *
 * @param config_path A std::string - path to the configuration file
 *
 * @return void 
 **/
extern void mozaikCoreApp(const fs::path &config_path);


/**
 * Main function of the mozaik app.
 * It reads the configuration file and runs the mozaik app.
 The config file is read from the location of the core.hpp file.
 *
 * @return void
 **/
extern void mozaik();


/**
 * Main function of the mozaik app.
 * It allows user to input the path to the configuration file.
 *
 * @return  void
 **/
extern void mozaik2();
