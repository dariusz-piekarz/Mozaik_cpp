#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "core.hpp"
#include "utils.hpp"

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireColor(const cv::Mat &image, int row, int column,
                  const cv::Vec3b &expected) {
    const auto actual = image.at<cv::Vec3b>(row, column);
    for (int channel = 0; channel < 3; ++channel) {
        require(actual[channel] == expected[channel],
                "Rendered pixel has the wrong color");
    }
}

void requireSolidBlock(const cv::Mat &image, int top, int left, int size,
                       const cv::Vec3b &expected) {
    for (int row = top; row < top + size; ++row) {
        for (int column = left; column < left + size; ++column) {
            requireColor(image, row, column, expected);
        }
    }
}

void testDuplicationOrderAndDimensions() {
    const cv::Vec3b colors[]{
        {10, 20, 30}, {40, 50, 60}, {70, 80, 90}, {100, 110, 120}
    };
    std::vector<cv::Mat> tiles;
    for (const auto &color : colors) {
        tiles.emplace_back(2, 2, CV_8UC3, cv::Scalar(color[0], color[1], color[2]));
    }
    const cv::Mat source(2, 2, CV_8UC3, cv::Scalar(0, 0, 0));

    const cv::Mat result = renderMosaic(tiles, source, "duplication", false);
    require(result.rows == 4 && result.cols == 4,
            "A 2x2 source with 2x2 tiles should produce a 4x4 image");
    requireSolidBlock(result, 0, 0, 2, colors[0]);
    requireSolidBlock(result, 0, 2, 2, colors[1]);
    requireSolidBlock(result, 2, 0, 2, colors[2]);
    requireSolidBlock(result, 2, 2, 2, colors[3]);
    tiles.front().setTo(cv::Scalar(255, 255, 255));
    requireSolidBlock(result, 0, 0, 2, colors[0]);
}

void testPixelMeanSelection() {
    const std::vector<cv::Mat> tiles{
        cv::Mat(1, 1, CV_8UC3, cv::Scalar(0, 0, 0)),
        cv::Mat(1, 1, CV_8UC3, cv::Scalar(255, 255, 255))
    };
    cv::Mat source(2, 2, CV_8UC3);
    source.at<cv::Vec3b>(0, 0) = cv::Vec3b(0, 0, 0);
    source.at<cv::Vec3b>(0, 1) = cv::Vec3b(255, 255, 255);
    source.at<cv::Vec3b>(1, 0) = cv::Vec3b(255, 255, 255);
    source.at<cv::Vec3b>(1, 1) = cv::Vec3b(0, 0, 0);

    const cv::Mat result = renderMosaic(tiles, source, "pixel_mean", false);
    require(result.rows == 2 && result.cols == 2,
            "A 2x2 source with 1x1 tiles should produce a 2x2 image");
    for (int row = 0; row < 2; ++row) {
        for (int column = 0; column < 2; ++column) {
            requireColor(result, row, column, source.at<cv::Vec3b>(row, column));
        }
    }
}

void testFiltration() {
    const std::vector<cv::Mat> tiles{
        cv::Mat(2, 2, CV_8UC3, cv::Scalar(255, 255, 255))
    };
    cv::Mat source(2, 2, CV_8UC3);
    source.at<cv::Vec3b>(0, 0) = cv::Vec3b(0, 64, 255);
    source.at<cv::Vec3b>(0, 1) = cv::Vec3b(255, 0, 64);
    source.at<cv::Vec3b>(1, 0) = cv::Vec3b(64, 255, 0);
    source.at<cv::Vec3b>(1, 1) = cv::Vec3b(128, 128, 128);

    const cv::Mat result = renderMosaic(tiles, source, "duplication", true);
    require(result.rows == 4 && result.cols == 4,
            "Filtration changed the output dimensions");
    for (int row = 0; row < 2; ++row) {
        for (int column = 0; column < 2; ++column) {
            requireSolidBlock(result, row * 2, column * 2, 2,
                              source.at<cv::Vec3b>(row, column));
        }
    }
}

void testFilteredPatternMatchesPreviousColorRule() {
    cv::Mat tile(5, 7, CV_8UC3);
    for (int row = 0; row < tile.rows; ++row) {
        for (int col = 0; col < tile.cols; ++col) {
            tile.at<cv::Vec3b>(row, col) = cv::Vec3b(
                (row * 37 + col * 13) % 256,
                (row * 11 + col * 29) % 256,
                (row * 17 + col * 31) % 256);
        }
    }
    cv::Mat source(1, 2, CV_8UC3);
    source.at<cv::Vec3b>(0, 0) = cv::Vec3b(71, 128, 219);
    source.at<cv::Vec3b>(0, 1) = cv::Vec3b(255, 31, 0);

    const cv::Mat result = renderMosaic({tile}, source, "duplication", true);
    require(result.rows == tile.rows && result.cols == 2 * tile.cols,
            "Filtered mosaic has incorrect dimensions");
    for (int source_col = 0; source_col < 2; ++source_col) {
        const cv::Mat expected = projectImageToColor(tile, source.at<cv::Vec3b>(0, source_col));
        const cv::Mat actual = result(cv::Rect(source_col * tile.cols, 0, tile.cols, tile.rows));
        require(cv::countNonZero(expected.reshape(1) != actual.reshape(1)) == 0,
                "Direct filtering changed tile colors");
    }
}

void testUnfilteredPatternPreserved() {
    cv::Mat tile(2, 3, CV_8UC3);
    for (int row = 0; row < tile.rows; ++row) {
        for (int col = 0; col < tile.cols; ++col) {
            tile.at<cv::Vec3b>(row, col) = cv::Vec3b(
                row * 80 + col * 10, row * 20 + col * 40, row * 30 + col * 50);
        }
    }
    const cv::Mat source(1, 2, CV_8UC3, cv::Scalar(0, 0, 0));
    const cv::Mat result = renderMosaic({tile}, source, "duplication", false);
    require(result.size() == cv::Size(6, 2), "Unfiltered pattern has incorrect dimensions");
    for (int block = 0; block < 2; ++block) {
        for (int row = 0; row < tile.rows; ++row) {
            for (int col = 0; col < tile.cols; ++col) {
                requireColor(result, row, block * tile.cols + col,
                             tile.at<cv::Vec3b>(row, col));
            }
        }
    }
}

void testRandomStrategyWithOneTile() {
    const std::vector<cv::Mat> tiles{
        cv::Mat(1, 2, CV_8UC3, cv::Scalar(7, 8, 9))
    };
    const cv::Mat source(1, 2, CV_8UC3, cv::Scalar(0, 0, 0));
    const cv::Mat result = renderMosaic(tiles, source, "pixel_mean_random", false);
    require(result.rows == 1 && result.cols == 4,
            "Random strategy changed output dimensions");
    for (int col = 0; col < result.cols; ++col) {
        requireColor(result, 0, col, cv::Vec3b(7, 8, 9));
    }
}

void testParallelMatchesSequential() {
    std::vector<cv::Mat> tiles;
    for (int index = 0; index < 7; ++index) {
        cv::Mat tile(5, 3, CV_8UC3);
        for (int row = 0; row < tile.rows; ++row) {
            for (int col = 0; col < tile.cols; ++col) {
                tile.at<cv::Vec3b>(row, col) = cv::Vec3b(
                    (index * 31 + row * 7 + col) % 256,
                    (index * 17 + row + col * 11) % 256,
                    (index * 13 + row * 19 + col * 3) % 256);
            }
        }
        tiles.push_back(std::move(tile));
    }
    cv::Mat source(9, 11, CV_8UC3);
    for (int row = 0; row < source.rows; ++row) {
        for (int col = 0; col < source.cols; ++col) {
            source.at<cv::Vec3b>(row, col) = cv::Vec3b(
                (row * 29 + col * 5) % 256,
                (row * 3 + col * 23) % 256,
                (row * 17 + col * 7) % 256);
        }
    }

    for (const bool filtration : {false, true}) {
        const cv::Mat sequential = renderMosaic(
            tiles, source, "pixel_mean", filtration, RenderExecution::sequential);
        const cv::Mat parallel = renderMosaic(
            tiles, source, "pixel_mean", filtration, RenderExecution::parallel);
        require(cv::countNonZero(sequential.reshape(1) != parallel.reshape(1)) == 0,
                "Parallel rendering differs from sequential rendering");
    }
}

} // namespace

int main() {
    try {
        testDuplicationOrderAndDimensions();
        testPixelMeanSelection();
        testFiltration();
        testFilteredPatternMatchesPreviousColorRule();
        testUnfilteredPatternPreserved();
        testRandomStrategyWithOneTile();
        testParallelMatchesSequential();
        std::cout << "PASS mosaic rendering\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL mosaic rendering: " << error.what() << '\n';
        return 1;
    }
}
