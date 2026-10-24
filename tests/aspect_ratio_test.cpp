#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include <opencv2/opencv.hpp>

#include "core.hpp"

namespace {

class RedirectStream {
public:
    RedirectStream(std::ios &stream, std::streambuf *replacement)
        : stream_(stream), original_(stream.rdbuf()), state_(stream.rdstate()) {
        stream_.rdbuf(replacement);
        stream_.clear();
    }

    ~RedirectStream() {
        stream_.rdbuf(original_);
        stream_.clear(state_);
    }

private:
    std::ios &stream_;
    std::streambuf *original_;
    std::ios::iostate state_;
};

void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void checkResize(int source_width, int source_height, int requested_width,
                 int requested_height, const std::string &answers,
                 int expected_width, int expected_height, bool expect_prompt = true) {
    cv::Mat image(source_height, source_width, CV_8UC3, cv::Scalar(10, 20, 30));
    std::istringstream input(answers);
    std::ostringstream output;
    {
        RedirectStream redirect_input(std::cin, input.rdbuf());
        RedirectStream redirect_output(std::cout, output.rdbuf());
        mainImageResize(image, {requested_width, requested_height});
    }
    require(image.cols == expected_width && image.rows == expected_height,
            "Unexpected dimensions for answers: " + answers);
    require(output.str().empty() == !expect_prompt,
            "Unexpected aspect ratio prompt for requested dimensions");
}

void checkInvalidInput() {
    cv::Mat empty;
    try {
        mainImageResize(empty, {2, 2});
        throw std::runtime_error("Empty image was accepted");
    } catch (const std::invalid_argument &) {
    }

    cv::Mat image(3, 4, CV_8UC3, cv::Scalar(10, 20, 30));
    try {
        mainImageResize(image, {0, 2});
        throw std::runtime_error("Zero width was accepted");
    } catch (const std::invalid_argument &) {
    }
}

} // namespace

int main() {
    try {
        checkResize(4, 3, 8, 6, "N\n", 8, 6, false);
        checkResize(4, 3, 2, 5, "Y\nW\n", 2, 2); // 2 * 3 / 4 = 1.5, rounded up.
        checkResize(4, 3, 2, 2, "Y\nH\n", 3, 2);
        checkResize(4, 3, 2, 5, "N\n", 2, 5);
        checkResize(1000, 1, 1, 3, "Y\nW\n", 1, 1);
        checkInvalidInput();
        std::cout << "Aspect ratio test passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Aspect ratio test failed: " << error.what() << '\n';
        return 1;
    }
}
