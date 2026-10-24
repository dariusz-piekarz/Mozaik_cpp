#include <iostream>
#include <stdexcept>
#include <vector>

#include "utils.hpp"

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void testNearestTwo() {
    const cv::Vec3b target(0, 0, 0);
    const std::vector<cv::Vec3b> means{
        {10, 0, 0}, {1, 0, 0}, {5, 0, 0}, {2, 0, 0}, {8, 0, 0}
    };
    for (int draw = 0; draw < 32; ++draw) {
        const int index = selectClosestPictRandom(means, target, 2);
        require(index == 1 || index == 3, "Selected a candidate outside the two nearest");
    }
}

void testRankLargerThanInput() {
    const std::vector<cv::Vec3b> means{{1, 0, 0}, {2, 0, 0}, {3, 0, 0}};
    for (int draw = 0; draw < 32; ++draw) {
        const int index = selectClosestPictRandom(means, cv::Vec3b(0, 0, 0), 20);
        require(index >= 0 && index < static_cast<int>(means.size()),
                "Rank larger than input returned an invalid index");
    }
}

void testEqualDistances() {
    const std::vector<cv::Vec3b> means{
        {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {2, 0, 0}
    };
    for (int draw = 0; draw < 32; ++draw) {
        const int index = selectClosestPictRandom(means, cv::Vec3b(0, 0, 0), 3);
        require(index >= 0 && index < 3, "Selected a more distant candidate over a tie");
    }
}

void testEmptyInput() {
    require(selectClosestPictRandom({}, cv::Vec3b(0, 0, 0), 3) == -1,
            "Empty input should return -1");
}

void testInvalidRank() {
    const std::vector<cv::Vec3b> means{{0, 0, 0}};
    require(selectClosestPictRandom(means, cv::Vec3b(0, 0, 0), 0) == -1,
            "Zero rank should return -1");
    require(selectClosestPictRandom(means, cv::Vec3b(0, 0, 0), -3) == -1,
            "Negative rank should return -1");
}

void testBufferRefresh() {
    const std::vector<cv::Vec3b> means{{0, 0, 0}, {255, 0, 0}};
    for (int draw = 0; draw < 32; ++draw) {
        require(selectClosestPictRandom(means, cv::Vec3b(0, 0, 0), 1) == 0,
                "Wrong nearest candidate for black");
        require(selectClosestPictRandom(means, cv::Vec3b(255, 0, 0), 1) == 1,
                "Candidate buffer was not refreshed between calls");
    }
}

void testGreenAndRedChannels() {
    const std::vector<cv::Vec3b> means{
        {0, 0, 0}, {0, 255, 0}, {0, 0, 255}
    };
    require(selectClosestPictRandom(means, cv::Vec3b(0, 255, 0), 1) == 1,
            "Green channel was ignored when ranking candidates");
    require(selectClosestPictRandom(means, cv::Vec3b(0, 0, 255), 1) == 2,
            "Red channel was ignored when ranking candidates");
}

} // namespace

int main() {
    try {
        testNearestTwo();
        testRankLargerThanInput();
        testEqualDistances();
        testEmptyInput();
        testInvalidRank();
        testBufferRefresh();
        testGreenAndRedChannels();
        std::cout << "PASS random selection\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL random selection: " << error.what() << '\n';
        return 1;
    }
}
