#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

#include "core.hpp"
#include "utils.hpp"

namespace {

namespace fs = std::filesystem;

void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

class TemporaryConfig {
public:
    explicit TemporaryConfig(const std::string &contents) {
        static std::atomic<unsigned> sequence{0};
        const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = fs::temp_directory_path() /
                ("mozaik_validation_" + std::to_string(timestamp) + "_" +
                 std::to_string(sequence++) + ".json");

        std::ofstream file(path_, std::ios::binary);
        require(static_cast<bool>(file), "Could not create a temporary config file");
        file << contents;
        require(static_cast<bool>(file), "Could not write a temporary config file");
    }

    TemporaryConfig(const TemporaryConfig &) = delete;
    TemporaryConfig &operator=(const TemporaryConfig &) = delete;

    ~TemporaryConfig() {
        std::error_code error;
        fs::remove(path_, error);
    }

    const fs::path &path() const { return path_; }

private:
    fs::path path_;
};

class RedirectStream {
public:
    RedirectStream(std::ios &stream, std::streambuf *replacement)
        : stream_(stream), old_buffer_(stream.rdbuf()), old_state_(stream.rdstate()),
          old_exceptions_(stream.exceptions()) {
        stream_.exceptions(std::ios::goodbit);
        stream_.rdbuf(replacement);
        stream_.clear();
    }

    RedirectStream(const RedirectStream &) = delete;
    RedirectStream &operator=(const RedirectStream &) = delete;

    ~RedirectStream() {
        stream_.exceptions(std::ios::goodbit);
        stream_.rdbuf(old_buffer_);
        stream_.clear(old_state_);
        stream_.exceptions(old_exceptions_);
    }

private:
    std::ios &stream_;
    std::streambuf *old_buffer_;
    std::ios::iostate old_state_;
    std::ios::iostate old_exceptions_;
};

// Stop the old EOF retry loop after a few prompts, so a failing test never hangs.
class LimitedOutputBuffer final : public std::streambuf {
public:
    bool exceeded() const { return exceeded_; }

protected:
    std::streamsize xsputn(const char *, std::streamsize count) override {
        add(static_cast<std::size_t>(count));
        return count;
    }

    int_type overflow(int_type character) override {
        if (!traits_type::eq_int_type(character, traits_type::eof())) {
            add(1);
        }
        return character;
    }

private:
    void add(std::size_t count) {
        written_ += count;
        if (written_ > 1024) {
            exceeded_ = true;
            throw std::runtime_error("Too many prompts after input EOF");
        }
    }

    std::size_t written_ = 0;
    bool exceeded_ = false;
};

void expectConfigRejected(const std::string &json, const std::string &case_name) {
    const TemporaryConfig config(json);
    bool rejected = false;
    try {
        static_cast<void>(parseConfig(config.path()));
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, "parseConfig accepted " + case_name);
}

void testMalformedJson() {
    expectConfigRejected(R"({"image_path": "main.png",)", "malformed JSON");
}

void testMissingRequiredField() {
    expectConfigRejected(
        R"({"filler_images_dir_path":"tiles","image_size":[2,2],"sub_images_size":[1,1],"strategy":"duplication","filtration":false,"show":false,"output_image_path":"out.png"})",
        "a config without image_path");
}

void testNonpositiveSizes() {
    expectConfigRejected(
        R"({"image_path":"main.png","filler_images_dir_path":"tiles","image_size":[0,2],"sub_images_size":[1,1],"strategy":"duplication","filtration":false,"show":false,"output_image_path":"out.png"})",
        "image_size with a zero width");
    expectConfigRejected(
        R"({"image_path":"main.png","filler_images_dir_path":"tiles","image_size":[2,2],"sub_images_size":[1,-1],"strategy":"duplication","filtration":false,"show":false,"output_image_path":"out.png"})",
        "sub_images_size with a negative height");
}

void testPreserveAspect(const std::string &answers, int expected_width,
                        int expected_height) {
    cv::Mat image(1, 2, CV_8UC3, cv::Scalar(10, 20, 30));
    std::istringstream input(answers);
    std::ostringstream output;
    {
        RedirectStream redirect_input(std::cin, input.rdbuf());
        RedirectStream redirect_output(std::cout, output.rdbuf());
        mainImageResize(image, {4, 4});
    }
    require(image.cols == expected_width && image.rows == expected_height,
            "Aspect ratio was not preserved for answers " + answers);
}

void testPreserveWidth() { testPreserveAspect("Y\nW\n", 4, 2); }

void testPreserveHeight() { testPreserveAspect("Y\nH\n", 8, 4); }

void testEOF() {
    cv::Mat image(1, 2, CV_8UC3, cv::Scalar(10, 20, 30));
    std::istringstream input;
    LimitedOutputBuffer output;
    {
        RedirectStream redirect_input(std::cin, input.rdbuf());
        RedirectStream redirect_output(std::cout, &output);
        std::cout.exceptions(std::ios::badbit);
        try {
            mainImageResize(image, {4, 4});
        } catch (const std::exception &) {
            // A clear input error is an acceptable way to handle EOF.
        }
    }
    require(!output.exceeded(), "mainImageResize repeatedly prompted after input EOF");
}

void run(const char *name, void (*test)(), int &failures) {
    try {
        test();
        std::cout << "PASS " << name << '\n';
    } catch (const std::exception &error) {
        ++failures;
        std::cerr << "FAIL " << name << ": " << error.what() << '\n';
    }
}

} // namespace

int main() {
    int failures = 0;
    run("malformed JSON", testMalformedJson, failures);
    run("missing required field", testMissingRequiredField, failures);
    run("nonpositive sizes", testNonpositiveSizes, failures);
    run("preserve width", testPreserveWidth, failures);
    run("preserve height", testPreserveHeight, failures);
    run("input EOF", testEOF, failures);
    return failures == 0 ? 0 : 1;
}
