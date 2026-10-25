#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "core.hpp"
#include "utils.hpp"

namespace {

namespace fs = std::filesystem;

void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        static std::atomic<unsigned> sequence{0};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = fs::temp_directory_path() /
                ("mozaik validation " + std::to_string(stamp) + " " +
                 std::to_string(sequence++));
        fs::create_directories(path_);
    }

    TemporaryDirectory(const TemporaryDirectory &) = delete;
    TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;

    ~TemporaryDirectory() {
        std::error_code error;
        fs::remove_all(path_, error);
    }

    const fs::path &path() const { return path_; }

    fs::path write(const std::string &name, const std::string &contents) const {
        const fs::path file_path = path_ / name;
        std::ofstream file(file_path, std::ios::binary);
        require(static_cast<bool>(file), "Could not create test file: " + file_path.string());
        file << contents;
        require(static_cast<bool>(file), "Could not write test file: " + file_path.string());
        return file_path;
    }

private:
    fs::path path_;
};

class RedirectInput {
public:
    explicit RedirectInput(std::streambuf *replacement)
        : old_buffer_(std::cin.rdbuf()), old_state_(std::cin.rdstate()) {
        std::cin.rdbuf(replacement);
        std::cin.clear();
    }

    RedirectInput(const RedirectInput &) = delete;
    RedirectInput &operator=(const RedirectInput &) = delete;

    ~RedirectInput() {
        std::cin.rdbuf(old_buffer_);
        std::cin.clear(old_state_);
    }

private:
    std::streambuf *old_buffer_;
    std::ios::iostate old_state_;
};

constexpr std::string_view valid_json = R"({
    "image_path": "main.png",
    "filler_images_dir_path": "tiles",
    "image_size": [2, 2],
    "sub_images_size": [1, 1],
    "aspect_ratio": "preserve_width",
    "strategy": "duplication",
    "filtration": false,
    "show": false,
    "output_image_path": "result.png"
})";

std::string replace_once(std::string value, std::string_view old_text,
                         std::string_view new_text) {
    const auto position = value.find(old_text);
    require(position != std::string::npos, "Invalid test fixture");
    value.replace(position, old_text.size(), new_text);
    return value;
}

template<class Function>
void expect_error(Function action, std::string_view expected_message) {
    try {
        action();
    } catch (const std::exception &error) {
        require(std::string_view(error.what()).find(expected_message) != std::string_view::npos,
                "Unexpected error: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("Expected error containing: " + std::string(expected_message));
}

void test_config_validation() {
    TemporaryDirectory temp;
    const auto check = [&temp](const std::string &json, std::string_view error) {
        const fs::path config = temp.write("config.json", json);
        expect_error([&] { static_cast<void>(parseConfig(config)); }, error);
    };

    check(R"({"image_path": )", "Invalid JSON");
    check(replace_once(std::string(valid_json), "\"image_path\": \"main.png\",", ""),
          "image_path");
    check(replace_once(std::string(valid_json), "[2, 2]", "[0, 2]"), "image_size");
    check(replace_once(std::string(valid_json), "[1, 1]", "[1, -1]"), "sub_images_size");
    check(replace_once(std::string(valid_json), "\"show\": false", "\"show\": \"false\""),
          "show");
    check(replace_once(std::string(valid_json), "\"duplication\"", "\"missing\""),
          "Unsupported config strategy");
    check(replace_once(std::string(valid_json), "\"preserve_width\"", "\"missing\""),
          "Unsupported aspect_ratio mode");
    check(replace_once(std::string(valid_json), "\"preserve_width\"", "false"),
          "aspect_ratio");
    check(replace_once(std::string(valid_json), "\"show\": false", "\"unexpected\": 1, \"show\": false"),
          "Unknown config field");
}

void test_relative_paths_and_spaces() {
    TemporaryDirectory temp;
    const fs::path config_path = temp.write("config with spaces.json", std::string(valid_json));
    const Config config = parseConfig(config_path);
    require(config.aspect_ratio == "preserve_width", "Incorrect aspect-ratio mode");
    require(fs::path(config.image_path) == temp.path() / "main.png", "Incorrect main image path");
    require(fs::path(config.filler_images_dir_path) == temp.path() / "tiles",
            "Incorrect tile directory path");
    require(fs::path(config.output_image_path) == temp.path() / "result.png",
            "Incorrect output path");

    const auto no_output = replace_once(std::string(valid_json),
                                        "\"output_image_path\": \"result.png\"",
                                        "\"output_image_path\": \"\"");
    require(parseConfig(temp.write("config.json", no_output)).output_image_path.empty(),
            "Empty optional output path was rejected");

    const auto without_aspect = replace_once(std::string(valid_json),
        "    \"aspect_ratio\": \"preserve_width\",\n", "");
    require(parseConfig(temp.write("legacy.json", without_aspect)).aspect_ratio == "interactive",
            "Missing aspect-ratio mode did not default to interactive");
}

void test_unreadable_main_image() {
    TemporaryDirectory temp;
    temp.write("main.png", "This is not a PNG image");
    fs::create_directory(temp.path() / "tiles");
    cv::Mat tile(1, 1, CV_8UC3, cv::Scalar(1, 2, 3));
    require(cv::imwrite((temp.path() / "tiles" / "tile.png").string(), tile),
            "Could not create tile fixture");
    const fs::path config_path = temp.write("config.json", std::string(valid_json));
    expect_error([&] { mozaikCoreApp(config_path); }, "Cannot read main image");
}

void test_failed_output_save() {
    TemporaryDirectory temp;
    cv::Mat tile(1, 1, CV_8UC3, cv::Scalar(1, 2, 3));
    std::vector<std::vector<cv::Mat>> tiles{{tile}};
    const auto output = (temp.path() / "missing directory" / "result.png").string();
    expect_error([&] { glueImages(tiles, false, output); }, "Failed to save output image");
}

void test_failed_active_output_save() {
    TemporaryDirectory temp;
    const cv::Mat image(2, 2, CV_8UC3, cv::Scalar(1, 2, 3));
    require(cv::imwrite((temp.path() / "main.png").string(), image),
            "Could not create main image fixture");
    fs::create_directory(temp.path() / "tiles");
    require(cv::imwrite((temp.path() / "tiles" / "tile.png").string(), image),
            "Could not create tile fixture");
    const auto json = replace_once(std::string(valid_json),
                                   "\"output_image_path\": \"result.png\"",
                                   "\"output_image_path\": \"missing directory/result.png\"");
    const fs::path config_path = temp.write("config.json", json);
    expect_error([&] { mozaikCoreApp(config_path); }, "Failed to save output image");
}

void test_interactive_path_with_spaces() {
    TemporaryDirectory temp;
    const fs::path config_path = temp.write("invalid config.json", R"({"image_path": )");
    std::istringstream input(config_path.string() + "\n");
    RedirectInput redirect(input.rdbuf());
    expect_error([] { mozaik2(); }, "Invalid JSON");
}

void test_interactive_eof() {
    std::istringstream input;
    RedirectInput redirect(input.rdbuf());
    expect_error([] { mozaik2(); }, "Input ended before a config path");
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
    run("config validation", test_config_validation, failures);
    run("relative paths and spaces", test_relative_paths_and_spaces, failures);
    run("unreadable main image", test_unreadable_main_image, failures);
    run("failed output save", test_failed_output_save, failures);
    run("failed active output save", test_failed_active_output_save, failures);
    run("interactive path with spaces", test_interactive_path_with_spaces, failures);
    run("interactive EOF", test_interactive_eof, failures);
    return failures == 0 ? 0 : 1;
}
