#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <opencv2/opencv.hpp>

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
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = fs::temp_directory_path() / ("mozaik e2e " + std::to_string(stamp));
        require(fs::create_directory(path_), "Could not create test directory");
    }

    ~TemporaryDirectory() {
        std::error_code ignored;
        fs::remove_all(path_, ignored);
    }

    const fs::path &path() const { return path_; }

private:
    fs::path path_;
};

int runMozaik(const fs::path &executable, const fs::path &config) {
#ifdef _WIN32
    std::wstring command = L"\"" + executable.wstring() + L"\" \"" + config.wstring() + L"\"";
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, FALSE,
                        CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process)) {
        throw std::runtime_error("Could not start Mozaik (Windows error " +
                                 std::to_string(GetLastError()) + ")");
    }

    const DWORD wait_result = WaitForSingleObject(process.hProcess, 30000);
    if (wait_result != WAIT_OBJECT_0) {
        TerminateProcess(process.hProcess, 1);
        WaitForSingleObject(process.hProcess, 1000);
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
        throw std::runtime_error("Mozaik did not complete within 30 seconds");
    }
    DWORD exit_code = 0;
    const bool got_exit_code = GetExitCodeProcess(process.hProcess, &exit_code) != 0;
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    require(got_exit_code, "Could not read Mozaik exit code");
    return static_cast<int>(exit_code);
#else
    const pid_t child = fork();
    if (child == 0) {
        execl(executable.c_str(), executable.c_str(), config.c_str(), nullptr);
        _exit(127);
    }
    require(child > 0, "Could not start Mozaik");
    int status = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (true) {
        const pid_t result = waitpid(child, &status, WNOHANG);
        if (result == child) {
            break;
        }
        require(result == 0, "Could not wait for Mozaik");
        if (std::chrono::steady_clock::now() >= deadline) {
            kill(child, SIGKILL);
            waitpid(child, &status, 0);
            throw std::runtime_error("Mozaik did not complete within 30 seconds");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

void requireBlock(const cv::Mat &image, int top, int left, const cv::Vec3b &expected) {
    for (int row = top; row < top + 16; ++row) {
        for (int col = left; col < left + 16; ++col) {
            const auto actual = image.at<cv::Vec3b>(row, col);
            for (int channel = 0; channel < 3; ++channel) {
                require(actual[channel] == expected[channel], "Output PNG has incorrect pixels");
            }
        }
    }
}

void testEndToEnd(const fs::path &executable) {
    TemporaryDirectory temp;
    const fs::path tiles_dir = temp.path() / "tiles";
    fs::create_directory(tiles_dir);

    cv::Mat source(2, 2, CV_8UC3, cv::Scalar(0, 0, 0));
    source.at<cv::Vec3b>(0, 0) = cv::Vec3b(0, 255, 0);
    source.at<cv::Vec3b>(0, 1) = cv::Vec3b(0, 255, 0);
    const cv::Mat black_tile(16, 16, CV_8UC3, cv::Scalar(0, 0, 0));
    const cv::Mat greenish_tile(16, 16, CV_8UC3, cv::Scalar(100, 255, 100));
    require(cv::imwrite((temp.path() / "main.png").string(), source), "Could not write source PNG");
    require(cv::imwrite((tiles_dir / "a_black.png").string(), black_tile),
            "Could not write black tile PNG");
    require(cv::imwrite((tiles_dir / "b_greenish.png").string(), greenish_tile),
            "Could not write green tile PNG");

    const fs::path config_path = temp.path() / "config with spaces.json";
    {
        std::ofstream config(config_path);
        require(static_cast<bool>(config), "Could not create config file");
        config << R"({"image_path":"main.png","filler_images_dir_path":"tiles",)"
               << R"("image_size":[2,2],"sub_images_size":[16,16],)"
               << R"("strategy":"pixel_mean","filtration":true,"show":false,)"
               << R"("output_image_path":"result.png"})";
        config.close();
        require(static_cast<bool>(config), "Could not write config file");
    }

    require(runMozaik(executable, config_path) == 0, "Mozaik returned a nonzero exit code");
    const cv::Mat output = cv::imread((temp.path() / "result.png").string(), cv::IMREAD_COLOR);
    require(!output.empty(), "Mozaik did not save the output PNG");
    require(output.size() == cv::Size(32, 32), "Output PNG is not 32x32");
    requireBlock(output, 0, 0, cv::Vec3b(0, 255, 0));
    requireBlock(output, 0, 16, cv::Vec3b(0, 255, 0));
    requireBlock(output, 16, 0, cv::Vec3b(0, 0, 0));
    requireBlock(output, 16, 16, cv::Vec3b(0, 0, 0));
}

} // namespace

int main(int argc, char *argv[]) {
    try {
        require(argc == 2, "Expected the Mozaik executable path as an argument");
        testEndToEnd(argv[1]);
        std::cout << "PASS end-to-end PNG render\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL end-to-end PNG render: " << error.what() << '\n';
        return 1;
    }
}
