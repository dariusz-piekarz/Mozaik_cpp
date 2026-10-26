#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

#include <opencv2/opencv.hpp>

#include "core.hpp"
#include "utils.hpp"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int tile_count = 96;
const cv::Size tile_size(16, 16);
const cv::Size source_size(160, 100);

std::uint64_t benchmark_sink = 0;

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = fs::temp_directory_path() / ("mozaik-benchmark-" + std::to_string(suffix));
        fs::create_directories(path_);
    }

    ~TemporaryDirectory() {
        std::error_code error;
        fs::remove_all(path_, error);
    }

    const fs::path &path() const { return path_; }

private:
    fs::path path_;
};

class CoutSilencer {
public:
    CoutSilencer() : previous_(std::cout.rdbuf(buffer_.rdbuf())) {}
    ~CoutSilencer() { std::cout.rdbuf(previous_); }

private:
    std::ostringstream buffer_;
    std::streambuf *previous_;
};

std::size_t residentBytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&counters),
                             sizeof(counters))) {
        return counters.WorkingSetSize;
    }
#elif defined(__linux__)
    std::ifstream statm("/proc/self/statm");
    std::size_t total_pages = 0;
    std::size_t resident_pages = 0;
    if (statm >> total_pages >> resident_pages) {
        return resident_pages * static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    }
#endif
    return 0;
}

template <typename Function>
std::size_t measurePeakResidentDelta(Function &&function) {
    const std::size_t baseline = residentBytes();
    std::atomic<bool> running{true};
    std::atomic<std::size_t> peak{baseline};
    std::jthread sampler([&] {
        while (running.load(std::memory_order_relaxed)) {
            const std::size_t current = residentBytes();
            std::size_t previous = peak.load(std::memory_order_relaxed);
            while (current > previous &&
                   !peak.compare_exchange_weak(previous, current, std::memory_order_relaxed)) {
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    try {
        function();
    } catch (...) {
        running.store(false, std::memory_order_relaxed);
        sampler.join();
        throw;
    }
    const std::size_t final_sample = residentBytes();
    std::size_t previous = peak.load(std::memory_order_relaxed);
    while (final_sample > previous &&
           !peak.compare_exchange_weak(previous, final_sample, std::memory_order_relaxed)) {
    }
    running.store(false, std::memory_order_relaxed);
    sampler.join();
    return peak.load(std::memory_order_relaxed) > baseline
               ? peak.load(std::memory_order_relaxed) - baseline
               : 0;
}

template <typename Function>
std::vector<double> measureTimes(int iterations, Function &&function) {
    function(); // Warm the filesystem cache, allocator and code paths.
    std::vector<double> milliseconds;
    milliseconds.reserve(iterations);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto start = Clock::now();
        benchmark_sink += static_cast<std::uint64_t>(function());
        const auto stop = Clock::now();
        milliseconds.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }
    return milliseconds;
}

template <typename Function>
double measureMilliseconds(Function &&function) {
    const auto start = Clock::now();
    benchmark_sink += static_cast<std::uint64_t>(function());
    const auto stop = Clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

struct PairedTimes {
    std::vector<double> first;
    std::vector<double> second;
};

template <typename FirstFunction, typename SecondFunction>
PairedTimes measurePairedTimes(int iterations, FirstFunction &&first, SecondFunction &&second) {
    first();
    second();

    PairedTimes times;
    times.first.reserve(iterations);
    times.second.reserve(iterations);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        // Alternate the order so that neither variant always gets the same cache/thermal position.
        if (iteration % 2 == 0) {
            times.first.push_back(measureMilliseconds(first));
            times.second.push_back(measureMilliseconds(second));
        } else {
            times.second.push_back(measureMilliseconds(second));
            times.first.push_back(measureMilliseconds(first));
        }
    }
    return times;
}

double median(std::vector<double> values) {
    std::ranges::sort(values);
    return values.size() % 2 == 0
               ? (values[values.size() / 2 - 1] + values[values.size() / 2]) / 2.0
               : values[values.size() / 2];
}

struct Result {
    std::string stage;
    int iterations;
    double minimum_ms;
    double median_ms;
    double p95_ms;
    std::size_t peak_rss_delta;
};

Result makeResult(std::string stage, int iterations, std::vector<double> times,
                  std::size_t peak_rss_delta) {
    std::ranges::sort(times);
    const double median_ms = times.size() % 2 == 0
                                 ? (times[times.size() / 2 - 1] + times[times.size() / 2]) / 2.0
                                 : times[times.size() / 2];
    const std::size_t p95_index = (times.size() * 95 + 99) / 100 - 1;
    return {std::move(stage), iterations, times.front(), median_ms, times[p95_index],
            peak_rss_delta};
}

std::vector<fs::path> createTiles(const fs::path &directory) {
    std::vector<fs::path> paths;
    paths.reserve(tile_count);
    for (int index = 0; index < tile_count; ++index) {
        cv::Mat tile(tile_size, CV_8UC3);
        for (int row = 0; row < tile.rows; ++row) {
            for (int col = 0; col < tile.cols; ++col) {
                tile.at<cv::Vec3b>(row, col) = cv::Vec3b(
                    (index * 37 + row * 11 + col * 3) % 256,
                    (index * 17 + row * 5 + col * 13) % 256,
                    (index * 29 + row * 7 + col * 19) % 256);
            }
        }
        const fs::path path = directory / ("tile-" + std::to_string(index) + ".png");
        if (!cv::imwrite(path.string(), tile)) {
            throw std::runtime_error("Cannot create benchmark tile: " + path.string());
        }
        paths.push_back(path);
    }
    return paths;
}

cv::Mat createSource() {
    cv::Mat source(source_size, CV_8UC3);
    for (int row = 0; row < source.rows; ++row) {
        for (int col = 0; col < source.cols; ++col) {
            source.at<cv::Vec3b>(row, col) = cv::Vec3b(
                (row * 7 + col * 3) % 256,
                (row * 11 + col * 5) % 256,
                (row * 13 + col * 17) % 256);
        }
    }
    return source;
}

std::uint64_t imageChecksum(const cv::Mat &image) {
    const cv::Scalar sum = cv::sum(image);
    return static_cast<std::uint64_t>(sum[0] + sum[1] + sum[2]);
}

std::uint64_t quickImageToken(const cv::Mat &image) {
    const auto last = image.at<cv::Vec3b>(image.rows - 1, image.cols - 1);
    return image.total() + last[0] + last[1] + last[2];
}

int parseIterations(int argc, char **argv) {
    if (argc == 1) {
        return 5;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "Usage: mozaik_benchmark [--iterations N]\n";
        return 0;
    }
    if (argc != 3 || std::string_view(argv[1]) != "--iterations") {
        throw std::invalid_argument("Usage: mozaik_benchmark [--iterations N]");
    }
    const int iterations = std::stoi(argv[2]);
    if (iterations <= 0) {
        throw std::invalid_argument("Iteration count must be positive");
    }
    return iterations;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const int iterations = parseIterations(argc, argv);
        if (iterations == 0) {
            return 0;
        }

        TemporaryDirectory temporary;
        const auto paths = createTiles(temporary.path());
        const cv::Mat source = createSource();
        std::vector<Result> results;

        std::vector<cv::Mat> tiles;
        const std::size_t load_memory = measurePeakResidentDelta([&] {
            CoutSilencer silence;
            tiles = readImages(paths, tile_size);
        });
        auto load_times = measureTimes(iterations, [&] {
            CoutSilencer silence;
            const auto loaded = readImages(paths, tile_size);
            return loaded.size() + imageChecksum(loaded.front());
        });
        results.push_back(makeResult("tile_loading", iterations, std::move(load_times), load_memory));

        const auto means = calculateMeans(tiles);
        std::uint64_t search_checksum = 0;
        const std::size_t search_memory = measurePeakResidentDelta([&] {
            for (int row = 0; row < source.rows; ++row) {
                for (int col = 0; col < source.cols; ++col) {
                    search_checksum += static_cast<std::uint64_t>(
                        closestImage(means, source.at<cv::Vec3b>(row, col)));
                }
            }
        });
        benchmark_sink += search_checksum;
        auto search_times = measureTimes(iterations, [&] {
            std::uint64_t checksum = 0;
            for (int row = 0; row < source.rows; ++row) {
                for (int col = 0; col < source.cols; ++col) {
                    checksum += static_cast<std::uint64_t>(
                        closestImage(means, source.at<cv::Vec3b>(row, col)));
                }
            }
            return checksum;
        });
        results.push_back(makeResult("nearest_search", iterations, std::move(search_times), search_memory));

        cv::Mat sequential_output;
        const std::size_t sequential_memory = measurePeakResidentDelta([&] {
            sequential_output = renderMosaic(tiles, source, "duplication", true,
                                             RenderExecution::sequential);
        });
        const auto render_sequential = [&] {
            const cv::Mat output = renderMosaic(tiles, source, "duplication", true,
                                                RenderExecution::sequential);
            return quickImageToken(output);
        };

        cv::Mat parallel_output;
        const std::size_t parallel_memory = measurePeakResidentDelta([&] {
            parallel_output = renderMosaic(tiles, source, "duplication", true,
                                           RenderExecution::parallel);
        });
        const auto render_parallel = [&] {
            const cv::Mat output = renderMosaic(tiles, source, "duplication", true,
                                                RenderExecution::parallel);
            return quickImageToken(output);
        };

        auto render_times = measurePairedTimes(iterations, render_sequential, render_parallel);
        std::vector<double> paired_speedups;
        paired_speedups.reserve(render_times.first.size());
        for (std::size_t index = 0; index < render_times.first.size(); ++index) {
            paired_speedups.push_back(render_times.first[index] / render_times.second[index]);
        }
        const double parallel_speedup = median(std::move(paired_speedups));
        results.push_back(makeResult("render_sequential", iterations,
                                     std::move(render_times.first), sequential_memory));
        results.push_back(makeResult("render_parallel", iterations,
                                     std::move(render_times.second), parallel_memory));

        if (cv::countNonZero(sequential_output.reshape(1) != parallel_output.reshape(1)) != 0) {
            throw std::runtime_error("Parallel renderer produced a different image");
        }

        std::cout << "stage,iterations,min_ms,median_ms,p95_ms,peak_rss_delta_mib\n";
        for (const auto &result : results) {
            std::cout << result.stage << ',' << result.iterations << ',' << std::fixed
                      << std::setprecision(3) << result.minimum_ms << ',' << result.median_ms << ','
                      << result.p95_ms << ','
                      << static_cast<double>(result.peak_rss_delta) / (1024.0 * 1024.0) << '\n';
        }
        std::cerr << "dataset: " << tile_count << " tiles " << tile_size.width << 'x'
                  << tile_size.height << ", source " << source_size.width << 'x'
                  << source_size.height << ", output " << source_size.width * tile_size.width
                  << 'x' << source_size.height * tile_size.height << "\n";
        std::cerr << "hardware threads: " << std::thread::hardware_concurrency() << "\n";
        std::cerr << "render parallel paired median speedup: " << std::fixed
                  << std::setprecision(3) << parallel_speedup << "x\n";
        std::cerr << "checksum: " << benchmark_sink << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
