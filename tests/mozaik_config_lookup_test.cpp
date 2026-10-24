#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>

#include "core.hpp"

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
        path_ = fs::temp_directory_path() / ("mozaik lookup " + std::to_string(stamp));
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

class ScopedCurrentDirectory {
public:
    explicit ScopedCurrentDirectory(const fs::path &new_path)
        : previous_(fs::current_path()) {
        fs::current_path(new_path);
    }

    ~ScopedCurrentDirectory() {
        std::error_code ignored;
        fs::current_path(previous_, ignored);
    }

private:
    fs::path previous_;
};

std::string errorFromMozaik() {
    try {
        mozaik();
    } catch (const std::exception &error) {
        return error.what();
    }
    throw std::runtime_error("mozaik() unexpectedly succeeded");
}

void writeInvalidConfig(const fs::path &path, const char *marker) {
    std::ofstream file(path);
    require(static_cast<bool>(file), "Could not create config fixture");
    file << "{\"" << marker << "\":true}";
    file.close();
    require(static_cast<bool>(file), "Could not write config fixture");
}

void testLookup() {
    TemporaryDirectory temp;
    fs::create_directory(temp.path() / "examples");
    const fs::path local_config = temp.path() / "config.json";
    const fs::path example_config = temp.path() / "examples" / "config.json";
    writeInvalidConfig(local_config, "only_root");
    writeInvalidConfig(example_config, "only_example");

    ScopedCurrentDirectory cwd(temp.path());
    const std::string local_error = errorFromMozaik();
    require(local_error.find("only_root") != std::string::npos,
            "mozaik() did not choose ./config.json first");

    fs::remove(local_config);
    const std::string example_error = errorFromMozaik();
    require(example_error.find("only_example") != std::string::npos,
            "mozaik() did not fall back to ./examples/config.json");

    fs::remove(example_config);
    const std::string missing_error = errorFromMozaik();
    require(missing_error.find("config.json") != std::string::npos &&
            missing_error.find("examples") != std::string::npos,
            "Missing config error does not identify the searched paths");
}

} // namespace

int main() {
    try {
        testLookup();
        std::cout << "PASS config lookup\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL config lookup: " << error.what() << '\n';
        return 1;
    }
}
