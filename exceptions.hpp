#pragma once
#include <exception>
#include <string>

class ExtensionError final : public std::exception {
    std::string message;
public:
    explicit ExtensionError(std::string msg);
    [[nodiscard]] const char* what() const noexcept override;
};
