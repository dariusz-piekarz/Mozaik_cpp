#include <string>
#include <utility>

#include "exceptions.hpp"


ExtensionError::ExtensionError(std::string msg) : message(std::move(msg)){};

const char* ExtensionError::what() const noexcept {
    return message.c_str();
};
