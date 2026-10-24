cmake_minimum_required(VERSION 3.21)

project(Mozaik VERSION 0.1.0 LANGUAGES C CXX)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
find_package(spdlog CONFIG REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)
find_package(OpenMP COMPONENTS CXX)

add_library(mozaik_core STATIC
    core.cpp
    exceptions.cpp
    utils.cpp
)
target_compile_features(mozaik_core PUBLIC cxx_std_20)
set_target_properties(mozaik_core PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(mozaik_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(mozaik_core
    PUBLIC ${OpenCV_LIBS} spdlog::spdlog
    PRIVATE JsonCpp::JsonCpp
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(mozaik_core PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(Mozaik main.cpp)
target_link_libraries(Mozaik PRIVATE mozaik_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(image_paths_test tests/image_paths_test.cpp)
    target_link_libraries(image_paths_test PRIVATE mozaik_core)
    add_test(NAME image_paths_test COMMAND image_paths_test)

    add_executable(loader_order_test tests/loader_order_test.cpp)
    target_link_libraries(loader_order_test PRIVATE mozaik_core)
    add_test(NAME loader_order_test COMMAND loader_order_test)

    add_executable(validation_test tests/validation_test.cpp)
    target_link_libraries(validation_test PRIVATE mozaik_core)
    add_test(NAME validation_test COMMAND validation_test)
endif()
