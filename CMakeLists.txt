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

    add_executable(aspect_ratio_test tests/aspect_ratio_test.cpp)
    target_link_libraries(aspect_ratio_test PRIVATE mozaik_core)
    add_test(NAME aspect_ratio_test COMMAND aspect_ratio_test)

    add_executable(random_selection_test tests/random_selection_test.cpp)
    target_link_libraries(random_selection_test PRIVATE mozaik_core)
    add_test(NAME random_selection_test COMMAND random_selection_test)

    add_executable(mosaic_render_test tests/mosaic_render_test.cpp)
    target_link_libraries(mosaic_render_test PRIVATE mozaik_core)
    add_test(NAME mosaic_render_test COMMAND mosaic_render_test)

    add_executable(validation_resize_test tests/validation_resize_test.cpp)
    target_link_libraries(validation_resize_test PRIVATE mozaik_core)
    add_test(NAME validation_resize_test COMMAND validation_resize_test)

    add_executable(mosaic_e2e_test tests/mosaic_e2e_test.cpp)
    target_link_libraries(mosaic_e2e_test PRIVATE mozaik_core)
    add_dependencies(mosaic_e2e_test Mozaik)
    add_test(NAME mosaic_e2e_test COMMAND mosaic_e2e_test $<TARGET_FILE:Mozaik>)
    set_tests_properties(mosaic_e2e_test PROPERTIES TIMEOUT 45)

    add_executable(mozaik_config_lookup_test tests/mozaik_config_lookup_test.cpp)
    target_link_libraries(mozaik_config_lookup_test PRIVATE mozaik_core)
    add_test(NAME mozaik_config_lookup_test COMMAND mozaik_config_lookup_test)
endif()
