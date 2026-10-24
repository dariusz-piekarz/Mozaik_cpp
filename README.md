# Mozaik_cpp

Mozaik creates a photo mosaic from a source image and a directory of tile images. It is a C++20 version of the [Python Mozaik project](https://github.com/dariusz-piekarz/Mozaik). It can optionally adjust tile colors to match the source image.

## Build on Windows

Install CMake 3.21 or newer, Ninja, an LLVM-MinGW C++20 toolchain, and [vcpkg](https://github.com/microsoft/vcpkg). Put `cmake`, `ninja`, `clang`, and `clang++` on `PATH` and set `VCPKG_ROOT` to the vcpkg directory. The repository's vcpkg manifest installs OpenCV, JsonCpp, and spdlog automatically. The first configure may take some time while it builds these libraries.

From PowerShell in the repository directory:

```powershell
cmake --preset windows-llvm-mingw
cmake --build --preset windows-llvm-mingw
.\build\windows-llvm-mingw\Mozaik.exe .\examples\config.json
```

The sample produces `examples/output.png` (32 × 32 pixels). It uses the included input and tile PNGs, so no personal image paths are needed.

If automatic vcpkg installation cannot complete, install the dependencies separately and configure with the installed packages. For example, when OpenCV and spdlog are in the classic vcpkg triplet and JsonCpp is in a separate prefix:

```powershell
cmake -S . -B build/local -G Ninja `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-mingw-static -DVCPKG_MANIFEST_MODE=OFF `
  -Djsoncpp_DIR="C:/path/to/jsoncpp-install/lib/cmake/jsoncpp" `
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
cmake --build build/local
ctest --test-dir build/local --output-on-failure
```

## Configuration

Copy `examples/config.json` and change these fields for your own images:

| Field | Meaning |
| --- | --- |
| `image_path` | Source image file |
| `filler_images_dir_path` | Directory containing `.jpg`, `.jpeg`, or `.png` tiles |
| `image_size` | Mosaic width and height in tiles |
| `sub_images_size` | Width and height of each output tile in pixels |
| `strategy` | `duplication`, `pixel_mean`, or `pixel_mean_random` |
| `filtration` | Adjust selected tile colors to match source pixels |
| `show` | Open a window with the finished mosaic |
| `output_image_path` | Output file; leave empty to skip saving |

Relative paths in the configuration are interpreted from the directory containing that configuration file. Run with the JSON path as an argument, or omit the argument to enter it interactively. If the requested `image_size` changes the source image's aspect ratio, the program asks whether to preserve it.
