# Podręcznik: środowisko, zależności i uruchomienie

Ten plik zapisuje polecenia potrzebne do odtworzenia środowiska na Windows x64. Sekcja „Stan sprawdzony lokalnie” opisuje oddzielnie to, co da się potwierdzić w bieżącym katalogu. Polecenia instalacyjne poniżej są instrukcją dla nowego środowiska; nie są dziennikiem wykonanych tutaj pobrań.

## 1. Skąd pobrać narzędzia

Potrzebne są Git, [CMake](https://cmake.org/download/) w wersji co najmniej 3.21, [Ninja](https://github.com/ninja-build/ninja/releases), [LLVM-MinGW](https://github.com/mstorsjo/llvm-mingw/releases) dla Windows x64 oraz [vcpkg](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started-packaging). LLVM-MinGW zawiera `clang` i `clang++`; samodzielna instalacja LLVM bez środowiska MinGW nie jest odpowiednikiem tego zestawu. Git można pobrać ze [strony projektu](https://git-scm.com/download/win). Po instalacji Gita otwórz nową sesję PowerShell.

Poniższy przykład pobiera takie same wydania narzędzi, jakie są obecnie widoczne lokalnie: CMake 4.4.4, Ninja 1.13.2 i LLVM-MinGW 20260908 UCRT x86_64. Uruchom go w PowerShell na nowej instalacji; archiwa zostaną rozpakowane do `~/Tools/cpp`. Jeżeli wybierzesz inne wydania, zmień odpowiednio nazwy i adresy archiwów.

```powershell
$toolRoot = Join-Path $env:USERPROFILE 'Tools\cpp'
New-Item -ItemType Directory -Force -Path $toolRoot | Out-Null

$cmakeZip = Join-Path $toolRoot 'cmake-4.4.4-windows-x86_64.zip'
Invoke-WebRequest 'https://github.com/Kitware/CMake/releases/download/v4.4.4/cmake-4.4.4-windows-x86_64.zip' -OutFile $cmakeZip
Expand-Archive -LiteralPath $cmakeZip -DestinationPath $toolRoot

$ninjaZip = Join-Path $toolRoot 'ninja-1.13.2-win.zip'
Invoke-WebRequest 'https://github.com/ninja-build/ninja/releases/download/v1.13.2/ninja-win.zip' -OutFile $ninjaZip
Expand-Archive -LiteralPath $ninjaZip -DestinationPath (Join-Path $toolRoot 'ninja-1.13.2')

$llvmZip = Join-Path $toolRoot 'llvm-mingw-20260908-ucrt-x86_64.zip'
Invoke-WebRequest 'https://github.com/mstorsjo/llvm-mingw/releases/download/20260908/llvm-mingw-20260908-ucrt-x86_64.zip' -OutFile $llvmZip
Expand-Archive -LiteralPath $llvmZip -DestinationPath $toolRoot
```

Dodaj katalogi z programami do `PATH` bieżącej sesji. Ta zmiana nie zapisuje się na stałe; w nowym oknie PowerShell wykonaj ją ponownie albo dodaj te katalogi do zmiennej środowiskowej użytkownika w ustawieniach Windows. Kolejność ma znaczenie, jeśli masz też inne kompilatory na `PATH`.

```powershell
$env:Path = "$(Join-Path $toolRoot 'cmake-4.4.4-windows-x86_64\bin');$(Join-Path $toolRoot 'ninja-1.13.2');$(Join-Path $toolRoot 'llvm-mingw-20260908-ucrt-x86_64\bin');$env:Path"
Get-Command git, cmake, ninja, clang, clang++
cmake --version
ninja --version
clang++ --version
```

Sklonuj vcpkg do tego samego katalogu. `vcpkg.json` wskazuje commit bazowy `9e593bb18ea69cc5095e012465dcd675a822ed0d`; przejście na ten commit pozwala odtworzyć użyty zestaw portów. Wykonaj `git clone` tylko wtedy, gdy katalog `vcpkg` jeszcze nie istnieje.

```powershell
$env:VCPKG_ROOT = Join-Path $toolRoot 'vcpkg'
git clone https://github.com/microsoft/vcpkg.git $env:VCPKG_ROOT
git -C $env:VCPKG_ROOT checkout 9e593bb18ea69cc5095e012465dcd675a822ed0d
& (Join-Path $env:VCPKG_ROOT 'bootstrap-vcpkg.bat')
& (Join-Path $env:VCPKG_ROOT 'vcpkg.exe') version
```

`VCPKG_ROOT` również obowiązuje tylko w tej sesji. Preset CMake czyta z niego ścieżkę do `scripts/buildsystems/vcpkg.cmake`, dlatego ustaw tę zmienną **przed** pierwszym konfigurowaniem. [Instrukcja Microsoftu](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started-packaging) opisuje też trwałe ustawienie zmiennej. Manifest `vcpkg.json` żąda JsonCpp, spdlog oraz OpenCV z funkcjami `highgui`, `jpeg`, `openmp`, `png` i `win32ui`; CMake powinien pobrać i zbudować te zależności przy konfiguracji w trybie manifestu. Triplet `x64-mingw-static` odpowiada 64-bitowemu LLVM-MinGW użytemu przez preset. Pierwsza konfiguracja może długo budować OpenCV.

## 2. Konfiguracja, kompilacja i test

Wykonaj poniższe polecenia w katalogu `Mozaik_cpp`. Są to polecenia dla presetu z `CMakePresets.json`; w tym przeglądzie **nie potwierdzono**, że pełna instalacja zależności w trybie manifestu zakończyła się na tej maszynie sukcesem.

```powershell
Set-Location (Join-Path $env:USERPROFILE 'Desktop\projekty\cpp\Mozaik_cpp')
cmake --preset windows-llvm-mingw
cmake --build --preset windows-llvm-mingw
ctest --test-dir .\build\windows-llvm-mingw --output-on-failure
```

`CMakeLists.txt` rejestruje w CTest `image_paths_test` i `loader_order_test`. Pozostałe pliki w `tests/` nie oznaczają jeszcze, że uruchamia je `ctest`.

## 3. Przykład uruchomienia z obecnym kodem

Aktualny `main.cpp` wywołuje `mozaik2()`, który **pyta na wejściu standardowym o ścieżkę konfiguracji**; argument przekazany po nazwie EXE nie jest używany. Ścieżki `input.png`, `tiles` i `output.png` z `examples/config.json` są obecnie rozwiązywane względem katalogu pracy procesu. Dlatego uruchom program z katalogu `examples`, a na pytanie `Provide a path to the config file:` wpisz `.\config.json` i naciśnij Enter:

```powershell
Push-Location .\examples
& ..\build\windows-llvm-mingw\Mozaik.exe
# odpowiedź na pytanie programu: .\config.json
Pop-Location
Get-Item .\examples\output.png
```

Przykład ma siatkę `4 × 4` kafelków po `8 × 8` pikseli, więc zamierzony wynik ma `32 × 32` piksele. Jeżeli program dodatkowo zapyta o zmianę proporcji, odpowiedz interaktywnie zgodnie z komunikatem. `show` jest wyłączone, więc przykład nie otwiera okna podglądu.

Pliki wejściowe przykładu znajdują się w tym repozytorium: `examples/input.png` oraz cztery pliki `examples/tiles/{blue,green,red,yellow}.png`. Samo uruchomienie przykładu nie pobiera obrazów z sieci. Historia pochodzenia tych pięciu PNG, sposób ich wytworzenia i ich osobna licencja nie są zapisane w repozytorium; nie należy przypisywać im niepotwierdzonego źródła. `examples/output.png` jest wynikiem programu i jest ignorowany przez Git.

## 4. Stan sprawdzony lokalnie

W tej sesji polecenie `cmake --build build/local --target Mozaik image_paths_test` zakończyło się kodem 0, a `ctest --test-dir build/local -R ^image_paths_test$ --output-on-failure` pokazał `1/1 Passed`. Przykład uruchomiono z katalogu `examples` przez przekazanie ścieżki konfiguracji na standardowe wejście; program zgłosił `Program successfully run` i utworzył `examples/output.png`. Użyto istniejącej konfiguracji `build/local`, bez pobierania nowych zależności.

W obecnym środowisku wykryto CMake 4.4.4, Ninja 1.13.2, Clang 23.1.1 z LLVM-MinGW 20260908 oraz vcpkg z katalogu `C:\Users\DXD\Tools\cpp\vcpkg` na commicie `9e593bb18ea69cc5095e012465dcd675a822ed0d`. Są to obserwacje, a nie dowód, że wykonano podane wyżej polecenia pobierania.

Istniejący `build/local/CMakeCache.txt` wskazuje osobną konfigurację Ninja z `VCPKG_MANIFEST_MODE=OFF`, triplet `x64-mingw-static`, OpenCV i spdlog w instalacji vcpkg oraz JsonCpp w `C:\Users\DXD\Tools\cpp\jsoncpp-install`. To lokalny wariant awaryjny, **nie** wynik presetu manifestowego. `build/local/Testing/Temporary/LastTest.log` zapisuje udany test `image_paths_test` z 23 października 2026. Plik `build/local/smoke.png` istnieje, ale sam artefakt nie potwierdza dokładnej komendy, którą go utworzono. W razie problemu z manifestem przykład osobnej konfiguracji zależności znajduje się w `README.md`; nie ma tu kompletnego, sprawdzonego przepisu na zbudowanie osobnej instalacji JsonCpp.

W tej sesji sprawdzono także test równoległego czytania kafelków w istniejącej konfiguracji lokalnej:

```powershell
cmake --build build/local --target loader_order_test
ctest --test-dir build/local -R '^loader_order_test$' --output-on-failure
```

Oba polecenia zakończyły się kodem 0; CTest pokazał `1/1 Passed` w 1,67 s. Test używa 64 ścieżek, sprawdza zachowanie kolejności przy dwóch nieczytelnych plikach i przeniesienie wyjątku z wątku.

Po integracji poprawki wejścia z gałęzi `fix/validation` trzeba zaktualizować powyższy sposób uruchamiania: zmieni ona obsługę argumentu i ścieżek względnych. Do tego czasu obowiązuje procedura opisana w sekcji 3.
