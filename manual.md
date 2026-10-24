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

Dodaj katalogi z programami do `PATH` bieżącej sesji. Ta zmiana nie zapisuje się na stałe; w nowym oknie PowerShell ponownie zdefiniuj `$toolRoot` i wykonaj blok ustawiający `PATH`, albo dodaj te katalogi do zmiennej środowiskowej użytkownika w ustawieniach Windows. Kolejność ma znaczenie, jeśli masz też inne kompilatory na `PATH`.

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

`CMakeLists.txt` rejestruje w CTest dziewięć testów: `image_paths_test`, `loader_order_test`, `validation_test`, `aspect_ratio_test`, `random_selection_test`, `mosaic_render_test`, `validation_resize_test`, `mosaic_e2e_test` i `mozaik_config_lookup_test`.

## 3. Przykład uruchomienia i dane

Po poprawce walidacji program przyjmuje ścieżkę JSON jako opcjonalny argument. Ścieżki obrazów i wyniku w JSON są liczone względem katalogu tego pliku. Z katalogu głównego repozytorium uruchom:

```powershell
.\build\windows-llvm-mingw\Mozaik.exe .\examples\config.json
Get-Item .\examples\output.png
```

Bez argumentu program zapyta o ścieżkę konfiguracji. Wpisanie jej jako całej linii pozwala używać nazw ze spacjami. Błędny JSON, nieczytelny obraz i błąd zapisu kończą program z komunikatem i kodem niezerowym. Publiczna funkcja biblioteczna `mozaik()` wybiera `./config.json` z bieżącego katalogu, a gdy go nie ma — `./examples/config.json`; polecenie CLI z argumentem korzysta bezpośrednio ze wskazanej ścieżki.

Przykład ma siatkę `4 × 4` kafelków po `8 × 8` pikseli, więc wynik ma `32 × 32` piksele. `show` jest wyłączone; program nie otwiera okna podglądu.

Pliki wejściowe są dołączone do repozytorium: `examples/input.png` oraz `examples/tiles/{blue,green,red,yellow}.png`. Uruchomienie przykładu nie pobiera obrazów z sieci. Pięć małych PNG wygenerowano podczas tej pracy w Pythonie 3.14.8 przy użyciu modułów standardowych `struct` i `zlib`; skrypt generatora nie jest częścią repozytorium. `examples/output.png` jest wynikiem programu i jest ignorowany przez Git.

## 4. Stan sprawdzony lokalnie

W tej sesji polecenie `cmake --build build/local --target Mozaik image_paths_test` zakończyło się kodem 0, a `ctest --test-dir build/local -R ^image_paths_test$ --output-on-failure` pokazał `1/1 Passed`. Przykład uruchomiono z katalogu `examples` przez przekazanie ścieżki konfiguracji na standardowe wejście; program zgłosił `Program successfully run` i utworzył `examples/output.png`. Użyto istniejącej konfiguracji `build/local`, bez pobierania nowych zależności.

W obecnym środowisku wykryto CMake 4.4.4, Ninja 1.13.2, Clang 23.1.1 z LLVM-MinGW 20260908 oraz vcpkg z katalogu `C:\Users\DXD\Tools\cpp\vcpkg` na commicie `9e593bb18ea69cc5095e012465dcd675a822ed0d`. Są to obserwacje, a nie dowód, że wykonano podane wyżej polecenia pobierania.

Istniejący `build/local/CMakeCache.txt` wskazuje osobną konfigurację Ninja z `VCPKG_MANIFEST_MODE=OFF`, triplet `x64-mingw-static`, OpenCV i spdlog w instalacji vcpkg oraz JsonCpp w `C:\Users\DXD\Tools\cpp\jsoncpp-install`. To lokalny wariant awaryjny, **nie** wynik presetu manifestowego. Plik `build/local/smoke.png` istnieje, ale sam artefakt nie potwierdza dokładnej komendy, którą go utworzono. W razie problemu z manifestem przykład osobnej konfiguracji zależności znajduje się w `README.md`; poniżej podano także polecenia faktycznie wykonane przy lokalnej instalacji JsonCpp.

### Ręczna instalacja JsonCpp wykonana 23.10.2026

Automatyczne `vcpkg install jsoncpp:x64-mingw-static` zawiesiło się dwukrotnie podczas `detect_compiler`; druga próba użyła `VCPKG_FORCE_SYSTEM_BINARIES=1`. Procesy przerwano, więc instalacja manifestowa nie została potwierdzona. OpenCV 4.12.0 i spdlog 1.17.0 dla tripletu `x64-mingw-static` były już wcześniej w lokalnej instalacji vcpkg. Wykonano poniższe polecenia z katalogu repozytorium; klon JsonCpp 1.9.6 wskazywał commit `89e2973c754a9c02a49974d839779b151e95afd6`:

```powershell
git clone --depth 1 --branch 1.9.6 https://github.com/open-source-parsers/jsoncpp.git C:\Users\DXD\Tools\cpp\jsoncpp-src
cmake -S C:\Users\DXD\Tools\cpp\jsoncpp-src -B C:\Users\DXD\Tools\cpp\jsoncpp-build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DJSONCPP_WITH_TESTS=OFF -DJSONCPP_WITH_POST_BUILD_UNITTEST=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=C:\Users\DXD\Tools\cpp\jsoncpp-install
cmake --build C:\Users\DXD\Tools\cpp\jsoncpp-build --parallel 5
cmake --install C:\Users\DXD\Tools\cpp\jsoncpp-build
cmake -S . -B build\local -G Ninja -DCMAKE_TOOLCHAIN_FILE=C:\Users\DXD\Tools\cpp\vcpkg\scripts\buildsystems\vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-mingw-static -DVCPKG_MANIFEST_MODE=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -Djsoncpp_DIR=C:\Users\DXD\Tools\cpp\jsoncpp-install\lib\cmake\jsoncpp
cmake --build build\local --parallel 5
ctest --test-dir build\local --output-on-failure
```

Do powtórzenia tych poleceń na nowej maszynie trzeba wcześniej przygotować vcpkg z OpenCV i spdlog dla tego samego tripletu oraz umieścić Clang i Ninja na `PATH`. Samo sklonowanie JsonCpp nie zastępuje tych pakietów. Lokalny ignorowany przez Git `CMakeUserPresets.json` zapisuje `VCPKG_MANIFEST_MODE=OFF`, `jsoncpp_DIR=C:/Users/DXD/Tools/cpp/jsoncpp-install/lib/cmake/jsoncpp` i katalog `build/local`; wykonano też `cmake --preset local-manual-jsoncpp` oraz `cmake --build --preset local-manual-jsoncpp --parallel 5`.

Przed dodaniem argumentu CLI wykonano test z `build/local/smoke-config.json` zawierającym absolutne ścieżki do przykładowych obrazów: `$cfgPath | & .\build\local\Mozaik.exe`. Program zakończył się kodem 0 i utworzył `build/local/smoke.png` (447 bajtów, PNG `32 × 32`). Jest to zapis historycznego przebiegu; obecnie wystarcza prostsza komenda z sekcji 3.

W tej sesji sprawdzono także test równoległego czytania kafelków w istniejącej konfiguracji lokalnej:

```powershell
cmake --build build/local --target loader_order_test
ctest --test-dir build/local -R '^loader_order_test$' --output-on-failure
```

Oba polecenia zakończyły się kodem 0; CTest pokazał `1/1 Passed` w 1,67 s. Test używa 64 ścieżek, sprawdza zachowanie kolejności przy dwóch nieczytelnych plikach i przeniesienie wyjątku z wątku.

Po dołączeniu poprawki walidacji w tej sesji wykonano `cmake --build build/local --target Mozaik validation_test` oraz `ctest --test-dir build/local -R ^validation_test$ --output-on-failure`; oba polecenia zakończyły się kodem 0, a CTest pokazał `1/1 Passed`. Polecenie `build\local\Mozaik.exe examples\config.json` uruchomione z katalogu repozytorium również zakończyło się kodem 0 i zgłosiło `Program successfully run`.

Po poprawce proporcji wykonano:

```powershell
cmake --build build/local --target aspect_ratio_test
ctest --test-dir build/local -R '^aspect_ratio_test$' --output-on-failure
```

Kompilacja zakończyła się kodem 0, a test wynikiem `1/1 Passed`. Sprawdza obie drogi zachowania proporcji, zaokrąglenie połówki piksela, dolną granicę jednego piksela, świadome zniekształcenie i niedodatnie wymiary.

Po poprawce wyboru losowego kafelka wykonano:

```powershell
cmake --build build/local --target random_selection_test
ctest --test-dir build/local -R '^random_selection_test$' --output-on-failure
```

Oba polecenia zakończyły się kodem 0, a test wynikiem `1/1 Passed`. Sprawdza wybór tylko z najbliższych kandydatów, remisy, puste wejście, niepoprawny i nadmierny `rank` oraz ponowne użycie bufora.

Po zmianie renderowania wykonano `cmake --build build/local --target Mozaik mosaic_render_test validation_test`, `ctest --test-dir build/local --output-on-failure` oraz `build\local\Mozaik.exe examples\config.json` z katalogu repozytorium. Kompilacja i program zakończyły się kodem 0, a CTest pokazał `6/6 Passed`; program ponownie wygenerował `examples/output.png`. Test renderowania sprawdza rozmiar, układ kafelków, dobór według średniego koloru, filtrowanie i strategię losową z jednym kafelkiem. Nowy sposób filtrowania porównano piksel po pikselu z dotychczasową regułą dla kafelka o zróżnicowanych kolorach.

Dodatkowo dołączono `validation_resize_test`, który sprawdza zakończenie interaktywnych pytań po EOF. Polecenia `cmake --build build/local --target validation_resize_test` i `ctest --test-dir build/local -R '^validation_resize_test$' --output-on-failure` zakończyły się kodem 0, a test wynikiem `1/1 Passed`. Ponowne uruchomienie pełnego `ctest --test-dir build/local --output-on-failure` po dołączeniu testu pokazało `7/7 Passed`.

Po poprawce domyślnej konfiguracji i rozszerzeniu testów wykonano `cmake --build build/local --parallel 5` oraz `ctest --test-dir build/local --output-on-failure`: kompilacja zakończyła się kodem 0, a CTest pokazał `9/9 Passed`. `mosaic_e2e_test` uruchamia właściwy program z konfiguracją w ścieżce ze spacjami, sprawdza względne ścieżki danych, wybór kafelków, filtrację i zapis PNG `32 × 32`. `mozaik_config_lookup_test` sprawdza kolejność `./config.json`, `./examples/config.json` i komunikat o braku obu plików. Ponownie uruchomiono `.\build\local\Mozaik.exe .\examples\config.json` z katalogu repozytorium: kod wyjścia wyniósł 0, a `examples/output.png` miał 447 bajtów i wymiary `32 × 32`. Piaskownica zablokowała uruchomienie `clang-23`, dlatego udana kompilacja została uruchomiona po zatwierdzeniu wykonania poza nią.

## 5. IntelliSense w VS Code

Zainstaluj rozszerzenia VS Code **C/C++** (`ms-vscode.cpptools`) i **CMake Tools** (`ms-vscode.cmake-tools`). Otwórz bezpośrednio katalog `Mozaik_cpp` albo workspace, w którym jest on osobnym folderem. Ustawienia folderu `.vscode/settings.json` korzystają z `compile_commands.json` wygenerowanego przez CMake dla `build/local` lub `build/windows-llvm-mingw`; nie wpisują osobno katalogów nagłówków bibliotek. Preset projektu włącza `CMAKE_EXPORT_COMPILE_COMMANDS=ON`. Jeśli otwierasz katalog nadrzędny `cpp` jako jedyny folder, jego własny `.vscode/settings.json` musi wskazywać `Mozaik_cpp/build/.../compile_commands.json`, ponieważ VS Code nie czyta wtedy ustawień z podkatalogu repozytorium. Przed pierwszą konfiguracją przygotuj `PATH` i `VCPKG_ROOT` zgodnie z sekcją 1, a następnie skonfiguruj i zbuduj wybrany preset. Dla lokalnego wariantu z osobnym JsonCpp polecenia do samodzielnego wykonania są następujące:

```powershell
cmake --preset local-manual-jsoncpp
cmake --build --preset local-manual-jsoncpp --parallel 5
Test-Path .\build\local\compile_commands.json
code .
```

Preset `local-manual-jsoncpp` istnieje tylko w ignorowanym przez Git `CMakeUserPresets.json` i wymaga lokalnych pakietów opisanych w sekcji 4. Na nowym komputerze użyj presetu `windows-llvm-mingw` z sekcji 2; jego baza poleceń będzie w `build/windows-llvm-mingw`. W ustawieniach folderu automatyczne konfigurowanie po otwarciu jest wyłączone, aby edytor nie uruchamiał samodzielnie długiej instalacji zależności z manifestu. W CMake Tools można świadomie wybrać preset i uruchomić **CMake: Configure**. Jeśli otwarte już okno nie odświeży diagnostyki po zmianie ustawień, użyj **Developer: Reload Window**.

Jeśli konfigurujesz projekt ręcznym poleceniem `cmake -S . -B build/local ...` zamiast presetu, dodaj opcję `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`; bez niej baza dla edytora nie powstanie.

Wykonano tutaj `cmake --preset local-manual-jsoncpp`, `cmake --build --preset local-manual-jsoncpp --parallel 5` i `ctest --test-dir build/local --output-on-failure`. CMake utworzył `build/local/compile_commands.json` z 13 wpisami; wpis `main.cpp` wskazuje Clang oraz katalogi OpenCV i vcpkg, a wpis `core.cpp` dodatkowo katalog JsonCpp. Kompilacja zakończyła się kodem 0, testy wynikiem `9/9 Passed`. Sprawdzono poprawność JSON ustawień edytora. Z lokalnego pliku ustawień użytkownika VS Code (`C:\Users\DXD\AppData\Roaming\Code\User\settings.json`) usunięto `C_Cpp.default.systemIncludePath`, które wskazywało `C:\msys64\ucrt64\bin\**` i nadpisywało wykryte przez kompilator nagłówki systemowe. W katalogu nadrzędnym ustawiono dodatkowo `../.vscode/settings.json` z bazą komend i pełną ścieżką do Clang, aby rozwiązanie działało także po otwarciu samego `cpp`. Oba pliki poza repozytorium są lokalne i nie zostaną wysłane na GitHub. Ustawienia repozytorium wybierają `clang++` i tryb `windows-clang-x64`, więc pozostałe globalne ustawienia GCC nie są używane dla tego projektu. Nie odczytywano bezpośrednio diagnostyki z otwartego okna VS Code; polecenie `code .` i odświeżenie okna są instrukcjami, a nie wykonanymi tu krokami.

## 6. Jak zachować zasady dokumentowania w innych projektach

W tym środowisku utworzono `C:\Users\DXD\.codex\AGENTS.md` z regułą prowadzenia `manual.md` i `mods.md` w każdym projekcie. Kopia w tym repozytorium to `AGENTS.md`. Zgodnie z [instrukcją Codex](https://learn.chatgpt.com/docs/agent-configuration/agents-md) plik w katalogu domowym Codex jest globalny, a plik w repozytorium działa lokalnie; Codex odczytuje je przy rozpoczęciu nowej sesji. Lokalizację domową można zmienić przez `CODEX_HOME`.

Na innym komputerze skopiuj regułę z tego repozytorium do katalogu domowego Codex. Gdy docelowy plik już istnieje, otwórz go i dopisz regułę ręcznie, aby nie zastąpić innych preferencji:

```powershell
$codexHomeDir = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $env:USERPROFILE '.codex' }
New-Item -ItemType Directory -Force -Path $codexHomeDir | Out-Null
$globalRules = Join-Path $codexHomeDir 'AGENTS.md'
if (Test-Path -LiteralPath $globalRules) {
    notepad $globalRules
} else {
    Copy-Item -LiteralPath .\AGENTS.md -Destination $globalRules
}
Get-Content -LiteralPath $globalRules -Encoding utf8
```

W bieżącej sesji potwierdzono utworzenie i treść pliku globalnego oraz kopii w repozytorium. Nie uruchamiano osobnej nowej sesji Codex do sprawdzenia automatycznego odczytu reguły.
