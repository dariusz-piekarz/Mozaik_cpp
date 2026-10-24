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

`CMakeLists.txt` rejestruje w CTest siedem testów: `image_paths_test`, `loader_order_test`, `validation_test`, `aspect_ratio_test`, `random_selection_test`, `mosaic_render_test` i `validation_resize_test`.

## 3. Przykład uruchomienia i dane

Po poprawce walidacji program przyjmuje ścieżkę JSON jako opcjonalny argument. Ścieżki obrazów i wyniku w JSON są liczone względem katalogu tego pliku. Z katalogu głównego repozytorium uruchom:

```powershell
.\build\windows-llvm-mingw\Mozaik.exe .\examples\config.json
Get-Item .\examples\output.png
```

Bez argumentu program zapyta o ścieżkę konfiguracji. Wpisanie jej jako całej linii pozwala używać nazw ze spacjami. Błędny JSON, nieczytelny obraz i błąd zapisu kończą program z komunikatem i kodem niezerowym.

Przykład ma siatkę `4 × 4` kafelków po `8 × 8` pikseli, więc wynik ma `32 × 32` piksele. `show` jest wyłączone; program nie otwiera okna podglądu.

Pliki wejściowe są dołączone do repozytorium: `examples/input.png` oraz `examples/tiles/{blue,green,red,yellow}.png`. Uruchomienie przykładu nie pobiera obrazów z sieci. Historia pochodzenia tych pięciu PNG, sposób ich wytworzenia i ich osobna licencja nie są zapisane w repozytorium. `examples/output.png` jest wynikiem programu i jest ignorowany przez Git.

## 4. Stan sprawdzony lokalnie

W tej sesji polecenie `cmake --build build/local --target Mozaik image_paths_test` zakończyło się kodem 0, a `ctest --test-dir build/local -R ^image_paths_test$ --output-on-failure` pokazał `1/1 Passed`. Przykład uruchomiono z katalogu `examples` przez przekazanie ścieżki konfiguracji na standardowe wejście; program zgłosił `Program successfully run` i utworzył `examples/output.png`. Użyto istniejącej konfiguracji `build/local`, bez pobierania nowych zależności.

W obecnym środowisku wykryto CMake 4.4.4, Ninja 1.13.2, Clang 23.1.1 z LLVM-MinGW 20260908 oraz vcpkg z katalogu `C:\Users\DXD\Tools\cpp\vcpkg` na commicie `9e593bb18ea69cc5095e012465dcd675a822ed0d`. Są to obserwacje, a nie dowód, że wykonano podane wyżej polecenia pobierania.

Istniejący `build/local/CMakeCache.txt` wskazuje osobną konfigurację Ninja z `VCPKG_MANIFEST_MODE=OFF`, triplet `x64-mingw-static`, OpenCV i spdlog w instalacji vcpkg oraz JsonCpp w `C:\Users\DXD\Tools\cpp\jsoncpp-install`. To lokalny wariant awaryjny, **nie** wynik presetu manifestowego. Plik `build/local/smoke.png` istnieje, ale sam artefakt nie potwierdza dokładnej komendy, którą go utworzono. W razie problemu z manifestem przykład osobnej konfiguracji zależności znajduje się w `README.md`; nie ma tu kompletnego, sprawdzonego przepisu na zbudowanie osobnej instalacji JsonCpp.

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

## 5. Jak zachować zasady dokumentowania w innych projektach

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
