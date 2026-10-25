# Pomysły na dalszy rozwój

Lista jest uporządkowana według proponowanego priorytetu. Wdrożone zadania pozostają na liście jako zapis kierunku rozwoju.

- [x] **Automatyczne CI.** Przy każdym `push` i `pull_request` konfigurować projekt przez CMake, kompilować go i uruchamiać pełny CTest. Workflow można także uruchomić ręcznie. Implementacja: `.github/workflows/ci.yml`.
- [x] **Nieinteraktywny wybór proporcji.** Konfiguracja przyjmuje `interactive`, `preserve_width`, `preserve_height` albo `stretch`; trzy ostatnie tryby nie oczekują na odpowiedź z terminala.
- [ ] **Kontrolowane ziarno losowania.** Opcjonalne `random_seed` ma umożliwić odtworzenie identycznej mozaiki; brak wartości powinien zachować niedeterministyczne losowanie.
- [ ] **Benchmarki czasu i pamięci.** Osobno mierzyć wczytywanie kafelków, wyszukiwanie najbliższego kafelka oraz renderowanie. Na podstawie pomiarów zdecydować, czy warto równoleglić zapis do rozłącznych fragmentów obrazu.
- [ ] **Cache metadanych kafelków.** Zapisywać średnie kolory razem ze ścieżką, rozmiarem i czasem modyfikacji pliku, aby nie dekodować ponownie niezmienionego katalogu obrazów.
- [ ] **Pełniejsze CLI.** Dodać opcje takie jak `--config`, `--output`, `--seed`, `--threads`, `--no-preview` i `--aspect`, z jasno określonym pierwszeństwem względem JSON.
- [ ] **Opcjonalne przeszukiwanie podkatalogów.** Pozwolić używać bibliotek kafelków zagnieżdżonych w wielu katalogach, zachowując obecny tryb płaski jako domyślny.
- [ ] **Kontrola lokalnych powtórzeń.** Dodać karę za ponowne użycie tego samego kafelka w sąsiedztwie albo minimalną odległość między jego wystąpieniami.
