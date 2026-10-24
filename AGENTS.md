# Stałe zasady dokumentowania projektów

Przy pracy nad dowolnym projektem, szczególnie C++, prowadź w katalogu głównym jego repozytorium dwa pliki:

- `manual.md`: zapisuj konkretne polecenia przygotowania środowiska, pobierania danych i zależności, konfiguracji narzędzi, budowania, uruchamiania i testowania. Podawaj wymagane wersje, zmienne środowiskowe i katalogi. Wyraźnie odróżniaj polecenia rzeczywiście wykonane od instrukcji przygotowanych do samodzielnego wykonania.
- `mods.md`: uzasadniaj każdą zmianę kodu lub konfiguracji. Wyjaśniaj problem, wybrane rozwiązanie, wpływ na działanie, kompromisy i sposób sprawdzenia. Decyzje takie jak użycie OpenMP albo `std::jthread` opisuj zgodnie z faktycznie zmienionym kodem; nie przypisuj zmian, których nie było.

Aktualizuj te pliki razem ze zmianami projektu. Pisz tak, aby właściciel mógł powtórzyć kroki i zrozumieć decyzje.
