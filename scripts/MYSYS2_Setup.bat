@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM Prüfen auf Test-Modus
set TEST_MODE=0
if "%1"=="/test" (
    set TEST_MODE=1
    echo ===== MSYS2 Aktualisierung und Paketinstallation [TEST-MODUS] =====
    echo Es werden keine Änderungen vorgenommen.
) else (
    echo ===== MSYS2 Aktualisierung und Paketinstallation =====
)

REM Abhängigkeiten als Parameter
set DEPENDENCIES=%2
if "%DEPENDENCIES%"=="" (
    set DEPENDENCIES=pkg-config,mingw-w64-x86_64-poppler
)

set MSYS2_PATH=C:\msys64

if not exist "%MSYS2_PATH%\usr\bin\bash.exe" (
    echo MSYS2 ist nicht installiert oder der Pfad ist falsch.
    echo Bitte installieren Sie MSYS2 oder korrigieren Sie den Pfad.
    goto :END
)

echo Dynamische Abhängigkeiten: %DEPENDENCIES%

REM Prüfe jede Abhängigkeit
for %%d in (%DEPENDENCIES:,= %) do (
    echo Überprüfe %%d...
    "%MSYS2_PATH%\usr\bin\bash.exe" -lc "pacman -Q %%d 2>/dev/null"
    if !ERRORLEVEL! neq 0 (
        echo %%d ist nicht installiert, wird installiert.
        set "INSTALL_PACKAGES=!INSTALL_PACKAGES! %%d"
    ) else (
        echo %%d ist bereits installiert.
    )
)

echo.
set /p UPDATE_MSYS2=Möchten Sie MSYS2 aktualisieren? (y/n): 
if /I "%UPDATE_MSYS2%"=="y" (
    if !TEST_MODE!==1 (
        echo [TEST-MODUS] Würde MSYS2 aktualisieren...
    ) else (
        echo Aktualisiere MSYS2...
        "%MSYS2_PATH%\usr\bin\bash.exe" -lc "pacman -Syu --noconfirm"
        echo MSYS2 wurde aktualisiert.
    )
) else (
    echo MSYS2 Aktualisierung übersprungen.
)

if defined INSTALL_PACKAGES (
    echo.
    if !TEST_MODE!==1 (
        echo [TEST-MODUS] Würde folgende Pakete installieren:!INSTALL_PACKAGES!
    ) else (
        echo Installiere Pakete:!INSTALL_PACKAGES!
        "%MSYS2_PATH%\usr\bin\bash.exe" -lc "pacman -S --noconfirm!INSTALL_PACKAGES!"
        echo Installation abgeschlossen.
    )
)

:END
echo.
if !TEST_MODE!==1 (
    echo MSYS2 Setup abgeschlossen [TEST-MODUS].
) else (
    echo MSYS2 Setup abgeschlossen.
)
pause
exit /b 0
