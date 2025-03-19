# Parameter fuer Testmodus und Plattformwechsel
param(
    [switch]$Test,
    [ValidateSet("Windows", "Linux", "MacOS")]
    [string]$Platform = "Windows"
)

# Je nach Plattform andere Aktionen ausführen
switch ($Platform) {
    "Windows" {
        # Wenn im Testmodus, Hinweis anzeigen
        if ($Test) {
            Write-Host "=========================================================" -ForegroundColor Yellow
            Write-Host "                      TEST-MODUS                         " -ForegroundColor Yellow
            Write-Host "  Das Skript wird nur simuliert, keine Aenderungen       " -ForegroundColor Yellow
            Write-Host "  werden am System vorgenommen.                          " -ForegroundColor Yellow
            Write-Host "=========================================================" -ForegroundColor Yellow
        }

        # Starten der Protokollierung
        $logPath = "$env:TEMP\DevEnv_Setup_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
        Start-Transcript -Path $logPath

        # Anzeige des Haupttitels
        Clear-Host
        Write-Host "=========================================================" -ForegroundColor Cyan
        Write-Host "               ENTWICKLUNGSUMGEBUNG SETUP                " -ForegroundColor Cyan
        Write-Host "         Visual Studio Build Tools & MSYS2 Installer     " -ForegroundColor Cyan
        Write-Host "=========================================================" -ForegroundColor Cyan
        Write-Host "Script wird ausgefuehrt. Protokollierung in: $logPath" -ForegroundColor Green

        # Funktionen definieren
        function Get-VSInstallationPath {
            $vsPaths = @(
                "C:\BuildTools\Common7\Tools",
                "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE",
                "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\Common7\IDE",
                "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE",
                "C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional\Common7\IDE",
                "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE",
                "C:\Program Files (x86)\Microsoft Visual Studio\2022\Enterprise\Common7\IDE"
            )
            
            foreach ($path in $vsPaths) {
                if (Test-Path $path) {
                    return $path
                }
            }
            
            return $null
        }

        function Get-LatestMSYS2Installer {
            Write-Host "Ermittle neueste MSYS2-Version..." -ForegroundColor Yellow
            
            try {
                # GitHub API abfragen, um die neuesten Releases zu bekommen
                $apiUrl = "https://api.github.com/repos/msys2/msys2-installer/releases/latest"
                $release = Invoke-RestMethod -Uri $apiUrl -Headers @{
                    "Accept" = "application/vnd.github.v3+json"
                }
                
                # x64 oder x86 basierend auf der Systemarchitektur wählen
                $is64bit = [Environment]::Is64BitOperatingSystem
                $pattern = $is64bit ? "msys2-x86_64-*.exe" : "msys2-i686-*.exe"
                
                # Passenden Asset finden
                $asset = $release.assets | Where-Object { $_.name -like $pattern } | Select-Object -First 1
                
                if ($asset) {
                    Write-Host "Neueste MSYS2-Version gefunden: $($asset.name)" -ForegroundColor Green
                    return @{
                        Url = $asset.browser_download_url
                        Version = $release.tag_name
                        FileName = $asset.name
                    }
                }
            } catch {
                Write-Host "Fehler beim Ermitteln der neuesten MSYS2-Version: $_" -ForegroundColor Red
            }
            
            # Fallback zur statischen URL wenn API-Abfrage fehlschlägt
            Write-Host "Verwende Standard-MSYS2-Version als Fallback" -ForegroundColor Yellow
            return @{
                Url = "https://github.com/msys2/msys2-installer/releases/download/2023-05-26/msys2-x86_64-20230526.exe"
                Version = "2023-05-26"
                FileName = "msys2-x86_64-20230526.exe"
            }
        }

        function Get-VisualStudioInstallerInfo {
            Write-Host "Ermittle passende Visual Studio-Version..." -ForegroundColor Yellow
            
            # Ermittle Windows-Version
            $osInfo = Get-CimInstance Win32_OperatingSystem
            $winVersion = [Version]$osInfo.Version
            
            # Prüfe Systemanforderungen für verschiedene VS-Versionen
            $vs2022MinVersion = [Version]"10.0.17763.0"  # Windows 10 Version 1809
            $vs2019MinVersion = [Version]"10.0.16299.0"  # Windows 10 Fall Creators Update
            
            # Prozessorarchitektur prüfen
            $is64bit = [Environment]::Is64BitOperatingSystem
            
            # Bestimme die beste VS-Version basierend auf Systemanforderungen
            if ($winVersion -ge $vs2022MinVersion -and $is64bit) {
                Write-Host "System unterstützt Visual Studio 2022" -ForegroundColor Green
                return @{
                    Url = "https://aka.ms/vs/17/release/vs_buildtools.exe"
                    Version = "2022"
                    RecommendedComponents = @(
                        "Microsoft.VisualStudio.Workload.VCTools", 
                        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", 
                        "Microsoft.VisualStudio.Component.Windows10SDK"
                    )
                }
            } elseif ($winVersion -ge $vs2019MinVersion) {
                Write-Host "System unterstützt Visual Studio 2019" -ForegroundColor Green
                return @{
                    Url = "https://aka.ms/vs/16/release/vs_buildtools.exe"
                    Version = "2019"
                    RecommendedComponents = @(
                        "Microsoft.VisualStudio.Workload.VCTools", 
                        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", 
                        "Microsoft.VisualStudio.Component.Windows10SDK"
                    )
                }
            } else {
                Write-Host "System erfüllt minimale Anforderungen für Visual Studio 2019 nicht" -ForegroundColor Yellow
                Write-Host "Verwende Visual Studio 2017 als Fallback" -ForegroundColor Yellow
                return @{
                    Url = "https://aka.ms/vs/15/release/vs_buildtools.exe"
                    Version = "2017"
                    RecommendedComponents = @(
                        "Microsoft.VisualStudio.Workload.VCTools", 
                        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
                    )
                }
            }
        }

        # Schritt 1: Visual Studio 2022 pruefen (Build Tools oder Vollversion)
        Write-Host "`nSchritt 1: Ueberpruefe Visual Studio..." -ForegroundColor Cyan

        # Visual Studio Build Tools herunterladen und installieren
        $vsInstallerUrl = "https://aka.ms/vs/17/release/vs_buildtools.exe"
        $vsInstallerPath = "$env:TEMP\vs_buildtools.exe"
        $vsWorkloadParams = "--quiet --wait --norestart --nocache --installPath `"C:\BuildTools`" --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows10SDK"

        Write-Host "Lade Visual Studio Build Tools herunter..." -ForegroundColor Yellow
        Invoke-WebRequest -Uri $vsInstallerUrl -OutFile $vsInstallerPath

        Write-Host "Installiere Visual Studio Build Tools..." -ForegroundColor Yellow
        Start-Process -FilePath $vsInstallerPath -ArgumentList $vsWorkloadParams -Wait -NoNewWindow

        # Warte kurz und prüfe, ob die Installation erfolgreich war
        Start-Sleep -Seconds 5
        $vsPath = Get-VSInstallationPath
        $vsInstalled = $null -ne $vsPath

        if ($vsInstalled) {
            Write-Host "Visual Studio Build Tools wurden erfolgreich installiert." -ForegroundColor Green
        } else {
            Write-Host "Installation der Visual Studio Build Tools konnte nicht verifiziert werden. Bitte überprüfen Sie manuell." -ForegroundColor Yellow
        }

        # Schritt 2: MSYS2 Installation pruefen
        Write-Host "`nSchritt 2: Ueberpruefe MSYS2..." -ForegroundColor Cyan

        $msys2Path = "C:\msys64"
        if (Test-Path "$msys2Path\usr\bin\bash.exe") {
            Write-Host "MSYS2 ist bereits installiert." -ForegroundColor Green
        } 
        else {
            Write-Host "MSYS2 ist nicht installiert." -ForegroundColor Yellow
            if (-not $Test) {
                Write-Host "MSYS2 wird heruntergeladen und installiert..." -ForegroundColor Yellow
                # MSYS2 herunterladen und installieren
                $msys2Installer = Get-LatestMSYS2Installer
                $msys2InstallerPath = "$env:TEMP\$($msys2Installer.FileName)"
                $msys2InstallParams = "/InstallDir=C:\msys64 /Quiet"

                Write-Host "Lade MSYS2 Installer herunter..." -ForegroundColor Yellow
                Invoke-WebRequest -Uri $msys2Installer.Url -OutFile $msys2InstallerPath

                Write-Host "Installiere MSYS2..." -ForegroundColor Yellow
                Start-Process -FilePath $msys2InstallerPath -ArgumentList $msys2InstallParams -Wait -NoNewWindow

                # Initialisiere MSYS2 (erster Start)
                if (Test-Path "C:\msys64\usr\bin\bash.exe") {
                    Write-Host "Initialisiere MSYS2..." -ForegroundColor Yellow
                    Start-Process -FilePath "C:\msys64\usr\bin\bash.exe" -ArgumentList "-lc 'exit'" -Wait -NoNewWindow
                    Start-Sleep -Seconds 3
                    Write-Host "MSYS2 wurde erfolgreich installiert und initialisiert." -ForegroundColor Green
                } else {
                    Write-Host "MSYS2 Installation konnte nicht verifiziert werden. Bitte überprüfen Sie manuell." -ForegroundColor Yellow
                }
            } else {
                Write-Host "TEST-MODUS: MSYS2 würde heruntergeladen und installiert werden" -ForegroundColor Yellow
            }
        }

        # Schritt 3: MSYS2 aktualisieren und Pakete installieren
        Write-Host "`nSchritt 3: MSYS2 aktualisieren und Pakete installieren..." -ForegroundColor Cyan

        # Pfad zur Batch-Datei - mit Prüfung mehrerer möglicher Orte
        $batName = "MYSYS2_Setup.bat"
        $possiblePaths = @(
            "$PSScriptRoot\$batName",                  # Im gleichen Verzeichnis wie das PS1-Skript
            "$PSScriptRoot\MSYS2_Setup.bat",           # Alternative Schreibweise
            ".\$batName",                              # Im aktuellen Verzeichnis
            ".\MSYS2_Setup.bat",                       # Alternative Schreibweise
            "$PWD\$batName",                           # Absoluter Pfad zum aktuellen Verzeichnis
            "$PWD\MSYS2_Setup.bat"                     # Alternative Schreibweise
        )

        $batScriptPath = $null
        foreach ($path in $possiblePaths) {
            if (Test-Path $path) {
                $batScriptPath = $path
                Write-Host "MYSYS2_Setup.bat gefunden in: $batScriptPath" -ForegroundColor Green
                break
            }
        }

        # Dynamische Abhängigkeiten aus requirements.txt ermitteln
        Write-Host "`nSchritt 4: Analysiere Python-Abhängigkeiten..." -ForegroundColor Cyan

        $reqFilePaths = @(
            "$PSScriptRoot\..\requirements.txt",
            "$PSScriptRoot\..\localgpt_vision_django\requirements.txt",
            ".\requirements.txt"
        )

        $requirementFile = $null
        foreach ($path in $reqFilePaths) {
            if (Test-Path $path) {
                $requirementFile = $path
                Write-Host "requirements.txt gefunden in: $requirementFile" -ForegroundColor Green
                break
            }
        }

        $msys2Dependencies = @{}
        $vsComponents = @{}

        if ($requirementFile) {
            $requirements = Get-Content $requirementFile
            
            # Laden des Mappings aus der externen JSON-Datei
            $mappingFilePaths = @(
                "$PSScriptRoot\package_mapping.json",
                ".\package_mapping.json"
            )
            
            $mappingFile = $null
            foreach ($path in $mappingFilePaths) {
                if (Test-Path $path) {
                    $mappingFile = $path
                    Write-Host "Paket-Mapping gefunden in: $mappingFile" -ForegroundColor Green
                    break
                }
            }
            
            # Laden des Paket-Mappings
            if ($mappingFile) {
                try {
                    # Konvertiere die JSON-Datei in ein PowerShell-Objekt
                    $mappingContent = Get-Content $mappingFile -Raw
                    $mappingJson = ConvertFrom-Json $mappingContent
                    
                    # Extrahiere die verschiedenen Mappings
                    $packageMappings = $mappingJson.packages
                    $vsComponentMappings = $mappingJson.vs_components
                    
                    Write-Host "Paket-Mapping erfolgreich geladen." -ForegroundColor Green
                } catch {
                    Write-Host "Fehler beim Laden des Paket-Mappings: $_" -ForegroundColor Red
                    # Fallback auf minimale Mappings
                    $packageMappings = @{
                        "pkgconfig" = @("pkg-config")
                        "python-poppler" = @("mingw-w64-x86_64-poppler")
                    }
                    $vsComponentMappings = @{
                        "torch" = @("Microsoft.VisualStudio.Component.VC.Tools.x86.x64")
                    }
                    Write-Host "Verwende minimales Fallback-Mapping." -ForegroundColor Yellow
                }
            } else {
                Write-Host "Keine package_mapping.json gefunden. Verwende minimales Mapping." -ForegroundColor Yellow
                # Minimale Fallback-Mappings
                $packageMappings = @{
                    "pkgconfig" = @("pkg-config")
                    "python-poppler" = @("mingw-w64-x86_64-poppler")
                }
                $vsComponentMappings = @{
                    "torch" = @("Microsoft.VisualStudio.Component.VC.Tools.x86.x64")
                }
            }
            
            # Analyse der requirements.txt
            foreach ($req in $requirements) {
                # Ignoriere Kommentare und leere Zeilen
                $req = $req.Trim()
                if ($req -eq "" -or $req.StartsWith("#")) { continue }
                
                # Extrahiere Paketnamen (ohne Version)
                if ($req -match "^([a-zA-Z0-9\._-]+)") {
                    $packageName = $matches[1]
                    
                    # Prüfe auf MSYS2-Abhängigkeiten
                    if ($packageMappings.PSObject.Properties[$packageName]) {
                        $deps = $packageMappings.PSObject.Properties[$packageName].Value
                        foreach ($dep in $deps) {
                            if (-not $msys2Dependencies.ContainsKey($dep)) {
                                $msys2Dependencies[$dep] = $packageName
                            }
                        }
                    }
                    
                    # Prüfe auf Visual Studio-Komponenten
                    if ($vsComponentMappings.PSObject.Properties[$packageName]) {
                        $comps = $vsComponentMappings.PSObject.Properties[$packageName].Value
                        foreach ($comp in $comps) {
                            if (-not $vsComponents.ContainsKey($comp)) {
                                $vsComponents[$comp] = $packageName
                            }
                        }
                    }
                }
            }
            
            # Zeige erkannte MSYS2-Abhängigkeiten an
            if ($msys2Dependencies.Count -gt 0) {
                Write-Host "Erkannte MSYS2-Abhängigkeiten:" -ForegroundColor Cyan
                foreach ($dep in $msys2Dependencies.Keys) {
                    Write-Host "  - $dep (benötigt für $($msys2Dependencies[$dep]))" -ForegroundColor Yellow
                }
            } else {
                Write-Host "Keine bekannten MSYS2-Abhängigkeiten in requirements.txt gefunden." -ForegroundColor Yellow
            }
            
            # Zeige erkannte Visual Studio-Komponenten an
            if ($vsComponents.Count -gt 0) {
                Write-Host "Erkannte Visual Studio-Komponenten:" -ForegroundColor Cyan
                foreach ($comp in $vsComponents.Keys) {
                    Write-Host "  - $comp (benötigt für $($vsComponents[$comp]))" -ForegroundColor Yellow
                }
                
                # Wenn Visual Studio installiert ist und Komponenten fehlen
                if ($vsInstalled -and -not $Test) {
                    # Fehlende Visual Studio-Komponenten installieren
                    if ($vsComponents.Count -gt 0) {
                        # Prüfe, ob wir die Build Tools oder die vollständige VS-Version haben
                        $vsInstallerPath = ""
                        $vsInstallationPath = ""
                        
                        if (Test-Path "C:\BuildTools") {
                            $vsInstallerPath = "C:\BuildTools\Common7\IDE\CommonExtensions\Microsoft\VSI\bin\vs_installer.exe"
                            if (-not (Test-Path $vsInstallerPath)) {
                                $vsInstallerPath = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe"
                            }
                            $vsInstallationPath = "C:\BuildTools"
                        } else {
                            $vsInstallerPath = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe"
                            # Finde den tatsächlichen Installationspfad
                            $possiblePaths = @(
                                "C:\Program Files\Microsoft Visual Studio\2022\Community",
                                "C:\Program Files\Microsoft Visual Studio\2022\Professional",
                                "C:\Program Files\Microsoft Visual Studio\2022\Enterprise",
                                "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community",
                                "C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional",
                                "C:\Program Files (x86)\Microsoft Visual Studio\2022\Enterprise"
                            )
                            
                            foreach ($path in $possiblePaths) {
                                if (Test-Path $path) {
                                    $vsInstallationPath = $path
                                    break
                                }
                            }
                        }
                        
                        if (Test-Path $vsInstallerPath) {
                            # Bereite die Liste der zu installierenden Komponenten vor
                            $componentsList = $vsComponents.Keys -join " --add "
                            
                            if (-not $Test) {
                                Write-Host "Installiere fehlende Visual Studio-Komponenten..." -ForegroundColor Yellow
                                $modifyArgs = "modify --installPath `"$vsInstallationPath`" --passive --norestart --add $componentsList"
                                Start-Process -FilePath $vsInstallerPath -ArgumentList $modifyArgs -Wait -NoNewWindow
                                Write-Host "Visual Studio-Komponenten wurden installiert." -ForegroundColor Green
                            } else {
                                Write-Host "TEST-MODUS: Würde folgende VS-Komponenten installieren:" -ForegroundColor Yellow
                                $vsComponents.Keys | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
                            }
                        } else {
                            Write-Host "Visual Studio Installer nicht gefunden. Komponenten müssen manuell installiert werden." -ForegroundColor Red
                            Write-Host "Benötigte Komponenten:" -ForegroundColor Yellow
                            $vsComponents.Keys | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
                        }
                    }
                } elseif ($Test) {
                    Write-Host "TEST-MODUS: Folgende Visual Studio-Komponenten würden installiert werden:" -ForegroundColor Yellow
                    $vsComponents.Keys | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
                }
            } else {
                Write-Host "Keine bekannten Visual Studio-Komponenten in requirements.txt gefunden." -ForegroundColor Yellow
            }
            
            # Übergebe die MSYS2-Abhängigkeiten an die Batch-Datei
            $depsParam = [string]::Join(",", $msys2Dependencies.Keys)
        } else {
            Write-Host "Keine requirements.txt gefunden. Verwende Standard-Abhängigkeiten." -ForegroundColor Yellow
            $depsParam = "pkg-config,mingw-w64-x86_64-poppler"
        }

        if (Test-Path "$msys2Path\usr\bin\bash.exe") {
            if ($batScriptPath) {
                Write-Host "Fuehre $batName mit ermittelten Abhängigkeiten aus..." -ForegroundColor Yellow
                
                # Übergeben der Parameter an die Batch-Datei
                if ($Test) {
                    & $batScriptPath "/test" $depsParam
                    Write-Host "$batName wurde im TEST-MODUS ausgefuehrt." -ForegroundColor Yellow
                } else {
                    & $batScriptPath "normal" $depsParam
                    Write-Host "$batName wurde ausgefuehrt." -ForegroundColor Green
                }
            } else {
                Write-Host "Fehler: $batName wurde nicht gefunden. Bitte stellen Sie sicher, dass sie im gleichen Verzeichnis wie dieses Skript existiert." -ForegroundColor Red
            }
        } else {
            Write-Host "MSYS2 ist nicht installiert. Ueberspringe Ausfuehrung von $batName." -ForegroundColor Yellow
        }

        # Abschluss
        Write-Host "`nInstallation abgeschlossen" -ForegroundColor Cyan

        Write-Host "`n=========================================================" -ForegroundColor Green
        Write-Host "                 INSTALLATION ABGESCHLOSSEN               " -ForegroundColor Green
        Write-Host "=========================================================" -ForegroundColor Green

        Write-Host "`nZusammenfassung:" -ForegroundColor Cyan
        Write-Host "- Visual Studio: " -NoNewline
        if ($vsInstalled) { 
            Write-Host "Installiert" -ForegroundColor Green 
        } else { 
            Write-Host "Nicht installiert" -ForegroundColor Red 
        }

        Write-Host "- MSYS2: " -NoNewline
        if (Test-Path "$msys2Path\usr\bin\bash.exe") { 
            Write-Host "Installiert" -ForegroundColor Green 
        } else { 
            Write-Host "Nicht installiert" -ForegroundColor Red 
        }

        Write-Host "- VS-Komponenten: " -NoNewline
        if ($vsComponents.Count -gt 0) {
            Write-Host "$($vsComponents.Count) erkannt" -ForegroundColor Green
        } else { 
            Write-Host "Keine erkannt" -ForegroundColor Yellow 
        }

        Write-Host "- MSYS2-Pakete: " -NoNewline
        if ($msys2Dependencies.Count -gt 0) {
            Write-Host "$($msys2Dependencies.Count) erkannt" -ForegroundColor Green
        } else { 
            Write-Host "Keine erkannt" -ForegroundColor Yellow 
        }

        # Beenden der Protokollierung
        Stop-Transcript
        Write-Host "`nProtokoll gespeichert in: $logPath" -ForegroundColor Cyan

        # Warten auf Benutzereingabe
        Write-Host "`nDruecken Sie eine beliebige Taste, um das Script zu beenden..."
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    }
    "Linux" {
        # Rufe Linux-Installationsroutinen auf
    }
    "MacOS" {
        # Rufe macOS-Installationsroutinen auf
    }
}
