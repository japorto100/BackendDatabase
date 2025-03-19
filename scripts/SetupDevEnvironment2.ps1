# Plattformübergreifendes Entwicklungsumgebungs-Setup
# Kann auf Windows, Linux (mit PowerShell Core) und macOS ausgeführt werden

param(
    [switch]$Test
)

# Bestimme das Betriebssystem
$global:IsWindows = $PSVersionTable.PSVersion.Major -ge 5 -or $PSVersionTable.Platform -eq 'Win32NT' -or $env:OS -match 'Windows'
$global:IsLinux = $PSVersionTable.Platform -eq 'Unix' -and $PSVersionTable.OS -match 'Linux'
$global:IsMacOS = $PSVersionTable.Platform -eq 'Unix' -and $PSVersionTable.OS -match 'Darwin'

# Konfiguration für verschiedene Plattformen
$config = @{
    Windows = @{
        Compilers = @("Visual Studio Build Tools", "MSYS2/MinGW")
        PackageManagers = @("pip", "pacman")
        BasePaths = @{
            VS = "C:\BuildTools"
            MSYS2 = "C:\msys64"
        }
    }
    Linux = @{
        Compilers = @("gcc/g++", "clang")
        PackageManagers = @("apt", "dnf", "yum", "zypper", "pacman")
        BasePaths = @{
            Compilers = "/usr/bin"
            Libraries = "/usr/lib"
        }
    }
    MacOS = @{
        Compilers = @("Xcode Command Line Tools", "clang")
        PackageManagers = @("brew", "port")
        BasePaths = @{
            Brew = "/usr/local/bin"
            XCode = "/Applications/Xcode.app"
        }
    }
}

# Zeige Systeminformationen
Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "             ENTWICKLUNGSUMGEBUNG SETUP                  " -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "Betriebssystem:        $($PSVersionTable.OS)" -ForegroundColor Yellow
Write-Host "PowerShell Version:    $($PSVersionTable.PSVersion)" -ForegroundColor Yellow
Write-Host "Prozessorarchitektur:  $((Get-CimInstance Win32_Processor).AddressWidth) Bit" -ForegroundColor Yellow
Write-Host "Test-Modus:            $($Test)" -ForegroundColor Yellow
Write-Host "=========================================================" -ForegroundColor Cyan

# Überprüfung von Administratorrechten
function Test-AdminPrivileges {
    $isAdmin = $false
    
    # Überprüfung für Windows
    if ((Get-OperatingSystem) -eq "Windows") {
        $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = New-Object Security.Principal.WindowsPrincipal -ArgumentList $identity
        $isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    }
    # Überprüfung für Linux/MacOS
    else {
        $isAdmin = (id -u) -eq 0
    }
    
    return $isAdmin
}

# Am Anfang des Skripts einfügen:
if (-not (Test-AdminPrivileges)) {
    Write-Host "Dieses Skript benötigt Administratorrechte bzw. Root-Rechte." -ForegroundColor Red
    Write-Host "Bitte starten Sie PowerShell als Administrator/Root und versuchen Sie es erneut." -ForegroundColor Red
    exit 1
}

# Detaillierte OS-Informationen
function Get-OSDetails {
    if ($global:IsWindows) {
        $osInfo = Get-CimInstance Win32_OperatingSystem
        $winVer = [System.Environment]::OSVersion.Version
        return @{
            Name = $osInfo.Caption
            Version = $winVer.ToString()
            Build = $osInfo.BuildNumber
            Architecture = [System.Environment]::Is64BitOperatingSystem ? "x64" : "x86"
        }
    }
    elseif ($global:IsLinux) {
        $distro = if (Test-Path "/etc/os-release") {
            Get-Content "/etc/os-release" | ConvertFrom-StringData
        } else { @{NAME="Linux"; VERSION_ID="Unknown"} }
        
        return @{
            Name = $distro.NAME
            Version = $distro.VERSION_ID
            Architecture = uname -m
        }
    }
    elseif ($global:IsMacOS) {
        $osVer = sw_vers
        $productName = ($osVer | Where-Object { $_ -match "ProductName" }).Split(":")[1].Trim()
        $productVersion = ($osVer | Where-Object { $_ -match "ProductVersion" }).Split(":")[1].Trim()
        
        return @{
            Name = $productName
            Version = $productVersion
            Architecture = uname -m
        }
    }
}

$osDetails = Get-OSDetails()
Write-Host "Erkanntes System: $($osDetails.Name) $($osDetails.Version) ($($osDetails.Architecture))" -ForegroundColor Green

# Erkennung des Betriebssystems (kompatibel mit älteren PowerShell-Versionen)
function Get-OperatingSystem {
    if ($PSVersionTable.PSVersion.Major -ge 6) {
        # PowerShell Core (6+) hat diese Variablen eingebaut
        if ($global:IsWindows) { return "Windows" }
        if ($global:IsLinux) { return "Linux" }
        if ($global:IsMacOS) { return "MacOS" }
    } else {
        # PowerShell 5.1 oder älter - manuelle Erkennung
        if ($env:OS -match "Windows") { return "Windows" }
        $uname = if (Get-Command "uname" -ErrorAction SilentlyContinue) { & uname } else { "" }
        if ($uname -eq "Linux") { return "Linux" }
        if ($uname -eq "Darwin") { return "MacOS" }
    }
    return "Unknown"
}

# Installiere plattformspezifisch alle erforderlichen Basis-Werkzeuge
function Install-RequiredTools {
    $os = Get-OperatingSystem
    switch ($os) {
        "Windows" { Install-WindowsTools; break }
        "Linux" { Install-LinuxTools; break }
        "MacOS" { Install-MacTools; break }
        default { Write-Host "Betriebssystem nicht erkannt!" -ForegroundColor Red }
    }
}

function Install-WindowsTools {
    Write-Host "Installiere Windows-Entwicklungstools..." -ForegroundColor Yellow
    
    # 1. Visual Studio Build Tools erkennen/installieren
    $vsPath = Get-VSInstallationPath
    $vsInstalled = $null -ne $vsPath
    
    if (-not $vsInstalled) {
        Write-Host "Visual Studio Build Tools nicht gefunden." -ForegroundColor Yellow
        
        if (-not $Test) {
            # Dynamische Ermittlung der passenden VS-Version
            $vsInfo = Get-VisualStudioInstallerInfo
            $vsInstallerPath = "$env:TEMP\vs_buildtools.exe"
            $componentsParam = "--add " + ($vsInfo.RecommendedComponents -join " --add ")
            $vsWorkloadParams = "--quiet --wait --norestart --nocache --installPath `"C:\BuildTools`" $componentsParam"
            
            Write-Host "Lade Visual Studio $($vsInfo.Version) Build Tools herunter..." -ForegroundColor Yellow
            Invoke-WebRequest -Uri $vsInfo.Url -OutFile $vsInstallerPath
            
            Write-Host "Installiere Visual Studio Build Tools..." -ForegroundColor Yellow
            Start-Process -FilePath $vsInstallerPath -ArgumentList $vsWorkloadParams -Wait -NoNewWindow
            
            # Warte kurz und prüfe, ob Installation erfolgreich war
            Start-Sleep -Seconds 5
            $vsPath = Get-VSInstallationPath
            $vsInstalled = $null -ne $vsPath
            
            if ($vsInstalled) {
                Write-Host "Visual Studio Build Tools wurden erfolgreich installiert." -ForegroundColor Green
            } else {
                Write-Host "Installation der Visual Studio Build Tools konnte nicht verifiziert werden." -ForegroundColor Red
            }
        } else {
            Write-Host "TEST-MODUS: Würde Visual Studio Build Tools installieren" -ForegroundColor Yellow
        }
    } else {
        Write-Host "Visual Studio Build Tools gefunden in: $vsPath" -ForegroundColor Green
    }
    
    # 2. MSYS2 erkennen/installieren
    $msys2Path = "C:\msys64"
    if (-not (Test-Path "$msys2Path\usr\bin\bash.exe")) {
        Write-Host "MSYS2 nicht gefunden." -ForegroundColor Yellow
        
        if (-not $Test) {
            # Dynamische Ermittlung der neuesten MSYS2-Version
            $msys2Info = Get-LatestMSYS2Installer
            $msys2InstallerPath = "$env:TEMP\$($msys2Info.FileName)"
            $msys2InstallParams = "/InstallDir=C:\msys64 /Quiet"
            
            Write-Host "Lade MSYS2 $($msys2Info.Version) herunter..." -ForegroundColor Yellow
            Invoke-WebRequest -Uri $msys2Info.Url -OutFile $msys2InstallerPath
            
            Write-Host "Installiere MSYS2..." -ForegroundColor Yellow
            Start-Process -FilePath $msys2InstallerPath -ArgumentList $msys2InstallParams -Wait -NoNewWindow
            
            # Initialisiere MSYS2 (erster Start)
            if (Test-Path "$msys2Path\usr\bin\bash.exe") {
                Write-Host "Initialisiere MSYS2..." -ForegroundColor Yellow
                Start-Process -FilePath "$msys2Path\usr\bin\bash.exe" -ArgumentList "-lc 'exit'" -Wait -NoNewWindow
                Start-Sleep -Seconds 3
                Write-Host "MSYS2 wurde erfolgreich installiert und initialisiert." -ForegroundColor Green
            } else {
                Write-Host "MSYS2 Installation konnte nicht verifiziert werden." -ForegroundColor Red
            }
        } else {
            Write-Host "TEST-MODUS: Würde MSYS2 installieren" -ForegroundColor Yellow
        }
    } else {
        Write-Host "MSYS2 gefunden in: $msys2Path" -ForegroundColor Green
    }
    
    # 3. Python-Abhängigkeiten analysieren und entsprechende VS-Komponenten installieren
    if ($vsInstalled -and $requirementFile -and $mappingFile) {
        $pythonPackages = Get-Content $requirementFile | Where-Object { $_ -match '^[a-zA-Z0-9\._-]+' } | ForEach-Object { ($_ -split '==|>=|<=|>|<|~=|!=|===')[0].Trim().ToLower() }
        
        # Lade Mapping-Daten
        $mappingContent = Get-Content $mappingFile -Raw
        $mappingJson = ConvertFrom-Json $mappingContent
        
        # VS-Komponenten identifizieren
        $vsComponents = @{}
        
        if ($pythonPackages -and $mappingJson.windows.vs_components) {
            foreach ($package in $pythonPackages) {
                if ($mappingJson.windows.vs_components.$package) {
                    foreach ($component in $mappingJson.windows.vs_components.$package) {
                        $vsComponents[$component] = $package
                    }
                }
            }
        }
        
        # VS-Komponenten installieren falls notwendig
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
                    Write-Host "Installiere benötigte Visual Studio-Komponenten..." -ForegroundColor Yellow
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
    }
    
    # 4. MSYS2-Pakete installieren
    if (Test-Path "$msys2Path\usr\bin\bash.exe" -and $requirementFile -and $mappingFile) {
        # MSYS2-Abhängigkeiten identifizieren
        $pythonPackages = Get-Content $requirementFile | Where-Object { $_ -match '^[a-zA-Z0-9\._-]+' } | ForEach-Object { ($_ -split '==|>=|<=|>|<|~=|!=|===')[0].Trim().ToLower() }
        
        # Lade Mapping-Daten
        $mappingContent = Get-Content $mappingFile -Raw
        $mappingJson = ConvertFrom-Json $mappingContent
        
        # MSYS2-Pakete identifizieren
        $msys2Dependencies = @{}
        
        if ($pythonPackages -and $mappingJson.windows.packages) {
            foreach ($package in $pythonPackages) {
                if ($mappingJson.windows.packages.$package) {
                    foreach ($dep in $mappingJson.windows.packages.$package) {
                        $msys2Dependencies[$dep] = $package
                    }
                }
            }
        }
        
        # MSYS2-Pakete installieren
        if ($msys2Dependencies.Count -gt 0) {
            $depsParam = [string]::Join(",", $msys2Dependencies.Keys)
            $batPath = "$PSScriptRoot\MYSYS2_Setup.bat"
            
            if (Test-Path $batPath) {
                $testParam = if ($Test) { "/test" } else { "" }
                Write-Host "Führe MSYS2-Paketinstallation aus..." -ForegroundColor Yellow
                Start-Process -FilePath $batPath -ArgumentList $testParam, $depsParam -Wait -NoNewWindow
                Write-Host "MSYS2-Paketinstallation abgeschlossen." -ForegroundColor Green
            } else {
                Write-Host "MSYS2_Setup.bat nicht gefunden in $batPath" -ForegroundColor Red
            }
        } else {
            Write-Host "Keine MSYS2-Abhängigkeiten für die Python-Pakete gefunden." -ForegroundColor Yellow
        }
    }
}

function Install-LinuxTools {
    Write-Host "Installiere Linux-Entwicklungstools..." -ForegroundColor Yellow
    
    # Linux-Distribution erkennen
    $distro = Get-LinuxDistribution
    
    if ($distro -eq $null) {
        Write-Host "Linux-Distribution konnte nicht erkannt werden." -ForegroundColor Red
        return
    }
    
    Write-Host "Erkannte Distribution: $($distro.NAME) $($distro.VERSION_ID)" -ForegroundColor Green
    
    # Je nach Distro verschiedene Compiler und Entwicklungstools installieren
    if ($distro.ID -match "ubuntu|debian") {
        if (-not $Test) {
            Write-Host "Installiere build-essential und Entwicklertools..." -ForegroundColor Yellow
            bash -c "sudo apt-get update && sudo apt-get install -y build-essential cmake git python3-dev python3-pip"
            Write-Host "Basis-Entwicklungstools wurden installiert." -ForegroundColor Green
        } else {
            Write-Host "TEST-MODUS: Würde build-essential und Entwicklertools installieren" -ForegroundColor Yellow
        }
    }
    elseif ($distro.ID -match "fedora|rhel|centos") {
        if (-not $Test) {
            Write-Host "Installiere Development Tools und Bibliotheken..." -ForegroundColor Yellow
            bash -c "sudo dnf groupinstall -y 'Development Tools' && sudo dnf install -y cmake git python3-devel python3-pip"
            Write-Host "Basis-Entwicklungstools wurden installiert." -ForegroundColor Green
        } else {
            Write-Host "TEST-MODUS: Würde Development Tools und Bibliotheken installieren" -ForegroundColor Yellow
        }
    }
    elseif ($distro.ID -match "arch|manjaro") {
        if (-not $Test) {
            Write-Host "Installiere base-devel und Entwicklertools..." -ForegroundColor Yellow
            bash -c "sudo pacman -Sy --noconfirm base-devel cmake git python python-pip"
            Write-Host "Basis-Entwicklungstools wurden installiert." -ForegroundColor Green
        } else {
            Write-Host "TEST-MODUS: Würde base-devel und Entwicklertools installieren" -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "Die Distribution $($distro.ID) wird aktuell nicht unterstützt." -ForegroundColor Red
        Write-Host "Bitte installieren Sie Entwicklungswerkzeuge manuell." -ForegroundColor Red
    }
}

function Install-MacTools {
    Write-Host "Installiere macOS-Entwicklungstools..." -ForegroundColor Yellow
    
    # 1. Xcode Command Line Tools prüfen und installieren
    $xcodeInstalled = $false
    $xcodeCheckOutput = bash -c "xcode-select -p 2>/dev/null"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Xcode Command Line Tools sind bereits installiert." -ForegroundColor Green
        $xcodeInstalled = $true
    } else {
        Write-Host "Xcode Command Line Tools sind nicht installiert." -ForegroundColor Yellow
        
        if (-not $Test) {
            Write-Host "Starte Installation der Xcode Command Line Tools..." -ForegroundColor Yellow
            bash -c "xcode-select --install"
            Write-Host "Bitte befolgen Sie die Anweisungen im Popup-Fenster und drücken Sie eine Taste, wenn die Installation abgeschlossen ist." -ForegroundColor Yellow
            $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
            
            $xcodeCheckOutput = bash -c "xcode-select -p 2>/dev/null"
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Xcode Command Line Tools wurden erfolgreich installiert." -ForegroundColor Green
                $xcodeInstalled = $true
            } else {
                Write-Host "Xcode Command Line Tools konnten nicht installiert werden." -ForegroundColor Red
            }
        } else {
            Write-Host "TEST-MODUS: Würde Xcode Command Line Tools installieren" -ForegroundColor Yellow
        }
    }
    
    # 2. Homebrew prüfen und installieren
    $brewInstalled = $false
    $brewCheckOutput = bash -c "which brew 2>/dev/null"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Homebrew ist bereits installiert." -ForegroundColor Green
        $brewInstalled = $true
    } else {
        Write-Host "Homebrew ist nicht installiert." -ForegroundColor Yellow
        
        if (-not $Test) {
            if ($xcodeInstalled) {
                Write-Host "Installiere Homebrew..." -ForegroundColor Yellow
                bash -c '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
                
                $brewCheckOutput = bash -c "which brew 2>/dev/null"
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "Homebrew wurde erfolgreich installiert." -ForegroundColor Green
                    $brewInstalled = $true
                } else {
                    Write-Host "Homebrew konnte nicht installiert werden." -ForegroundColor Red
                }
            } else {
                Write-Host "Xcode Command Line Tools sind für die Homebrew-Installation erforderlich." -ForegroundColor Red
            }
        } else {
            Write-Host "TEST-MODUS: Würde Homebrew installieren" -ForegroundColor Yellow
        }
    }
    
    # 3. Grundlegende Entwicklerpakete installieren
    if ($brewInstalled -and -not $Test) {
        Write-Host "Installiere grundlegende Entwicklerpakete..." -ForegroundColor Yellow
        bash -c "brew install python cmake git"
        Write-Host "Grundlegende Entwicklerpakete wurden installiert." -ForegroundColor Green
    } elseif ($Test) {
        Write-Host "TEST-MODUS: Würde grundlegende Entwicklerpakete (python, cmake, git) installieren" -ForegroundColor Yellow
    }
}

# Hauptausführung
Install-RequiredTools

# Suche requirements.txt
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

if ($requirementFile) {
    Install-Packages -RequirementsFile $requirementFile
} else {
    Write-Host "Keine requirements.txt gefunden. Keine Systemabhängigkeiten werden installiert." -ForegroundColor Yellow
}

# Status-Zusammenfassung am Ende
Write-Host "`nZusammenfassung:" -ForegroundColor Cyan

# Plattformspezifische Statusanzeige
$os = Get-OperatingSystem
switch ($os) {
    "Windows" {
        # VS und MSYS2 Status anzeigen
        Write-Host "- Visual Studio: " -NoNewline
        if (Get-VSInstallationPath) { 
            Write-Host "Installiert" -ForegroundColor Green 
        } else { 
            Write-Host "Nicht installiert" -ForegroundColor Red 
        }

        $msys2Path = "C:\msys64"
        Write-Host "- MSYS2: " -NoNewline
        if (Test-Path "$msys2Path\usr\bin\bash.exe") { 
            Write-Host "Installiert" -ForegroundColor Green 
        } else { 
            Write-Host "Nicht installiert" -ForegroundColor Red 
        }
    }
    "Linux" {
        # Linux-spezifischer Status
        $distro = Get-LinuxDistribution
        Write-Host "- Linux-Distribution: " -NoNewline
        Write-Host "$($distro.NAME) $($distro.VERSION_ID)" -ForegroundColor Green
        
        # GCC/Clang Status
        Write-Host "- Compiler: " -NoNewline
        $gccVersion = bash -c "gcc --version 2>/dev/null | head -n 1"
        if ($LASTEXITCODE -eq 0) {
            Write-Host "GCC $gccVersion" -ForegroundColor Green
        } else {
            Write-Host "GCC nicht gefunden" -ForegroundColor Yellow
            
            $clangVersion = bash -c "clang --version 2>/dev/null | head -n 1"
            if ($LASTEXITCODE -eq 0) {
                Write-Host "- Alternate Compiler: Clang $clangVersion" -ForegroundColor Green
            }
        }
    }
    "MacOS" {
        # macOS-spezifischer Status
        Write-Host "- Xcode Command Line Tools: " -NoNewline
        $xcodeCheckOutput = bash -c "xcode-select -p 2>/dev/null"
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Installiert" -ForegroundColor Green 
        } else { 
            Write-Host "Nicht installiert" -ForegroundColor Red 
        }
        
        # Homebrew Status
        Write-Host "- Homebrew: " -NoNewline
        $brewCheckOutput = bash -c "which brew 2>/dev/null"
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Installiert" -ForegroundColor Green
            $brewVersion = bash -c "brew --version | head -n 1"
            Write-Host "  Version: $brewVersion" -ForegroundColor Green
        } else { 
            Write-Host "Nicht installiert" -ForegroundColor Red 
        }
    }
}

# Installation abschließen
Write-Host "`nSetup abgeschlossen! Das System wurde für die Entwicklung vorbereitet." -ForegroundColor Green

# Beenden der Protokollierung
Stop-Transcript
Write-Host "`nProtokoll gespeichert in: $logPath" -ForegroundColor Cyan

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

# Funktion zum Installieren der Abhängigkeiten basierend auf requirements.txt
function Install-Packages {
    param(
        [string]$RequirementsFile
    )

    if (-not (Test-Path $RequirementsFile)) {
        Write-Host "Keine requirements.txt gefunden in: $RequirementsFile" -ForegroundColor Yellow
        return
    }

    Write-Host "`nAnalysiere Python-Abhängigkeiten aus $RequirementsFile..." -ForegroundColor Cyan
    
    # Mapping-Datei finden
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
    
    if (-not $mappingFile) {
        Write-Host "Keine package_mapping.json gefunden. Keine Systemabhängigkeiten werden installiert." -ForegroundColor Yellow
        return
    }

    # Laden und Parsen der Python-Pakete
    $pythonPackages = Get-Content $RequirementsFile | 
                      Where-Object { $_ -match '^[a-zA-Z0-9\._-]+' } | 
                      ForEach-Object { ($_ -split '==|>=|<=|>|<|~=|!=|===')[0].Trim().ToLower() }
    
    Write-Host "Gefundene Python-Pakete: $($pythonPackages.Count)" -ForegroundColor Yellow
    
    # Konvertiere die JSON-Datei
    $mappingContent = Get-Content $mappingFile -Raw
    $mappingJson = ConvertFrom-Json $mappingContent
    
    # Plattformspezifische Installation
    if ($global:IsWindows) {
        Install-WindowsDependencies -MappingJson $mappingJson -PythonPackages $pythonPackages
    }
    elseif ($global:IsLinux) {
        Install-LinuxDependencies -MappingJson $mappingJson -PythonPackages $pythonPackages
    }
    elseif ($global:IsMacOS) {
        Install-MacDependencies -MappingJson $mappingJson -PythonPackages $pythonPackages
    }
}

# Windows-spezifische Abhängigkeiten installieren
function Install-WindowsDependencies {
    param(
        $MappingJson,
        $PythonPackages
    )
    
    # 1. VS-Komponenten identifizieren
    $vsComponents = @{}
    
    foreach ($package in $PythonPackages) {
        if ($MappingJson.windows.vs_components.$package) {
            foreach ($comp in $MappingJson.windows.vs_components.$package) {
                $vsComponents[$comp] = $package
            }
        }
    }
    
    # 2. MSYS2-Pakete identifizieren
    $msys2Dependencies = @{}
    
    foreach ($package in $PythonPackages) {
        if ($MappingJson.windows.packages.$package) {
            foreach ($dep in $MappingJson.windows.packages.$package) {
                $msys2Dependencies[$dep] = $package
            }
        }
    }
    
    # Zeige erkannte Komponenten
    if ($vsComponents.Count -gt 0) {
        Write-Host "Erkannte Visual Studio-Komponenten:" -ForegroundColor Cyan
        foreach ($comp in $vsComponents.Keys) {
            Write-Host "  - $comp (benötigt für $($vsComponents[$comp]))" -ForegroundColor Yellow
        }
        
        # VS-Komponenten installieren
        Install-VisualStudioComponents -Components $vsComponents
    }
    
    if ($msys2Dependencies.Count -gt 0) {
        Write-Host "Erkannte MSYS2-Abhängigkeiten:" -ForegroundColor Cyan
        foreach ($dep in $msys2Dependencies.Keys) {
            Write-Host "  - $dep (benötigt für $($msys2Dependencies[$dep]))" -ForegroundColor Yellow
        }
        
        # MSYS2-Pakete installieren
        Install-MSYS2Packages -Dependencies $msys2Dependencies
    }
}

# Linux-spezifische Abhängigkeiten installieren
function Install-LinuxDependencies {
    param(
        $MappingJson,
        $PythonPackages
    )
    
    # Linux-Distribution erkennen
    $distroInfo = Get-LinuxDistribution
    $distro = $distroInfo.ID.ToLower()
    
    # Unterstützte Distributionen prüfen
    $supportedDistros = @("ubuntu", "debian", "fedora", "rhel", "centos", "arch", "manjaro")
    $mappingSection = ""
    
    # Richtige Mapping-Sektion auswählen
    if ($distro -eq "ubuntu" -or $distro -eq "debian") {
        $mappingSection = "ubuntu" # Debian-basierte Systeme
        $packageManager = "apt"
    }
    elseif ($distro -eq "fedora" -or $distro -eq "rhel" -or $distro -eq "centos") {
        $mappingSection = "fedora" # RedHat-basierte Systeme
        $packageManager = if ($distro -eq "fedora") { "dnf" } else { "yum" }
    }
    elseif ($distro -eq "arch" -or $distro -eq "manjaro") {
        $mappingSection = "arch"   # Arch-basierte Systeme
        $packageManager = "pacman"
    }
    else {
        Write-Host "Ihre Linux-Distribution '$distro' wird nicht direkt unterstützt. Versuche generische Installation..." -ForegroundColor Yellow
        if (Get-Command apt -ErrorAction SilentlyContinue) { 
            $mappingSection = "ubuntu"
            $packageManager = "apt"
        }
        elseif (Get-Command dnf -ErrorAction SilentlyContinue) { 
            $mappingSection = "fedora" 
            $packageManager = "dnf"
        }
        elseif (Get-Command pacman -ErrorAction SilentlyContinue) { 
            $mappingSection = "arch"
            $packageManager = "pacman" 
        }
        else {
            Write-Host "Konnte keinen unterstützten Paketmanager finden. Bitte installieren Sie die Abhängigkeiten manuell." -ForegroundColor Red
            return
        }
    }
    
    # Pakete identifizieren
    $packagesToInstall = @()
    
    foreach ($package in $PythonPackages) {
        if ($MappingJson.linux.$mappingSection.$package) {
            foreach ($dep in $MappingJson.linux.$mappingSection.$package) {
                if ($packagesToInstall -notcontains $dep) {
                    $packagesToInstall += $dep
                }
            }
        }
    }
    
    # Zeige erkannte Pakete
    if ($packagesToInstall.Count -gt 0) {
        Write-Host "Erkannte Linux-Pakete ($mappingSection):" -ForegroundColor Cyan
        foreach ($pkg in $packagesToInstall) {
            Write-Host "  - $pkg" -ForegroundColor Yellow
        }
        
        # Pakete installieren
        if (-not $Test) {
            Write-Host "Installiere $($packagesToInstall.Count) Systemabhängigkeiten..." -ForegroundColor Cyan
            
            $packages = $packagesToInstall -join " "
            
            switch ($packageManager) {
                "apt" {
                    $installCmd = "sudo apt update && sudo apt install -y $packages"
                }
                "dnf" {
                    $installCmd = "sudo dnf install -y $packages"
                }
                "yum" {
                    $installCmd = "sudo yum install -y $packages"
                }
                "pacman" {
                    $installCmd = "sudo pacman -Sy --noconfirm $packages"
                }
            }
            
            # Linux-Befehl ausführen
            bash -c "$installCmd"
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Abhängigkeiten erfolgreich installiert." -ForegroundColor Green
            } else {
                Write-Host "Fehler bei der Installation der Abhängigkeiten." -ForegroundColor Red
            }
        } else {
            Write-Host "TEST-MODUS: Würde folgende Pakete installieren: $($packagesToInstall -join ", ")" -ForegroundColor Yellow
        }
    } else {
        Write-Host "Keine Linux-Paketabhängigkeiten für die Python-Pakete gefunden." -ForegroundColor Yellow
    }
}

# macOS-spezifische Abhängigkeiten installieren
function Install-MacDependencies {
    param(
        $MappingJson,
        $PythonPackages
    )
    
    # Prüfen, ob Homebrew installiert ist
    $hasHomebrew = $null -ne (Get-Command brew -ErrorAction SilentlyContinue)
    
    if (-not $hasHomebrew) {
        Write-Host "Homebrew ist nicht installiert, aber wird für die Paketinstallation benötigt." -ForegroundColor Yellow
        
        if (-not $Test) {
            Write-Host "Installiere Homebrew..." -ForegroundColor Cyan
            bash -c '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
            
            # Prüfe erneut
            $hasHomebrew = $null -ne (Get-Command brew -ErrorAction SilentlyContinue)
            if (-not $hasHomebrew) {
                Write-Host "Homebrew konnte nicht installiert werden. Bitte installieren Sie es manuell." -ForegroundColor Red
                return
            }
        } else {
            Write-Host "TEST-MODUS: Würde Homebrew installieren" -ForegroundColor Yellow
            return
        }
    }
    
    # Pakete identifizieren
    $brewPackages = @()
    
    foreach ($package in $PythonPackages) {
        if ($MappingJson.macos.homebrew.$package) {
            foreach ($dep in $MappingJson.macos.homebrew.$package) {
                if ($brewPackages -notcontains $dep) {
                    $brewPackages += $dep
                }
            }
        }
    }
    
    # Zeige erkannte Pakete
    if ($brewPackages.Count -gt 0) {
        Write-Host "Erkannte Homebrew-Pakete:" -ForegroundColor Cyan
        foreach ($pkg in $brewPackages) {
            Write-Host "  - $pkg" -ForegroundColor Yellow
        }
        
        # Pakete installieren
        if (-not $Test) {
            Write-Host "Installiere $($brewPackages.Count) Homebrew-Pakete..." -ForegroundColor Cyan
            
            $packages = $brewPackages -join " "
            bash -c "brew install $packages"
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Abhängigkeiten erfolgreich installiert." -ForegroundColor Green
            } else {
                Write-Host "Fehler bei der Installation der Abhängigkeiten." -ForegroundColor Red
            }
        } else {
            Write-Host "TEST-MODUS: Würde folgende Homebrew-Pakete installieren: $($brewPackages -join ", ")" -ForegroundColor Yellow
        }
    } else {
        Write-Host "Keine Homebrew-Paketabhängigkeiten für die Python-Pakete gefunden." -ForegroundColor Yellow
    }
}

# Hilfsfunktion für Linux-Distribution
function Get-LinuxDistribution {
    if (Test-Path "/etc/os-release") {
        $osRelease = Get-Content "/etc/os-release" | ForEach-Object {
            $parts = $_ -split "="
            if ($parts.Count -eq 2) {
                $key = $parts[0]
                $value = $parts[1].Trim('"')
                [PSCustomObject]@{
                    Key = $key
                    Value = $value
                }
            }
        }

        $result = @{}
        foreach ($item in $osRelease) {
            $result[$item.Key] = $item.Value
        }

        return [PSCustomObject]$result
    }
    return $null
}

# Protokollierung einrichten
$logPath = ""
if ((Get-OperatingSystem) -eq "Windows") {
    $logPath = "$env:TEMP\DevEnv_Setup_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
} else {
    $logPath = "/tmp/DevEnv_Setup_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
}

# Titelbildschirm anzeigen
Clear-Host
Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "               ENTWICKLUNGSUMGEBUNG SETUP                " -ForegroundColor Cyan
Write-Host "     Plattformübergreifender Abhängigkeitsinstaller     " -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "Setup wird ausgeführt. Protokollierung in: $logPath" -ForegroundColor Green

# Protokollierung starten
Start-Transcript -Path $logPath

# Testmodus-Hinweis
if ($Test) {
    Write-Host "=========================================================" -ForegroundColor Yellow
    Write-Host "                      TEST-MODUS                         " -ForegroundColor Yellow
    Write-Host "  Das Skript wird nur simuliert, keine Aenderungen       " -ForegroundColor Yellow
    Write-Host "  werden am System vorgenommen.                          " -ForegroundColor Yellow
    Write-Host "=========================================================" -ForegroundColor Yellow
}

# Funktionen für VS und MSYS2
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

# Plattformspezifische Installer-Funktionen
function Install-WindowsTools {
    # ... bestehender Code ...
}

function Install-LinuxTools {
    # ... bestehender Code ...
}

function Install-MacTools {
    # ... bestehender Code ...
}

# Hauptausführung
Install-RequiredTools

# Suche requirements.txt
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

if ($requirementFile) {
    Install-Packages -RequirementsFile $requirementFile
} else {
    Write-Host "Keine requirements.txt gefunden. Keine Systemabhängigkeiten werden installiert." -ForegroundColor Yellow
}

# Status-Zusammenfassung am Ende
Write-Host "`nZusammenfassung:" -ForegroundColor Cyan

# Plattformspezifische Statusanzeige
$os = Get-OperatingSystem
switch ($os) {
    "Windows" {
        # VS und MSYS2 Status anzeigen
        Write-Host "- Visual Studio: " -NoNewline
        if (Get-VSInstallationPath) { 
            Write-Host "Installiert" -ForegroundColor Green 
        } else { 
            Write-Host "Nicht installiert" -ForegroundColor Red 
        }

        $msys2Path = "C:\msys64"
        Write-Host "- MSYS2: " -NoNewline
        if (Test-Path "$msys2Path\usr\bin\bash.exe") { 
            Write-Host "Installiert" -ForegroundColor Green 
        } else { 
            Write-Host "Nicht installiert" -ForegroundColor Red 
        }
    }
    "Linux" {
        # Linux-spezifischer Status
        $distro = Get-LinuxDistribution
        Write-Host "- Linux-Distribution: " -NoNewline
        Write-Host "$($distro.NAME) $($distro.VERSION_ID)" -ForegroundColor Green
        
        # GCC/Clang Status
        Write-Host "- Compiler: " -NoNewline
        $gccVersion = bash -c "gcc --version 2>/dev/null | head -n 1"
        if ($LASTEXITCODE -eq 0) {
            Write-Host "GCC $gccVersion" -ForegroundColor Green
        } else {
            Write-Host "GCC nicht gefunden" -ForegroundColor Yellow
            
            $clangVersion = bash -c "clang --version 2>/dev/null | head -n 1"
            if ($LASTEXITCODE -eq 0) {
                Write-Host "- Alternate Compiler: Clang $clangVersion" -ForegroundColor Green
            }
        }
    }
    "MacOS" {
        # macOS-spezifischer Status
        Write-Host "- Xcode Command Line Tools: " -NoNewline
        $xcodeCheckOutput = bash -c "xcode-select -p 2>/dev/null"
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Installiert" -ForegroundColor Green 
        } else { 
            Write-Host "Nicht installiert" -ForegroundColor Red 
        }
        
        # Homebrew Status
        Write-Host "- Homebrew: " -NoNewline
        $brewCheckOutput = bash -c "which brew 2>/dev/null"
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Installiert" -ForegroundColor Green
            $brewVersion = bash -c "brew --version | head -n 1"
            Write-Host "  Version: $brewVersion" -ForegroundColor Green
        } else { 
            Write-Host "Nicht installiert" -ForegroundColor Red 
        }
    }
}

# Installation abschließen
Write-Host "`nSetup abgeschlossen! Das System wurde für die Entwicklung vorbereitet." -ForegroundColor Green

# Beenden der Protokollierung
Stop-Transcript
Write-Host "`nProtokoll gespeichert in: $logPath" -ForegroundColor Cyan 