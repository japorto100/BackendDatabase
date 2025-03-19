# update_github.ps1 - PowerShell-Script zum einfachen Aktualisieren des GitHub-Repositories

Write-Host "=== GitHub Update Script ===" -ForegroundColor Yellow

# Überprüfen Sie die aktuelle Remote-URL
Write-Host "Aktuelle Remote-URL:" -ForegroundColor Blue
git remote -v

# Ändern Sie die Remote-URL auf Ihr Repository
Write-Host "Setze Remote-URL auf dein Repository..." -ForegroundColor Blue
git remote set-url origin https://github.com/japorto100/Backend-to-merge.git

# Überprüfen Sie, ob die Änderung erfolgreich war
Write-Host "Neue Remote-URL:" -ForegroundColor Blue
git remote -v

# Entfernen Sie das Submodul aus der Git-Konfiguration
Write-Host "Entferne localgpt_vision_django aus der Git-Konfiguration..." -ForegroundColor Blue
git rm --cached localgpt_vision_django 2>$null
git config -f .git/config --remove-section submodule.localgpt_vision_django 2>$null
Remove-Item -Path .git/modules/localgpt_vision_django -Recurse -Force -ErrorAction SilentlyContinue

# 1. Aktuelle Änderungen anzeigen
Write-Host "Aktuelle Änderungen:" -ForegroundColor Blue
git status -s

# 2. Bestätigung vom Benutzer einholen
$confirm = Read-Host "Änderungen committen und pushen? (Y/n)"
if ($confirm -eq "n" -or $confirm -eq "N") {
    Write-Host "Vorgang abgebrochen." -ForegroundColor Yellow
    exit
}

# Standardnachricht verwenden oder benutzerdefinierte Nachricht abfragen
$use_default = Read-Host "Standardnachricht 'Update Repository' verwenden? (Y/n)"
if ($use_default -eq "n" -or $use_default -eq "N") {
    $commit_msg = Read-Host "Commit-Nachricht eingeben"
} else {
    $commit_msg = "Update Repository"
}

# 3. Alle Änderungen (neue, modifizierte und gelöschte Dateien) hinzufügen
Write-Host "Füge alle Änderungen hinzu..." -ForegroundColor Blue
git add --all

# 4. Commit erstellen
Write-Host "Erstelle Commit..." -ForegroundColor Blue
git commit -m $commit_msg

# 5. Zu GitHub pushen
Write-Host "Pushe zu GitHub..." -ForegroundColor Blue
git push

# 6. Erfolgsmeldung
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Update erfolgreich! Änderungen wurden zu GitHub gepusht." -ForegroundColor Green
} else {
    Write-Host "⚠ Es gab ein Problem beim Pushen. Bitte überprüfen Sie die Fehlermeldungen." -ForegroundColor Yellow
}
