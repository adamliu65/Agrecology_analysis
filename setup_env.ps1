# PowerShell script to add Anaconda to PATH
Write-Host "Adding Anaconda to Windows PATH..." -ForegroundColor Cyan

$anacondaPath = "C:\Users\adam.liu\AppData\Local\anaconda3"
$scriptsPath = "$anacondaPath\Scripts"

# Get current PATH
$currentPath = [System.Environment]::GetEnvironmentVariable("PATH", [System.EnvironmentVariableTarget]::User)

# Check if already in PATH
if ($currentPath -notcontains $anacondaPath) {
    $newPath = "$currentPath;$anacondaPath;$scriptsPath"
    [System.Environment]::SetEnvironmentVariable("PATH", $newPath, [System.EnvironmentVariableTarget]::User)
    Write-Host "✓ Successfully added to PATH" -ForegroundColor Green
    Write-Host "Please close and reopen PowerShell for changes to take effect" -ForegroundColor Yellow
} else {
    Write-Host "✓ Anaconda is already in PATH" -ForegroundColor Green
}

# Verify installation
Write-Host "`nVerifying installation..." -ForegroundColor Cyan
& python --version
& streamlit --version
