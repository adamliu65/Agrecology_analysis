# PowerShell Profile setup script
# Run this to enable 'streamlit' and 'python' commands in PowerShell

$profileDir = Split-Path $PROFILE
if (!(Test-Path $profileDir)) { New-Item -ItemType Directory -Path $profileDir -Force | Out-Null }

$anacondaPath = "C:\Users\adam.liu\AppData\Local\anaconda3"

# Add to PATH in current session
$env:PATH = "$anacondaPath;$anacondaPath\Scripts;" + $env:PATH

Write-Host "Python and Streamlit are now available in this session!" -ForegroundColor Green
Write-Host "Testing..." -ForegroundColor Cyan

python --version
streamlit --version
