$ProjectRoot = "C:\Users\beast\Documents\Kuliah\Computer Vision\UASComVis\UAS-Comvis-main\UAS-Comvis-main"
$VenvPython = "C:\Users\beast\Documents\Kuliah\Computer Vision\UASComVis\UAS-Comvis-main\.venv\Scripts\python.exe"

Write-Host "Starting Autoencoder Training..." -ForegroundColor Green
Write-Host "Project: $ProjectRoot" -ForegroundColor Cyan

Push-Location $ProjectRoot
$Args = $args -join " "

if ($Args) {
    Write-Host "Running with args: $Args" -ForegroundColor Yellow
    & $VenvPython -m src.autoencoder.train $Args.Split(" ")
} else {
    & $VenvPython -m src.autoencoder.train
}

Pop-Location

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nTraining completed successfully!" -ForegroundColor Green
} else {
    Write-Host "`nTraining exited with errors." -ForegroundColor Red
}
