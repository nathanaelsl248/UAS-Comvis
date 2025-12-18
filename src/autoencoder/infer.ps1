# Launcher untuk Inference Autoencoder
# Contoh penggunaan:
#   .\infer.ps1 single input.png output.png
#   .\infer.ps1 batch input_folder output_folder
#   .\infer.ps1 evaluate test_X4.json output_folder

$ProjectRoot = "C:\Users\beast\Documents\Kuliah\Computer Vision\UASComVis\UAS-Comvis-main\UAS-Comvis-main"
$VenvPython = "C:\Users\beast\Documents\Kuliah\Computer Vision\UASComVis\UAS-Comvis-main\.venv\Scripts\python.exe"

Write-Host "Starting Autoencoder Inference..." -ForegroundColor Green
Write-Host "Project: $ProjectRoot" -ForegroundColor Cyan

Push-Location $ProjectRoot

# Parse mode dan arguments
if ($args.Count -eq 0) {
    Write-Host "`nUsage:" -ForegroundColor Yellow
    Write-Host "  Single image:  .\infer.ps1 single <input.png> <output.png>"
    Write-Host "  Batch folder:  .\infer.ps1 batch <input_folder> <output_folder>"
    Write-Host "  Evaluate JSON: .\infer.ps1 evaluate <test_X4.json> <output_folder>"
    Write-Host "  Demo mode:     .\infer.ps1 demo"
    Pop-Location
    exit 1
}

$Mode = $args[0]
$InferArgs = @("--mode", $Mode)

if ($Mode -eq "single" -and $args.Count -ge 3) {
    $InferArgs += @("--input", $args[1], "--output", $args[2])
    Write-Host "Mode: Single image" -ForegroundColor Cyan
    Write-Host "Input: $($args[1])" -ForegroundColor Cyan
    Write-Host "Output: $($args[2])" -ForegroundColor Cyan
}
elseif ($Mode -eq "batch" -and $args.Count -ge 3) {
    $InferArgs += @("--input", $args[1], "--output", $args[2])
    Write-Host "Mode: Batch processing" -ForegroundColor Cyan
    Write-Host "Input folder: $($args[1])" -ForegroundColor Cyan
    Write-Host "Output folder: $($args[2])" -ForegroundColor Cyan
}
elseif ($Mode -eq "evaluate" -and $args.Count -ge 3) {
    $InferArgs += @("--test-json", $args[1], "--output", $args[2])
    Write-Host "Mode: Evaluation" -ForegroundColor Cyan
    Write-Host "Test JSON: $($args[1])" -ForegroundColor Cyan
    Write-Host "Output folder: $($args[2])" -ForegroundColor Cyan
}
elseif ($Mode -eq "demo") {
    Write-Host "Mode: Interactive demo" -ForegroundColor Cyan
}
else {
    Write-Host "Invalid arguments. See usage above." -ForegroundColor Red
    Pop-Location
    exit 1
}

& $VenvPython -m src.autoencoder.infer $InferArgs

Pop-Location

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nInference completed successfully!" -ForegroundColor Green
} else {
    Write-Host "`nInference exited with errors." -ForegroundColor Red
}
