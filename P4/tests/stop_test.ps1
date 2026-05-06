# Stop all Python processes and remove Qdrant lock files

Write-Host "Stopping all Python processes..." -ForegroundColor Yellow
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force

Write-Host "Removing Qdrant lock files..." -ForegroundColor Yellow
$lockPaths = @(
    "P3\data\processed\qdrant\*.lock",
    "data\processed\qdrant\*.lock"
)
foreach ($path in $lockPaths) {
    Remove-Item -Path $path -Force -ErrorAction SilentlyContinue
}

Write-Host "Cleanup completed." -ForegroundColor Green