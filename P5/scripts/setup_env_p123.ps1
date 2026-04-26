param(
    [string]$VenvPath = ".venv_p123",
    [switch]$Lightweight
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$p5Root = Join-Path $scriptDir ".."
$workspaceRoot = Join-Path $p5Root ".."

$venvFullPath = Join-Path $p5Root $VenvPath

if (-not (Test-Path $venvFullPath)) {
    py -3.12 -m venv $venvFullPath
}

$pythonExe = Join-Path $venvFullPath "Scripts/python.exe"

& $pythonExe -m pip install --upgrade pip setuptools wheel

if (-not $Lightweight) {
    & $pythonExe -m pip install -r (Join-Path $workspaceRoot "P2/requirements.txt")
    & $pythonExe -m pip install -e (Join-Path $workspaceRoot "P1[data,model,ner,dev]")
    & $pythonExe -m pip install -e (Join-Path $workspaceRoot "P3[dev]")

    # Probe torch import; some Windows environments fail with WinError 1114 on newer wheels.
    & $pythonExe -c "import torch; print('torch', torch.__version__)" *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Torch import failed; applying CPU fallback (torch==2.3.1)."
        & $pythonExe -m pip uninstall -y torch torchvision torchaudio *> $null
        & $pythonExe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
        & $pythonExe -c "import torch; print('torch_ok', torch.__version__)"
    }
}

& $pythonExe -m pip install -e $p5Root

Write-Host "Environment ready at $venvFullPath"
Write-Host "Use: $pythonExe"
Write-Host "If torch import fails with WinError 1114, install/repair Microsoft Visual C++ Redistributable (x64)."
