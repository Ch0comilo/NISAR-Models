param([string]$Notebook = 'notebooks/NISAR_Water_Improved_Classifier.ipynb')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$notebookPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $notebookPython)) {
    throw 'Restore the environment with: uv sync --frozen --extra test --extra notebook'
}
& $notebookPython -m jupyterlab $Notebook
exit $LASTEXITCODE
