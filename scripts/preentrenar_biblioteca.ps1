<#
.SYNOPSIS
  Prepara una biblioteca de clips de ecografia pulmonar SIN anotar y ejecuta el
  preentrenamiento autosupervisado de qLUS-Neo en este computador (Windows).

.DESCRIPTION
  1. Crea el entorno de Python (.venv) la primera vez e instala qLUS-Neo.
  2. qlus biblioteca: convierte los clips (DICOM/video, con subcarpetas) en
     datos\biblioteca\Bnnnnn.npz anonimizados + inventario.csv + mosaicos\.
     Se puede reanudar: lo ya procesado no se repite.
  3. qlus preentrenar: entrena la U-Net a reconstruir parches borrados
     -> modelos\preentrenado.pt (+ .ejemplos.png y .historial.json).
  Los clips nunca salen del computador.
  (Archivo solo en ASCII a proposito: Windows PowerShell 5.1 lee los .ps1 sin BOM como ANSI.)

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\preentrenar_biblioteca.ps1 `
    -Biblioteca "C:\Users\<usuario>\OneDrive\...\Biblioteca Ecografias pulmonares"

.EXAMPLE
  # Solo preparar la biblioteca y revisar los mosaicos antes de entrenar
  powershell -ExecutionPolicy Bypass -File scripts\preentrenar_biblioteca.ps1 -Biblioteca "..." -SoloIngesta
#>
param(
    [Parameter(Mandatory = $true)][string]$Biblioteca,
    [int]$Epocas = 30,
    [int]$IterPorEpoca = 200,
    [switch]$SoloIngesta
)
$ErrorActionPreference = "Stop"
$raiz = Split-Path -Parent $PSScriptRoot
Set-Location $raiz
# una barra final ("...\") romperia las comillas al pasar la ruta a Python
$Biblioteca = $Biblioteca.TrimEnd('\')
if (-not (Test-Path -LiteralPath $Biblioteca)) { throw "No existe la carpeta: $Biblioteca" }

$py = Join-Path $raiz ".venv\Scripts\python.exe"
function Ejecutar([string[]]$argumentos) {
    & $py @argumentos
    if ($LASTEXITCODE -ne 0) { throw "Fallo: python $($argumentos -join ' ')" }
}

if (-not (Test-Path $py)) {
    Write-Host "Creando el entorno de Python (.venv)..."
    if (Get-Command py -ErrorAction SilentlyContinue) { py -3 -m venv .venv }
    elseif (Get-Command python -ErrorAction SilentlyContinue) { python -m venv .venv }
    else { throw "No se encontro Python. Instalelo desde https://www.python.org/downloads/ (marque 'Add python.exe to PATH')." }
    Ejecutar @("-m", "pip", "install", "--upgrade", "pip")
}
Write-Host "Instalando o actualizando qLUS-Neo y sus dependencias..."
Ejecutar @("-m", "pip", "install", "-q", "-e", ".[jpeg]")

$cuda = & $py -c "import torch; print(torch.cuda.is_available())"
if ($cuda -ne "True" -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Host "Hay una GPU NVIDIA pero PyTorch no la usa. Para entrenar mucho mas rapido:" -ForegroundColor Yellow
    Write-Host "  .venv\Scripts\python.exe -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126" -ForegroundColor Yellow
}

Write-Host "`n== 1/2 Preparando la biblioteca ==" -ForegroundColor Cyan
Ejecutar @("-m", "qlus.cli", "biblioteca", "--origen", $Biblioteca, "--salida", "datos\biblioteca")
Write-Host "Revise datos\biblioteca\mosaicos\ (que no quede texto con datos del paciente) e inventario.csv."
if ($SoloIngesta) { return }

Write-Host "`n== 2/2 Preentrenamiento (Ctrl+C lo interrumpe; el mejor modelo ya queda guardado) ==" -ForegroundColor Cyan
Ejecutar @("-m", "qlus.cli", "preentrenar", "--datos", "datos\biblioteca", "--salida", "modelos\preentrenado.pt",
           "--epocas", "$Epocas", "--iter-por-epoca", "$IterPorEpoca")
Write-Host "`nListo: modelos\preentrenado.pt  (ver modelos\preentrenado.ejemplos.png)" -ForegroundColor Green
