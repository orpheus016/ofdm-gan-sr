# PowerShell script to simulate Simple GAN RTL
# Usage: .\simulate.ps1

param(
    [switch]$View,
    [switch]$Clean
)

$ErrorActionPreference = "Continue"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Simple GAN RTL Simulation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$VerilogFiles = @(
    "..\activation_tanh.v",
    "activation_sigmoid.v",
    "simple_gan_weights.v",
    "simple_generator.v",
    "simple_discriminator.v",
    "simple_gan_top.v",
    "tb_simple_gan.v"
)

$OutputVVP = "tb_simple_gan.vvp"
$OutputVCD = "tb_simple_gan.vcd"

if ($Clean) {
    Write-Host "Cleaning generated files..." -ForegroundColor Yellow
    Remove-Item -Path "*.vvp" -ErrorAction SilentlyContinue
    Remove-Item -Path "*.vcd" -ErrorAction SilentlyContinue
    Write-Host "Done." -ForegroundColor Green
    exit 0
}

# Check for Icarus Verilog
if (-not (Get-Command "iverilog" -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Icarus Verilog (iverilog) not found in PATH" -ForegroundColor Red
    Write-Host "Please install Icarus Verilog: https://bleyer.org/icarus/" -ForegroundColor Yellow
    exit 1
}

# Compile
Write-Host "Compiling Verilog sources..." -ForegroundColor Green
$CompileArgs = @("-g2012", "-o", $OutputVVP, "-I..") + $VerilogFiles
& iverilog $CompileArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Compilation failed" -ForegroundColor Red
    exit 1
}

Write-Host "Compilation successful!" -ForegroundColor Green
Write-Host ""

# Run simulation
Write-Host "Running simulation..." -ForegroundColor Green
Write-Host ""
& vvp $OutputVVP

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "WARNING: Simulation may have encountered issues" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Simulation complete!" -ForegroundColor Green

# Check for VCD file
if (Test-Path $OutputVCD) {
    $vcdSize = (Get-Item $OutputVCD).Length / 1KB
    Write-Host "VCD waveform file: $OutputVCD ($([math]::Round($vcdSize, 2)) KB)" -ForegroundColor Cyan
    
    if ($View) {
        Write-Host "Opening waveform viewer..." -ForegroundColor Green
        if (Get-Command "gtkwave" -ErrorAction SilentlyContinue) {
            & gtkwave $OutputVCD
        } else {
            Write-Host "GTKWave not found. Install it to view waveforms." -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "No VCD file generated" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
