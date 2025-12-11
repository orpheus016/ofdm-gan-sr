# ==============================================================================
# CWGAN-GP RTL Simulation Script for Windows
# 
# Usage (PowerShell):
#   .\simulate.ps1 -Target gen      # Simulate generator only
#   .\simulate.ps1 -Target full     # Simulate full system
#   .\simulate.ps1 -Target all      # Run all simulations
#   .\simulate.ps1 -Target clean    # Clean generated files
# ==============================================================================

param(
    [Parameter()]
    [ValidateSet("gen", "full", "all", "clean")]
    [string]$Target = "all"
)

# Tool paths (adjust as needed)
$IVERILOG = "iverilog"
$VVP = "vvp"

# Source files
$GEN_SRC = @(
    "conv1d_engine.v",
    "activation_lrelu.v",
    "activation_tanh.v",
    "upsample_nn.v",
    "weight_rom.v",
    "generator_mini.v"
)

$FULL_SRC = @(
    "conv1d_engine.v",
    "activation_lrelu.v",
    "activation_tanh.v",
    "upsample_nn.v",
    "sum_pool.v",
    "weight_rom.v",
    "generator_mini.v",
    "discriminator_mini.v",
    "cwgan_gp_top.v"
)

function Simulate-Generator {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Simulating Generator" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    $sources = ($GEN_SRC + "tb_generator_mini.v") -join " "
    
    Write-Host "Compiling..." -ForegroundColor Yellow
    $compileCmd = "$IVERILOG -g2012 -o tb_generator_mini.vvp $sources"
    Invoke-Expression $compileCmd
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Running simulation..." -ForegroundColor Yellow
        Invoke-Expression "$VVP tb_generator_mini.vvp"
        Write-Host "Generator simulation complete!" -ForegroundColor Green
    } else {
        Write-Host "Compilation failed!" -ForegroundColor Red
    }
}

function Simulate-Full {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Simulating Full CWGAN-GP" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    $sources = ($FULL_SRC + "tb_cwgan_gp.v") -join " "
    
    Write-Host "Compiling..." -ForegroundColor Yellow
    $compileCmd = "$IVERILOG -g2012 -o tb_cwgan_gp.vvp $sources"
    Invoke-Expression $compileCmd
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Running simulation..." -ForegroundColor Yellow
        Invoke-Expression "$VVP tb_cwgan_gp.vvp"
        Write-Host "Full system simulation complete!" -ForegroundColor Green
    } else {
        Write-Host "Compilation failed!" -ForegroundColor Red
    }
}

function Clean-Files {
    Write-Host "Cleaning generated files..." -ForegroundColor Yellow
    Remove-Item -Path "*.vvp" -ErrorAction SilentlyContinue
    Remove-Item -Path "*.vcd" -ErrorAction SilentlyContinue
    Remove-Item -Path "*.out" -ErrorAction SilentlyContinue
    Remove-Item -Path "*.log" -ErrorAction SilentlyContinue
    Write-Host "Clean complete!" -ForegroundColor Green
}

# Main
switch ($Target) {
    "gen" {
        Simulate-Generator
    }
    "full" {
        Simulate-Full
    }
    "all" {
        Simulate-Generator
        Write-Host ""
        Simulate-Full
    }
    "clean" {
        Clean-Files
    }
}
