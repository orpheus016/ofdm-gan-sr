# ==============================================================================
# Simple GAN RTL - Makefile
#
# Build and simulate the simple 3x3 GAN for circle/cross pattern generation
# 
# Architecture:
#   Generator:     latent(2) -> hidden(3) -> output(9)
#   Discriminator: input(9) -> hidden(3) -> output(1)
#
# Usage:
#   make sim       - Run full testbench simulation
#   make clean     - Remove generated files
#   make view      - View waveform in GTKWave
# ==============================================================================

# Verilog compiler (Icarus Verilog)
IVERILOG = iverilog
VVP = vvp
GTKWAVE = gtkwave

# Source files
TANH_SRC = ../activation_tanh.v
SRC_FILES = \
	activation_sigmoid.v \
	simple_gan_weights.v \
	simple_generator.v \
	simple_discriminator.v \
	simple_gan_top.v

# Testbench
TB_FILE = tb_simple_gan.v
TB_OUT = tb_simple_gan.vvp
VCD_FILE = tb_simple_gan.vcd

# Include path for parent directory modules
INCLUDE = -I..

# ==============================================================================
# Targets
# ==============================================================================

.PHONY: all sim clean view help

all: sim

# Compile and run simulation
sim: $(TB_OUT)
	@echo "========================================="
	@echo "Running Simple GAN Simulation"
	@echo "========================================="
	$(VVP) $(TB_OUT)

# Compile testbench (using SystemVerilog 2012 for unpacked arrays)
$(TB_OUT): $(SRC_FILES) $(TB_FILE) $(TANH_SRC)
	@echo "Compiling Simple GAN RTL..."
	$(IVERILOG) -g2012 -o $(TB_OUT) $(INCLUDE) $(TANH_SRC) $(SRC_FILES) $(TB_FILE)

# View waveform
view: $(VCD_FILE)
	$(GTKWAVE) $(VCD_FILE) &

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -f $(TB_OUT) $(VCD_FILE)
	rm -f *.vcd *.vvp

# Help
help:
	@echo "Simple GAN RTL Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  sim     - Compile and run simulation (default)"
	@echo "  view    - Open waveform in GTKWave"
	@echo "  clean   - Remove generated files"
	@echo "  help    - Show this help message"
	@echo ""
	@echo "Architecture:"
	@echo "  Generator:     2 -> 3 -> 9 (latent -> hidden -> 3x3 image)"
	@echo "  Discriminator: 9 -> 3 -> 1 (3x3 image -> hidden -> score)"
	@echo ""
	@echo "Files:"
	@echo "  activation_sigmoid.v    - Sigmoid activation (LUT-based)"
	@echo "  simple_generator.v      - Generator with 2 dense layers"
	@echo "  simple_discriminator.v  - Discriminator with 2 dense layers"
	@echo "  simple_gan_weights.v    - Weight ROM with initial values"
	@echo "  simple_gan_top.v        - Top-level GAN module"
	@echo "  tb_simple_gan.v         - Testbench"
