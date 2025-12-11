# ==============================================================================
# CWGAN-GP RTL Simulation Makefile
#
# Targets:
#   make sim_gen     - Simulate generator only
#   make sim_full    - Simulate full CWGAN-GP
#   make sim_all     - Run all simulations
#   make wave_gen    - View generator waveforms
#   make wave_full   - View full system waveforms
#   make clean       - Clean generated files
#   make lint        - Run Verilator lint check
# ==============================================================================

# Tools (configure paths as needed)
IVERILOG = iverilog
VVP = vvp
GTKWAVE = gtkwave
VERILATOR = verilator

# Source files - Generator
GEN_SRC = \
    conv1d_engine.v \
    activation_lrelu.v \
    activation_tanh.v \
    upsample_nn.v \
    weight_rom.v \
    generator_mini.v

# Source files - Discriminator
DISC_SRC = \
    conv1d_engine.v \
    activation_lrelu.v \
    sum_pool.v \
    weight_rom.v \
    discriminator_mini.v

# Source files - Full system
FULL_SRC = \
    conv1d_engine.v \
    activation_lrelu.v \
    activation_tanh.v \
    upsample_nn.v \
    sum_pool.v \
    weight_rom.v \
    generator_mini.v \
    discriminator_mini.v \
    cwgan_gp_top.v

# Testbenches
TB_GEN = tb_generator_mini.v
TB_FULL = tb_cwgan_gp.v

# Output files
VVP_GEN = tb_generator_mini.vvp
VVP_FULL = tb_cwgan_gp.vvp
VCD_GEN = tb_generator_mini.vcd
VCD_FULL = tb_cwgan_gp.vcd

# Iverilog flags
IVFLAGS = -g2012 -Wall

# ==============================================================================
# Simulation Targets
# ==============================================================================

.PHONY: all sim_gen sim_full sim_all wave_gen wave_full clean lint

all: sim_all

# Generator simulation
sim_gen: $(VCD_GEN)
	@echo "Generator simulation complete. View with: make wave_gen"

$(VVP_GEN): $(GEN_SRC) $(TB_GEN)
	$(IVERILOG) $(IVFLAGS) -o $@ $(GEN_SRC) $(TB_GEN)

$(VCD_GEN): $(VVP_GEN)
	$(VVP) $<

# Full system simulation
sim_full: $(VCD_FULL)
	@echo "Full system simulation complete. View with: make wave_full"

$(VVP_FULL): $(FULL_SRC) $(TB_FULL)
	$(IVERILOG) $(IVFLAGS) -o $@ $(FULL_SRC) $(TB_FULL)

$(VCD_FULL): $(VVP_FULL)
	$(VVP) $<

# Run all simulations
sim_all: sim_gen sim_full
	@echo "All simulations complete."

# ==============================================================================
# Waveform Viewing
# ==============================================================================

wave_gen: $(VCD_GEN)
	$(GTKWAVE) $< &

wave_full: $(VCD_FULL)
	$(GTKWAVE) $< &

# ==============================================================================
# Lint Check (Verilator)
# ==============================================================================

lint:
	$(VERILATOR) --lint-only -Wall $(FULL_SRC)

# ==============================================================================
# Clean
# ==============================================================================

clean:
	rm -f *.vvp *.vcd *.out *.log

# ==============================================================================
# Help
# ==============================================================================

help:
	@echo "CWGAN-GP RTL Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  sim_gen     - Simulate generator only"
	@echo "  sim_full    - Simulate full CWGAN-GP"
	@echo "  sim_all     - Run all simulations"
	@echo "  wave_gen    - View generator waveforms (GTKWave)"
	@echo "  wave_full   - View full system waveforms (GTKWave)"
	@echo "  lint        - Run Verilator lint check"
	@echo "  clean       - Remove generated files"
	@echo "  help        - Show this help"
