.PHONY: help install clean stimuli extract-krikri extract-gemma match-tokens figures figure1 figure2 figure3 all

# Default target
help:
	@echo "Greek SVO vs VSO Analysis - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make install          - Create conda environment from env.yaml"
	@echo "  make clean            - Remove generated files and caches"
	@echo "  make clean-figures    - Remove generated figures only"
	@echo "  make clean-all        - Deep clean (everything except data)"
	@echo "  make stimuli          - Generate Greek sentence stimuli"
	@echo "  make extract-krikri   - Extract activations from KriKri model"
	@echo "  make extract-gemma    - Extract activations from Gemma model (float32)"
	@echo "  make match-tokens     - Match tokens to parts of speech"
	@echo "  make figure1          - Generate Figure 1 (classification by layer)"
	@echo "  make figure2          - Generate Figure 2 (GAT matrix)"
	@echo "  make figure3          - Generate Figure 3 (cross-layer generalization)"
	@echo "  make figures          - Generate all figures"
	@echo "  make all              - Run full pipeline: stimuli -> extract -> match -> figures"
	@echo ""
	@echo "Model-specific targets:"
	@echo "  make figure1-krikri   - Figure 1 for KriKri"
	@echo "  make figure1-gemma    - Figure 1 for Gemma"
	@echo "  make figures-krikri   - All figures for KriKri"
	@echo "  make figures-gemma    - All figures for Gemma"

# Variables
PYTHON := python
CONDA_ENV := greek_movement
SRC_DIR := src
STIMULI_DIR := stimuli
ACTIVATIONS_DIR := $(SRC_DIR)
FIGURES_DIR := $(SRC_DIR)/figures

# Installation
install:
	@echo "Creating conda environment..."
	conda env create -f env.yaml
	@echo "Environment created. Activate with: conda activate $(CONDA_ENV)"

# Cleanup
clean:
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.py~" -delete
	rm -f $(SRC_DIR)/*.pdf $(SRC_DIR)/*.png
	@echo "Clean complete."

clean-figures:
	@echo "Cleaning generated figures..."
	rm -rf $(FIGURES_DIR)/*.pdf $(FIGURES_DIR)/*.png
	rm -f $(SRC_DIR)/*.pdf $(SRC_DIR)/*.png
	@echo "Figures cleaned."

clean-all: clean clean-figures
	@echo "Deep clean complete (excluding data files)."

# Step 1: Generate stimuli
stimuli:
	@echo "Generating Greek sentence stimuli..."
	cd $(SRC_DIR)/generate_stimuli && $(PYTHON) stimuli_creation.py
	@echo "Stimuli generated in $(STIMULI_DIR)/greek_sentences.csv"

# Step 2: Extract activations
extract-krikri:
	@echo "Extracting activations from KriKri model..."
	cd $(SRC_DIR) && $(PYTHON) forward_pass_and_activation_extraction.py \
		--model ilsp/Llama-Krikri-8B-Base \
		--csv ../$(STIMULI_DIR)/greek_sentences.csv \
		--save_dir krikri_activations \
		--precision float16
	@echo "KriKri activations saved to $(SRC_DIR)/krikri_activations/"

extract-gemma:
	@echo "Extracting activations from Gemma model (float32)..."
	cd $(SRC_DIR) && $(PYTHON) forward_pass_and_activation_extraction.py \
		--model google/gemma-3-12b-pt \
		--csv ../$(STIMULI_DIR)/greek_sentences.csv \
		--save_dir gemma3_12b_activations_f32 \
		--precision float32
	@echo "Gemma activations saved to $(SRC_DIR)/gemma3_12b_activations_f32/"

# Step 3: Token-POS matching
match-tokens:
	@echo "Matching tokens to parts of speech..."
	cd $(SRC_DIR) && $(PYTHON) token_pos_matching.py \
		--stimuli-path ../$(STIMULI_DIR)/greek_sentences.csv \
		--activations-path krikri_activations \
		--output-path ../token_pos_matches.csv
	@echo "Token matches saved to token_pos_matches.csv"

match-tokens-gemma:
	@echo "Matching tokens to parts of speech (Gemma)..."
	cd $(SRC_DIR) && $(PYTHON) token_pos_matching.py \
		--stimuli-path ../$(STIMULI_DIR)/greek_sentences.csv \
		--activations-path gemma3_12b_activations_f32 \
		--output-path ../token_pos_matches_gemma.csv
	@echo "Token matches saved to token_pos_matches_gemma.csv"

# Step 4: Generate figures
figure1:
	@echo "Generating Figure 1..."
	cd $(SRC_DIR) && $(PYTHON) figure_1.py --model krikri
	@echo "Figure 1 saved to $(FIGURES_DIR)/"

figure1-krikri:
	@echo "Generating Figure 1 (KriKri)..."
	cd $(SRC_DIR) && $(PYTHON) figure_1.py --model krikri
	@echo "Figure 1 (KriKri) saved to $(FIGURES_DIR)/figure_1_krikri.pdf"

figure1-gemma:
	@echo "Generating Figure 1 (Gemma)..."
	cd $(SRC_DIR) && $(PYTHON) figure_1.py --model gemma
	@echo "Figure 1 (Gemma) saved to $(FIGURES_DIR)/figure_1_gemma.pdf"

figure2:
	@echo "Generating Figure 2 (GAT matrix)..."
	cd $(SRC_DIR) && $(PYTHON) figure_2.py \
		--activations_path krikri_activations \
		--n_layers 32 \
		--region aft
	@echo "Figure 2 saved to $(FIGURES_DIR)/"

figure2-krikri:
	@echo "Generating Figure 2 (KriKri)..."
	cd $(SRC_DIR) && $(PYTHON) figure_2.py \
		--activations_path krikri_activations \
		--n_layers 32 \
		--region aft \
		--output_prefix figure_2_krikri
	@echo "Figure 2 (KriKri) saved to $(FIGURES_DIR)/figure_2_krikri.pdf"

figure2-gemma:
	@echo "Generating Figure 2 (Gemma)..."
	cd $(SRC_DIR) && $(PYTHON) figure_2.py \
		--activations_path gemma3_12b_activations_f32 \
		--n_layers 48 \
		--region aft \
		--output_prefix figure_2_gemma
	@echo "Figure 2 (Gemma) saved to $(FIGURES_DIR)/figure_2_gemma.pdf"

figure3:
	@echo "Generating Figure 3 (cross-layer generalization)..."
	cd $(SRC_DIR) && $(PYTHON) figure_3.py \
		--activations_path krikri_activations \
		--train_layers_start 20 \
		--train_layers_end 32 \
		--test_layers_start 0 \
		--test_layers_end 20
	@echo "Figure 3 saved to $(FIGURES_DIR)/"

figure3-krikri:
	@echo "Generating Figure 3 (KriKri)..."
	cd $(SRC_DIR) && $(PYTHON) figure_3.py \
		--activations_path krikri_activations \
		--train_layers_start 20 \
		--train_layers_end 32 \
		--test_layers_start 0 \
		--test_layers_end 20 \
		--output_prefix figure_3_krikri
	@echo "Figure 3 (KriKri) saved to $(FIGURES_DIR)/figure_3_krikri.pdf"

figure3-gemma:
	@echo "Generating Figure 3 (Gemma)..."
	cd $(SRC_DIR) && $(PYTHON) figure_3.py \
		--activations_path gemma3_12b_activations_f32 \
		--train_layers_start 30 \
		--train_layers_end 48 \
		--test_layers_start 0 \
		--test_layers_end 30 \
		--output_prefix figure_3_gemma
	@echo "Figure 3 (Gemma) saved to $(FIGURES_DIR)/figure_3_gemma.pdf"

figures: figure1 figure2 figure3
	@echo "All figures generated!"

figures-krikri: figure1-krikri figure2-krikri figure3-krikri
	@echo "All KriKri figures generated!"

figures-gemma: figure1-gemma figure2-gemma figure3-gemma
	@echo "All Gemma figures generated!"

# Full pipeline
all: stimuli extract-krikri match-tokens figures
	@echo "Full pipeline complete!"

all-krikri: stimuli extract-krikri match-tokens figures-krikri
	@echo "Full KriKri pipeline complete!"

all-gemma: stimuli extract-gemma match-tokens-gemma figures-gemma
	@echo "Full Gemma pipeline complete!"

