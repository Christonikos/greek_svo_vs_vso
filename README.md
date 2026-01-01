# Installation

To install the packages, use the `env.yaml` file:

```bash
conda env create -f env.yaml
conda activate greek_movement
```

Or if you have any issues with the versions, try installing the packages manually:

```bash
# Create conda environment and pip install packages
conda create -n greek_movement python=3.12
conda activate greek_movement

pip install ipdb transformers accelerate \
    torch \
    matplotlib \
    numpy \
    pandas \
    scipy \
    seaborn \
    pip \
    spyder-kernels \
    spyder \
    ipykernel \
    tqdm \
    scikit-learn \
    pyarrow \
    python-dateutil \
    pytz \
    six \
    tzdata \
    mne \
    prettytable
```

# Usage

## Step 1: Generate Stimuli

The first step is to generate the Greek sentence stimuli used in the experiment. The stimuli consist of relative clause sentences that test Subject-Verb-Object (SVO) vs Verb-Subject-Object (VSO) word order preferences in Greek.

### Running the Script

```bash
cd src/generate_stimuli
python stimuli_creation.py
```

**Command-line Arguments:**
- `--num_cells`: Number of experimental cells (default: 4)
- `--presentation_time`: Word presentation time in milliseconds (default: 200)
- `--soa`: Stimulus Onset Asynchrony in milliseconds (default: 366)
- `--question_display_time`: Question display time in milliseconds (default: 200)
- `--response_time`: Response time window in milliseconds (default: 500)
- `--output_dir`: Output directory for stimuli CSV file (default: `../../stimuli`)
- `--output_filename`: Output CSV filename (default: `greek_sentences.csv`)

**Example with custom parameters:**

```bash
python stimuli_creation.py --num_cells 8 --presentation_time 300 --soa 400 --output_filename custom_stimuli.csv
```

This will generate a CSV file containing all experimental sentences at the specified output location.

### Sentence Structure

The script generates sentences following two possible patterns:

1. **SVO order**: `Det N1 που Det N2 V_transitive V_intransitive.`
   - Example: *"Ο αθλητής που ο γυμναστής θαυμάζει φεύγει."* (The athlete that the coach admires leaves.)

2. **VSO order**: `Det N1 που V_transitive Det N2 V_intransitive.`
   - Example: *"Ο αθλητής που θαυμάζει ο γυμναστής φεύγει."* (The athlete that admires the coach leaves.)

Each sentence is followed by a comprehension question targeting either the transitive verb (V1) or intransitive verb (V2):
- *"Ποιος θαυμάζει?"* (Who admires?) → V1 question
- *"Ποιος φεύγει?"* (Who leaves?) → V2 question

### Counterbalancing

The experiment is fully counterbalanced across multiple dimensions:

1. **Word Order**: Equal number of SVO and VSO sentences
2. **Question Target**: Equal number of V1 (transitive) and V2 (intransitive) questions
3. **Number**: Equal number of singular and plural noun phrases
4. **Gender**: Balanced masculine and feminine nouns for both N1 and N2
5. **Gender Congruency**: All combinations of M-M, F-F, M-F, F-M
6. **Lexical Diversity**: N1 and N2 always have different stems (no repetition)


### Lexicon

The Greek lexicon is defined in `src/generate_stimuli/lexicon.py` and includes:
- **Determiners**: ο/η (singular), οι (plural)
- **Nouns**: 16 human nouns per gender/number combination (e.g., δάσκαλος, μαθητής, δασκάλα, μαθήτρια)
- **Transitive verbs**: 7 verbs per number (e.g., συμπαθεί, θαυμάζει, αγαπάει)
- **Intransitive verbs**: 6 verbs per number (e.g., φεύγει, κλαίει, γελάει)

### Output

The generated CSV file contains the following columns:
- `Sentence`: The full Greek sentence
- `Question`: The comprehension question
- `order`: SVO or VSO
- `question_for`: V1 or V2
- `N1_number`: sing or plur
- `N1_gender`: m or f
- `N2_gender`: m or f
- `correct_response`: The correct answer to the question
- `false_response`: The incorrect distractor answer

With `num_cells=4`, the script generates **128 unique sentences** (32 conditions × 4 cells).

## Step 2: Extract Model Activations

After generating the stimuli, extract hidden layer activations from the LLM by performing forward passes on each sentence.

### Running the Script

```bash
cd src
python forward_pass_and_activation_extraction.py --csv ../stimuli/greek_sentences.csv --save_dir krikri_activations --model ilsp/Llama-Krikri-8B-Base
```

### Command-line Arguments

- `--model`: HuggingFace model name (default: `ilsp/Llama-Krikri-8B-Base`)
  - Also tested with: `google/gemma-3-12b-pt`, `mistralai/Ministral-3-14B-Base-2512`
- `--csv`: Path to the stimuli CSV file (default: `../stimuli/greek_sentences.csv`)
- `--save_dir`: Directory to save activation files (default: `activations`)

The script:
1. Loads the specified model and tokenizer from HuggingFace
2. Processes each sentence through the model with `output_hidden_states=True`
3. Extracts hidden states from all layers for each token in the sentence
4. Saves activations to disk in PyTorch format with memory optimization (half precision, CPU storage)

### Output Format

For each sentence, a `.pt` file is saved containing:
- `sentence`: The original Greek sentence
- `tokens`: List of tokenized subwords
- `hidden_states`: Tuple of tensors containing activations from all layers (shape: `[num_layers+1, seq_len, hidden_size]`)
- `last_token`: Activation vector of the last token for each layer
- `mean`: Mean-pooled activation vector across all tokens for each layer

### Example Output Structure

```
src/krikri_activations/
├── sentence_0.pt
├── sentence_1.pt
├── sentence_2.pt
...
└── sentence_127.pt
```

## Step 3: Token-POS Matching

Match tokenized subwords to their corresponding parts of speech (POS) in each sentence in order to inspect the results.

### Running the Script

```bash
cd src
python token_pos_matching.py --stimuli_path ../stimuli/greek_sentences.csv --activations_path krikri_activations --output_path ../token_pos_matches.csv
```

**Command-line Arguments:**
- `--stimuli_path`: Path to stimuli CSV file (default: `../stimuli/greek_sentences.csv`)
- `--activations_path`: Path to activations directory (default: `../activations`)
- `--output_path`: Path for output CSV file (default: `../token_pos_matches.csv`)
- `--verbose`: Print detailed analysis for first few sentences (flag, default: False)
- `--n_examples`: Number of example sentences to analyze in verbose mode (default: 3)

**Example with verbose output:**
```bash
python token_pos_matching.py --activations_path krikri_activations --verbose --n_examples 5
```

The script:
1. Loads the generated sentences and their activation files
2. Identifies the position of the relative pronoun 'που' as an anchor point
3. Maps tokens to their linguistic roles based on sentence structure:
   - **Det_N1**: Determiner of the first noun (ο/η/οι)
   - **N1**: Head noun of the main clause
   - **που**: Relative pronoun
   - **Det_N2**: Determiner of the second noun
   - **N2**: Head noun of the relative clause
   - **V1**: Transitive verb
   - **V2**: Intransitive verb
4. Handles multi-token words (Greek words may be split into multiple subword tokens)
5. Stores token positions and their mappings

### Output Format

The `token_pos_matches.csv` file contains:
- **Metadata**: sentence_id, sentence, order, question_for, n1_number, n1_gender, n2_gender
- **Expected words**: det_n1_word, n1_word, det_n2_word, n2_word, v1_word, v2_word
- **Token positions**: Lists of token indices for each word (e.g., `[2, 3]` for multi-token words)
- **Token strings**: The actual tokenized subwords
- **Token counts**: Number of tokens per word

## Step 4: Generate Visualizations

After extracting activations and matching tokens to POS, generate figures showing the analysis results.

### Figure 1: Sentence Type Classification by Layer

Visualizes how well each layer can classify SVO vs VSO sentences using pre-clause vs post-clause activations, with cluster-based permutation tests.

```bash
cd src
python figure_1.py --activations_path krikri_activations --n_layers 32 --alpha 0.01
```

**Arguments:**
- `--activations_path`: Directory containing activation files (default: `krikri_activations`)
- `--n_layers`: Number of model layers (default: 32)
- `--n_folds`: Cross-validation folds (default: 20)
- `--alpha`: Significance level for cluster tests (default: 0.01)
- `--n_permutations`: Permutation test iterations (default: 5000)
- `--output_prefix`: Output filename prefix (default: `figure_1`)

**Output:**
- `figure_1.pdf` and `figure_1.png`: AUC curves showing classification performance across layers
- Horizontal bars indicate significant clusters (p < 0.01) where pre-clause or post-clause regions significantly differ from chance

**Interpretation:**
- **Post-clause (solid blue line)**: Shows when post-clause activations predict sentence type
- **Pre-clause (dashed pink line)**: Shows when pre-clause activations predict sentence type
- Higher AUC = better classification performance for that region

### Figure 2: Generalization Across Layers (GAT Matrix)

Creates a Generalization Across Time matrix showing how classifiers trained on one layer generalize to other layers.

```bash
cd src
python figure_2.py --activations_path krikri_activations --region aft --alpha 0.05
```

**Arguments:**
- `--activations_path`: Directory containing activation files (default: `krikri_activations`)
- `--n_layers`: Number of model layers (default: 32)
- `--n_folds`: Cross-validation folds (default: 20)
- `--alpha`: Significance level (default: 0.05)
- `--n_permutations`: Permutation iterations (default: 5000)
- `--region`: Region to analyze - `aft` (post-clause) or `bef` (pre-clause) (default: `aft`)
- `--output_prefix`: Output filename prefix (default: `figure_2`)

**Output:**
- GAT heatmap showing train layer (y-axis) vs test layer (x-axis) AUC values
- Red contours: Regions significantly above chance (AUC > 0.5)
- Blue contours: Regions significantly below chance (AUC < 0.5)
- Diagonal line: Perfect generalization (train = test)

**Interpretation:**
- Bright red areas: Strong generalization (classifier trained on layer X works well on layer Y)
- Dark blue areas: Inverse generalization (classifier systematically fails)
- Off-diagonal patterns reveal hierarchical structure in representations

### Figure 3: Cross-Layer Generalization Analysis

Shows systematic inversion pattern when training on late layers and testing on early layers.

```bash
cd src
python figure_3.py --activations_path krikri_activations --train_layers_start 20 --train_layers_end 32 --test_layers_start 0 --test_layers_end 20
```

**Arguments:**
- `--activations_path`: Directory containing activation files (default: `krikri_activations`)
- `--train_layers_start`: First training layer (default: 20)
- `--train_layers_end`: Last training layer (default: 32)
- `--test_layers_start`: First test layer (default: 0)
- `--test_layers_end`: Last test layer (default: 20)
- `--region`: Region to analyze - `after` or `before` (default: `after`)
- `--alpha`: Significance level (default: 0.01)
- `--output_prefix`: Output filename prefix (default: `figure_3_final`)

**Output:**
- `figure_3_final.pdf` and `figure_3_final.png`: AUC curve showing systematic below-chance performance
- Detailed tables showing classifier bias patterns
- Summary statistics of misclassification rates

**Interpretation:**
- **Below-chance AUC (< 0.5)**: Classifier systematically inverts predictions
- **Key finding**: Classifier trained on late layers (20-31) learns "SVO bias" but early layers contain different information
- Tables show percentage of VSO sentences incorrectly classified as SVO
