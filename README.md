# Ghostbuster (Local Enhanced Version)

This project is a robust, locally enhanced fork of [vivek3141/ghostbuster](https://github.com/vivek3141/ghostbuster), designed for reliable and user-friendly AI-generated text detection. Key features include:

## Main Features & Enhancements
- Local batch generation of logprobs (GPT-Neo-125M, GPT2) feature files, compatible with multiple dataset structures.
- Automated symbolic feature expression generation and batch feature extraction.
- Robust data processing: feature selection, feature/label alignment, outlier filtering, and feature name saving.
- Training pipeline automatically filters invalid features and saves normalization parameters, feature names, and models.
- Inference scripts support single-file, batch, and interactive (with PyQt5 GUI) detection.
- Evaluation script for batch testing and accuracy statistics on all samples in `test_files/`.
- All script/file/feature naming conventions are standardized and compatible with both new and legacy data.
- All debug output removed and warnings suppressed for production use.

## Environment
- Recommended: Python 3.10
- Dependencies: see `requirements.txt`
- Suggested setup:
  ```
  conda create -n ghostbuster python=3.10
  conda activate ghostbuster
  pip install -r requirements.txt
  pip install -e .
  ```
- Download required NLTK data (must run before first use!):
  ```python
  import nltk
  nltk.download(['brown', 'wordnet', 'omw-1.4', 'punkt'])
  ```

## Data Sources & Logprobs Generation

The following datasets have been batch-processed to generate logprobs feature files (GPT2 and GPT-Neo-125M):

- ghostbuster-data/wp/human
- ghostbuster-data/wp/gpt
- ghostbuster-data/reuter/human
- ghostbuster-data/reuter/gpt
- ghostbuster-data/essay/human
- ghostbuster-data/essay/gpt

All raw texts in these directories have corresponding logprobs feature files, ready for feature extraction and model training.

## Dataset Size & Training Results

- Total samples: 5994
- Training set size: 4795
- Validation set size: 1199
- Positive labels (AI-generated): 3000

Example train/validation split (indices):
Train/Test Split [5728 2154 2406 ... 3436 4256 5801] [4871 5373 5974 ... 1653 2607 2732]

Model performance on the validation set:

|   F1   | Accuracy |  AUC  |
|--------|----------|-------|
| 0.833  |  0.822   | 0.907 |

The model demonstrates strong discrimination between AI-generated and human-written text, suitable for practical use and further extension.

## Training Hardware & Time Consumption

All experiments were conducted on the following hardware:

- GPU: 1 × NVIDIA RTX 4090 (25.2 GB VRAM)
- CPU: 16-core AMD EPYC 9354
- RAM: 60.1 GB
- Storage: 751.6 GB SSD

Typical running times for key steps:

- `python generate.py --logprobs`: ~1 hour
- `python train.py --generate_symbolic_data`: ~2 hours
- `python train.py --perform_feature_selection`: ~10 hours

Note: Feature selection is the most time-consuming step due to the combinatorial search over symbolic feature expressions.

## Workflow

### 1. Generate logprobs
Batch-generate logprobs feature files (GPT-Neo-125M, GPT2) for all raw texts:
```bash
python generate.py --logprobs
```

### 2. Generate symbolic features
Extract all symbolic feature expressions and generate feature data in batch:
```bash
python train.py --generate_symbolic_data
```

### 3. Feature selection
Automatically select the best symbolic feature combinations:
```bash
python train.py --perform_feature_selection
```

### 4. Train and save the model
First generate labels, then train the model:
```bash
python gen_labels.py
python train.py --train_on_all_data
```
Model, feature names, normalization parameters, etc. will be saved in the `model/` directory.

### 5. Inference & Testing
- Single file inference:
  ```bash
  python classify.py --file input.txt
  ```
- Interactive detection (CLI/GUI):
  ```bash
  python interactive_detect.py
  ```

### 6. Batch evaluation
Batch evaluate all samples in `test_files/` and report accuracy:
```bash
python eval_test_files.py
```

## Main Scripts
- `generate.py`: Batch logprobs feature generation.
- `train.py`: Symbolic feature generation, feature selection, and model training.
- `gen_labels.py`: Generate `labels.npy` label file.
- `classify.py`: Single-file inference script.
- `interactive_detect.py`: Interactive inference (with PyQt5 GUI).
- `eval_test_files.py`: Batch evaluation script.
- `utils/`: Feature engineering, symbolic features, data loading, and utility functions.

## Key Differences from the Original
- Fully local workflow, no OpenAI API required.
- Multi-model logprobs features with unified naming.
- Strict alignment of data/features/labels for robustness.
- Enhanced inference, evaluation, and interactive experience.
- Cleaner code structure, easier to extend and maintain.

## Advanced Training Options & Extensibility

The codebase retains flexible training options for advanced users:

- You can specify different data splits, feature selection strategies, or model hyperparameters by modifying the training scripts or passing additional arguments.
- The pipeline supports adding more datasets from `ghostbuster-data/` or any other directory containing human- or AI-generated text. This enables training with larger, more diverse, or domain-specific corpora.
- You can easily extend the system to use richer or more challenging AI/human text sources for improved robustness.
- The logprobs generation step (`generate.py --logprobs`) supports swapping in different language models (e.g., other HuggingFace models) by adjusting the model loading code, allowing for experimentation with new detectors or ensemble approaches.

For details on customizing datasets, features, or models, see the scripts and comments in the `utils/` directory.

## Optional Training Commands

The following advanced training commands are available for flexible experimentation:

- `python train.py --generate_symbolic_data`               # Generate symbolic features (depth 3)
- `python train.py --generate_symbolic_data_four`            # Generate symbolic features (depth 4, more complex)
- `python train.py --generate_symbolic_data_eval`            # Generate symbolic features for evaluation datasets
- `python train.py --perform_feature_selection`             # Feature selection (default, depth 3)
- `python train.py --perform_feature_selection_one`          # Feature selection (max 2 tokens per feature)
- `python train.py --perform_feature_selection_two`          # Feature selection (max 4 tokens per feature)
- `python train.py --perform_feature_selection_four`         # Feature selection (on depth 4 features)
- `python train.py --perform_feature_selection_no_gpt`        # Feature selection (exclude GPT/Neo logprobs)
- `python train.py --perform_feature_selection_only_ada`      # Feature selection (only GPT2 logprobs)
- `python train.py --perform_feature_selection_domain`        # Feature selection for each domain (WP, Reuters, Essay)
- `python train.py --only_include_gpt`                   # Only include GPT-generated data for training
- `python train.py --train_on_all_data`                  # Train and save model on all data

You can combine these options as needed for custom experiments. See script comments for more details.

---
For custom datasets or feature extensions, see scripts in the `utils/` directory.