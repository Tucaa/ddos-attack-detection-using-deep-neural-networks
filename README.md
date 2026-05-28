# DDoS Attack Detection Using Deep Neural Networks

A deep learning system for classifying network traffic into **9 categories** — one normal class and eight distinct DDoS attack types — using a two-layer LSTM network with an attention mechanism, trained entirely on **synthetic traffic data**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Attack Classes](#attack-classes)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [File Reference](#file-reference)
  - [attacks.py](#attackspy)
  - [normal.py](#normalpy)
  - [functions.py](#functionspy)
  - [windowing.py](#windowingpy)
  - [dataset_generator.py](#dataset_generatorpy)
  - [torch_nn.py](#torch_nnpy)
  - [hyperparam.py](#hyperparampy)
  - [metrics_exporter.py](#metrics_exporterpy)
  - [visualization.py](#visualizationpy)
  - [inference_dsgenerator.py](#inference_dsgeneratorpy)
- [Getting Started](#getting-started)
- [Key Design Decisions](#key-design-decisions)

---

## Project Overview

The pipeline has three main stages:

1. **Synthetic Data Generation** — Realistic per-window traffic feature vectors are generated for each attack type and for normal traffic, with Gaussian noise, class transition blending, and minority-class oversampling applied to produce a balanced dataset.
2. **Model Training** — A PyTorch LSTM + Attention classifier is trained on sliding-window sequences using temporal cross-validation (TimeSeriesSplit) to prevent data leakage.
3. **Evaluation** — Full metrics are computed after training: confusion matrix (absolute and normalized), per-class classification report, Matthews Correlation Coefficient (MCC), and per-class ROC-AUC curves.

---

## Attack Classes

| Label | Description |
|---|---|
| `normal` | Legitimate mixed traffic (web, enterprise, streaming, DNS, email) |
| `syn_flood` | TCP SYN flood — overwhelms connection state tables |
| `ack_flood` | TCP ACK flood — sends unsolicited ACK packets at high rate |
| `icmp_flood` | ICMP echo request flood |
| `udp_flood_large` | UDP flood with abnormally large packets |
| `udp_flood_mixed` | UDP flood across randomized destination ports |
| `dns_amplification` | DNS reflection/amplification attack using open resolvers |
| `ntp_amplification` | NTP amplification via monlist requests (port 123) |
| `subnet_carpet_bombing` | Volumetric attack spread across an entire destination subnet |

---

## Architecture

```
Input (seq_len × 22 features)
        │
   ┌────▼────┐
   │  LSTM   │  2 layers, hidden_size=128, dropout=0.3
   └────┬────┘
        │ (batch, seq_len, hidden)
   ┌────▼─────────┐
   │  Attention   │  Learned per-timestep weights → context vector
   └────┬─────────┘
        │ (batch, hidden)
   ┌────▼────────────────┐
   │  Classifier Head    │  Linear → ReLU → Linear
   └────┬────────────────┘
        │
   (batch, 9 classes)
```

**Training setup:**
- Optimizer: Adam with `lr=1e-3`
- Loss: CrossEntropyLoss (inverse-frequency class weights)
- Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)
- Validation: 5-fold TimeSeriesSplit (chronological, no shuffling)
- Gradient clipping: max norm = 1.0

---

## Project Structure

```
.
├── attacks.py               # DDoS traffic feature generators (8 attack types)
├── normal.py                # Normal traffic feature generators (5 profiles)
├── functions.py             # Shared utility functions (noise, blending, I/O)
├── windowing.py             # Time-window management and timeline construction
├── dataset_generator.py     # Full dataset assembly with oversampling
├── torch_nn.py              # Model definition, training, and evaluation
├── hyperparam.py            # Optuna-based hyperparameter tuning
├── metrics_exporter.py      # JSON metrics export for downstream analysis
├── visualization.py         # Dataset and traffic visualization plots
└── inference_dsgenerator.py # Preprocessor for real CIC-DDoS2019 data
```

---

## File Reference

---

### `attacks.py`

Defines synthetic traffic feature generators for all eight DDoS attack types. Each function returns a dictionary of 22 network features (packet rate, byte rate, protocol ratios, TCP flag ratios, IP diversity counts, entropy values, DNS ratios, subnet spread) sampled from distributions calibrated to match realistic attack signatures.

**Functions:**

- `udp_flood_large()` — Generates features for a high-volume UDP flood with abnormally large packets (~1300 bytes average) targeting a small number of destination IPs.
- `dns_amplification()` — Generates features for a DNS amplification attack, characterized by a very high `dns_response_ratio` (~0.85–0.98) and large response packets coming from many source IPs.
- `subnet_carpet_bombing()` — Generates features for a carpet bombing attack, distinguished by an extremely high `dst_ip_entropy` and `dst_subnet_spread` as traffic is spread across an entire subnet.
- `syn_flood()` — Generates features for a TCP SYN flood, characterized by `tcp_syn_ratio` near 1.0 and very small packet sizes (~54–80 bytes).
- `icmp_flood()` — Generates features for an ICMP echo request flood where `icmp_ratio` dominates (~0.90–0.99) and DNS ratios are exactly 0.
- `udp_flood_mixed()` — Generates features for a UDP flood with randomized destination ports, distinguished from `udp_flood_large` by higher `dst_port_entropy` and smaller packet sizes.
- `ntp_amplification()` — Generates features for an NTP amplification attack (port 123), with medium-large response packets and high `top_dst_port_share`.
- `ack_flood()` — Generates features for a TCP ACK flood where `tcp_ack_ratio` is near 1.0 while `tcp_syn_ratio` stays very low.

The module also exports `ATTACK_GENERATORS`, a dictionary mapping attack type names to their generator functions.

---

### `normal.py`

Defines synthetic feature generators for five distinct profiles of legitimate network traffic. All functions return the same 22-feature dictionary format as the attack generators, with the label `"normal"`.

**Functions:**

- `normal_web_traffic()` — Generates features for HTTP/HTTPS traffic: TCP-dominant, moderate packet rates, balanced SYN/ACK/FIN flags, and high `top_dst_port_share` on ports 80/443.
- `normal_enterprise_traffic()` — Generates features for mixed business traffic with higher port entropy and a broader range of source/destination IPs than pure web traffic.
- `normal_streaming_traffic()` — Generates features for media streaming: UDP-dominant, large stable packets, limited destination IP diversity (few servers), and high byte rate.
- `normal_dns_traffic()` — Generates features for DNS resolver traffic, characterized by balanced `dns_query_ratio` and `dns_response_ratio` (~0.45–0.55 each) and very high `top_dst_port_share` on port 53.
- `normal_email_traffic()` — Generates features for SMTP/email traffic: TCP-dominant, low packet rate, and moderate port concentration on ports 25/465/587.
- `normal_mixed_traffic()` — Randomly selects one of the five profiles above using weighted probabilities (web 35%, enterprise 25%, streaming 20%, DNS 12%, email 8%) to simulate realistic mixed traffic.

---

### `functions.py`

Core utility module shared across the entire project. Contains mathematical helpers, noise injection, data blending, oversampling, class weight computation, CSV I/O, and timestamp formatting.

**Functions:**

- `rand_uniform(min_val, max_val)` — Returns a uniformly sampled float between the given bounds.
- `rand_normal(mean, std)` — Returns a normally distributed float using the Box-Muller transformation (no external dependency).
- `clamp(x, low, high)` — Clamps a value to the `[low, high]` range.
- `write_csv(dataset, filename)` — Writes a list of feature dictionaries to a CSV file, creating the header from dictionary keys.
- `format_timestamp(ms)` — Converts a Unix millisecond timestamp to an ISO 8601 string in UTC.
- `add_window_metadata(sample, window_id, timestamp, active_atk)` — Attaches time-series metadata fields (`window_id`, `timestamp`, `ts_formated`, `attack_active`) to a feature dictionary.
- `compute_class_weights(dataset, labels)` — Computes inverse-frequency weights per class and prints a summary table; used to balance loss during training.
- `get_class_weights_tensor(dataset, labels, device)` — Wraps `compute_class_weights` and returns a PyTorch tensor ordered to match the label encoder's class order.
- `oversample_minority_classes(dataset, labels, target_ratio)` — Oversamples attack classes so that each has at least `target_ratio * count(normal)` samples, adding light Gaussian noise to each duplicated sample.
- `add_noise(sample, noise_level)` — Adds Gaussian noise to all numeric features in a sample, clipping ratio features to `[0, 1]` and keeping volumetric features non-negative.
- `blend_samples(sample_a, sample_b, alpha)` — Linearly interpolates between two feature dictionaries; used to create smooth transitions between traffic types.
- `generate_transition(from_fn, to_fn, from_label, to_label, n_windows, start_ts, window_ms)` — Generates a transition period of `n_windows` windows blending from one traffic type to another, with the label switching at the halfway point.

---

### `windowing.py`

Manages the temporal structure of the dataset. Defines realistic attack duration profiles and provides functions to generate time-stamped windows, wave patterns, and multi-attack timelines.

**Data:**

- `ATTACK_PATTERN_DURATION` — Dictionary mapping each attack type (and `"normal"`) to duration parameters: `min_duration`, `max_duration`, `typical_duration`, `long_attack_prob`, and `long_duration_range` (all in seconds).

**Functions:**

- `format_timestamp(ms)` — Converts milliseconds to a UTC ISO 8601 string (also present in `functions.py`; kept here for module independence).
- `add_window_metadata(sample, window_id, timestamp, active_atk)` — Attaches temporal metadata to a feature sample (mirrors the version in `functions.py`).
- `define_duration(attack_type)` — Samples a realistic attack duration in seconds for the given attack type, including a probability-based chance of a long-running attack.
- `waves(attack_type)` — Randomly decides whether an attack occurs in multiple waves, based on per-attack-type probabilities.
- `wave_pattern(total_dur, wave_num)` — Divides a total attack duration into `wave_num` active sub-periods, each separated by a ~30% quiet gap.
- `generate_attack_windows(attack_fn, attack_type, start_timestamp, windows_ms)` — Generates all time windows for a single attack instance, applying wave patterns and a 10% random chance of marking a window as inactive even during active waves.
- `generate_windows_normal(attack_fn, n, start_timestamp, window_ms)` — Generates `n` consecutive windows of normal traffic, all marked as `attack_active=1`.
- `generate_attack_vector(attack_fn, attack_type, instances, duration_hours, start_timestamp, window_ms)` — Spreads multiple attack instances evenly across a given duration, tagging each window with an `instance_id`.
- `generate_timeline(window_ms, *attack_specs)` — Assembles a sequential, non-overlapping timeline from a mix of `"attack"`, `"vector"`, and `"normal"` specs, inserting a random 5–15 minute quiet period between each block.

---

### `dataset_generator.py`

Top-level orchestrator that combines all generators into a complete, balanced dataset. Handles the full day-by-day generation loop, transition insertion, noise application, and oversampling.

**Key constants:**

- `SAMPLES_PER_CLASS = 150` — Number of clean attack samples generated per class per iteration.
- `TRANSITION_WINDOWS = 20` — Number of blended transition windows added on each side of an attack block.
- `ALL_ATTACK_CONFIGS` — Ordered list of `(label, generator_fn)` pairs for all eight attack types.

**Functions:**

- `generate_mixed_dataset(window_ms, days)` — Main generation function: iterates day by day, shuffles attack class order each iteration, builds a timeline, generates balanced oversampled attack blocks, adds extra normal traffic, applies noise throughout, and runs `oversample_minority_classes` at the end.
- `generate_mixed_dataset_old(window_ms, days)` — Legacy version of the generator without oversampling and with a fixed attack ordering; kept for reference.
- `handle_generate()` — Entry point for interactive use; prompts for output filename and number of days, then calls `generate_mixed_dataset` and writes to CSV.

---

### `torch_nn.py`

Defines the neural network architectures, data preparation pipeline, training loop, cross-validation, and evaluation plots.

**Constants:**

- `SEQUENCE_LEN = 50`, `BATCH_SIZE = 64`, `HIDDEN_SIZE = 128`, `NUM_LAYERS = 2`, `DROPOUT = 0.3`, `LEARNING_RATE = 1e-3`, `EPOCHS = 20`, `N_SPLITS = 5`
- `LABELS` — Ordered list of 9 class names matching the label encoder.
- `EXCLUDED_COLS` — Metadata columns excluded from feature input.

**Classes:**

- `DDoSDataset(Dataset)` — PyTorch Dataset wrapper that converts NumPy sequence/label arrays into typed tensors; supports `__len__` and `__getitem__`.
- `AttentionLayer(nn.Module)` — Additive attention layer that learns a scalar score per LSTM timestep, applies softmax, and returns a weighted context vector along with the attention weights for interpretability.
- `DDoSLSTM(nn.Module)` — Baseline LSTM model (without attention) using only the last hidden state for classification; included as a simpler reference architecture.
- `DDoSLSTMAttention(nn.Module)` — Primary model: stacked LSTM feeding into `AttentionLayer`, followed by a two-layer classifier head (Linear → ReLU → Linear); `forward` returns both logits and attention weights.

**Functions:**

- `prepare_data(csv_path, seq_len)` — Loads the CSV, sorts by timestamp, StandardScaler-normalizes features, and creates overlapping sliding-window sequences of length `seq_len`.
- `make_dataloaders(csv_path, seq_len, batch_size)` — Wraps `prepare_data` and splits sequences into train/val/test loaders using a 60/20/20 temporal split.
- `singular_epoch(model, loader, optimizer, criterion, device)` — Runs one full training epoch with gradient clipping at norm 1.0, returning average loss and accuracy.
- `evaluate(model, loader, criterion, device)` — Runs inference-only evaluation and returns average loss and accuracy (lightweight, no plots).
- `evaluate_full(model, loader, criterion, device, label_names)` — Full evaluation pass that computes loss, accuracy, classification report, MCC, macro ROC-AUC, and generates confusion matrix and ROC curve plots.
- `plot_confusion_matrix(y_true, y_pred, label_names)` — Renders and saves side-by-side absolute and normalized confusion matrices as `confusion_matrix.png`.
- `plot_roc_curves(y_bin, all_probs, label_names)` — Renders and saves per-class ROC curves (one-vs-rest) with AUC values in the legend as `roc_curves.png`.
- `train(csv_path, save_path)` — Single-run training loop for the baseline `DDoSLSTM` model with class-weighted loss; saves the best checkpoint by validation loss.
- `train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, device, fold)` — Trains one fold of cross-validation and returns the metrics dict (loss, accuracy, MCC, predictions, probabilities) for the best epoch within that fold.
- `cross_validate(csv_path, save_path)` — Main training entry point: runs 5-fold TimeSeriesSplit cross-validation on `DDoSLSTMAttention`, saves the globally best model checkpoint, prints a fold-by-fold summary table, and exports full metrics to JSON via `metrics_exporter`.

---

### `hyperparam.py`

Optuna-based two-phase hyperparameter tuning pipeline. Phase 1 runs fast trials on 30% of the data; Phase 2 runs full cross-validation on the best configuration found.

**Constants:**

- `HYPERPARAMETER_SPACE` — Defines the search space: `hidden_size` in `[64, 128, 256]`, `num_layers` in `[1, 2]`, `dropout` in `(0.1, 0.4)`, `learning_rate` in `(1e-4, 1e-2)`, `seq_len` in `[20, 30, 50]`.
- `N_TRIALS = 30`, `TUNING_EPOCHS = 5`, `TUNING_DATA_RATIO = 0.3`

**Functions:**

- `objective_fast(trial, precomputed, num_features, num_classes, device)` — Optuna objective function: samples hyperparameters, trains for `TUNING_EPOCHS` on a single temporal split, supports early pruning via `MedianPruner`.
- `precompute_all_sequences(csv_path, seq_lens)` — Loads the CSV once and pre-builds sliding-window sequences for every candidate `seq_len`, avoiding redundant work inside each Optuna trial.
- `_evaluate_loss(model, loader, criterion, device)` — Computes average validation loss over a DataLoader without gradient tracking.
- `_print_trial_summary(trial)` — Prints a one-line summary of a completed Optuna trial including all sampled parameters and the resulting loss.
- `run_hyperparameter_tuning(csv_path, results_path)` — Orchestrates both phases: runs Optuna study with TPE sampler and median pruning, then calls `cross_validate` with the best parameters and saves results to a JSON file.
- `_plot_optuna_results(study)` — Generates and saves three Optuna diagnostic plots: optimization history, hyperparameter importance, and parallel coordinate chart.
- `train_best_model(csv_path, params_path, save_path)` — Loads a previously saved `best_hyperparams.json` and re-runs full cross-validation with those parameters.

---

### `metrics_exporter.py`

Serializes all evaluation metrics to a structured JSON file for storage, comparison, or downstream consumption (e.g. a LangGraph analysis agent).

**Functions:**

- `export_metrics_to_json(y_true, y_pred, y_proba, class_labels, output_dir, filename)` — Computes the full metrics suite (classification report, confusion matrix, MCC, per-class ROC-AUC, macro/weighted F1) and writes everything to a timestamped JSON file; returns the absolute path to the saved file.
- `_compute_roc_auc_per_class(y_true, y_proba, class_labels)` — Calculates ROC-AUC for each class using the one-vs-rest strategy, returning `0.0` for any class absent from `y_true` instead of raising an exception.
- `load_metrics_from_json(path)` — Reads a previously exported metrics JSON file and returns it as a Python dictionary, returning `None` if the file is missing or malformed.

---

### `visualization.py`

Memory-efficient dataset visualization module. Reads large CSVs in chunks and aggregates data before plotting to avoid loading the entire file into RAM.

**Constants:**

- `CLASS_COLORS` — Dictionary mapping each class label to a distinct hex color used consistently across all plots.
- `NEEDED_COLS` — The subset of CSV columns loaded during visualization (timestamp, label, and six key metrics).

**Functions:**

- `load_dataset(csv_path, resample)` — Reads the CSV in 200k-row chunks, aggregates each chunk by time interval and label, then merges all chunks into a single small resampled DataFrame alongside a class count Series.
- `plot_traffic_timeline(df_resampled, metric, title, save_path)` — Scatter plot of a chosen metric over time, with each class in its own color.
- `plot_multi_metric(df_resampled, metrics, save_path)` — Grid of scatter plots showing up to six metrics simultaneously, with a shared legend.
- `plot_class_distribution(df_counts, save_path)` — Side-by-side pie chart and horizontal bar chart showing class proportions and absolute sample counts.
- `plot_byte_timeline(df_resampled, save_path)` — Line plot of total byte rate over time with color-coded background spans indicating the dominant traffic class in each time interval; uses vectorized grouping instead of row-by-row iteration for performance.
- `_metric_label(metric)` — Returns a human-readable axis label string for a given feature column name.
- `visualize_dataset(csv_path, resample)` — Entry point that calls `load_dataset` once and then produces all four plots in sequence.

---

### `inference_dsgenerator.py`

Preprocessor for converting real-world **CIC-DDoS2019** dataset CSV files into the 22-feature format expected by the trained model. Handles column name aliases, protocol encoding, and rolling-window approximations for features that don't exist directly in the CIC data.

**Constants:**

- `TARGET_FEATURES` — Ordered list of the 22 feature names; order must match the model's input.
- `LABEL_MAP` — Maps CIC-DDoS2019 original label strings (e.g. `"Syn"`, `"LDAP"`, `"BENIGN"`) to the project's internal class names.
- `CIC_COL_ALIASES` — Maps canonical internal column names to lists of known CIC column name variants (CICFlowMeter changes column names across versions).

**Functions:**

- `resolve_col(df, canonical)` — Searches a DataFrame's columns for any known alias of a canonical field name, returning the first match or `None`.
- `get_col(df, canonical, default)` — Extracts a column by canonical name, coerces it to numeric, and fills missing values with the provided default.
- `compute_unique_rolling(df, window)` — Approximates unique source and destination IP counts using a rolling window over category codes, since exact per-window counts are not stored in CIC flow records.
- `preprocess(df, rolling_window)` — Main transformation function: maps CIC columns to the 22-target features, computes protocol ratios from protocol ID integers, derives TCP flag ratios from flag counts, fills unavailable features (entropies, DNS ratios, subnet spread) with `0.0`, and cleans infinite/NaN values.
- `main()` — CLI entry point: prompts for input folder, output paths, and window size, then recursively finds all CSV files, processes each with `preprocess`, and concatenates results into final feature and label CSV files.

---

## Getting Started

**1. Install dependencies**

```bash
pip install torch scikit-learn pandas numpy matplotlib seaborn optuna tensorflow
```

**2. Generate a synthetic dataset**

```bash
python dataset_generator.py
# Output file name: traffic_data.csv
# Number of days (timeperiod): 3
```

**3. Train the model**

```bash
python torch_nn.py
# Insert csv file path: traffic_data.csv
```

**4. (Optional) Run hyperparameter tuning**

```bash
python hyperparam.py
# CSV file path: traffic_data.csv
# Choice (1/2): 1
```

**5. (Optional) Visualize the dataset**

```bash
python visualization.py
# CSV file path: traffic_data.csv
# Resample interval (default 5min): 5min
```

**6. (Optional) Preprocess real CIC-DDoS2019 data**

```bash
python inference_dsgenerator.py
# Enter path to input folder: /path/to/cic-ddos2019/
# Enter path for output features CSV: features.csv
# Enter path for output labels CSV: labels.csv
```

---

## Key Design Decisions

- **Synthetic data with realistic noise** — Each generated sample has Gaussian noise added at `noise_level=0.05` to prevent the model from exploiting perfectly clean distributions. Transition windows between traffic types further blur class boundaries.
- **Randomized attack ordering** — The order of attack classes is shuffled on every generation iteration to prevent minority classes from consistently appearing at the end of the dataset and being underrepresented in temporal validation splits.
- **Oversampling + loss weighting** — Minority classes are oversampled to 50% of the normal class count, and `CrossEntropyLoss` receives inverse-frequency class weights, addressing imbalance from both the data and loss sides simultaneously.
- **TimeSeriesSplit cross-validation** — Standard k-fold would allow future data to leak into training. All splits are strictly chronological to simulate realistic deployment conditions.
- **Attention for interpretability** — The attention mechanism produces per-timestep weights that can be inspected to understand which windows in a sequence most influenced a classification decision.


---

## Inference API & Deployment

The trained model is served via a **FastAPI** application containerized with Docker. An **Ollama** LLM engine runs alongside it to power attack scenario simulation and automated metric analysis.

---

### System Architecture

```
┌──────────────────────────────────────────────────┐
│                  Docker Network                  │
│                                                  │
│  ┌─────────────────┐     ┌────────────────────┐  │
│  │  llama-engine   │     │   api-service      │  │
│  │  (Ollama LLM)   │◄────│   (FastAPI)        │  │
│  │  port 11434     │     │   port 8000        │  │
│  └─────────────────┘     └────────────────────┘  │
│                                                  │
│  ┌─────────────────┐                             │
│  │  ollama-init    │  pulls model on first run   │
│  │  (one-shot)     │                             │
│  └─────────────────┘                             │
└──────────────────────────────────────────────────┘
```

The `api-service` waits for a healthy `llama-engine` before starting (Docker healthcheck on `/api/tags`). The `ollama-init` container pulls the configured LLM model (`llama3.1:8b` by default) once on first `compose up`.

---

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Root health check |
| `GET` | `/health` | Model status, window size, feature count, class list |
| `GET` | `/attacks/info` | LLM-generated technical descriptions of all attack types (cached after first call) |
| `POST` | `/predict` | Single-window inference from a JSON feature matrix |
| `POST` | `/predict/file` | Batch inference from an uploaded CSV file |
| `POST` | `/simulate` | End-to-end pipeline: Ollama generates scenario → LSTM classifies → Ollama analyzes |
| `POST` | `/analyze` | LangGraph metric analysis: per-class evaluation + confusion matrix + recommendations |

Interactive API docs available at `http://localhost:8000/docs`.

---

### API File Reference

---

#### `config.py`

Central configuration module. All constants that must match the trained model are defined here and read from environment variables where applicable.

**Constants:**
- `MODEL_PATH` — Path to the `.pt` checkpoint file (env: `MODEL_PATH`, default `ddos_lstm_attention.pt`).
- `CLASS_LABELS` — Ordered list of 9 class names; order must match the label encoder used during training.
- `WINDOW_SIZE` — Sliding window length in samples (env: `WINDOW_SIZE`, default `20`); must match training.
- `FEATURE_NAMES` — Ordered list of 12 input feature names used by the model.
- `LSTM_HIDDEN_SIZE`, `LSTM_NUM_LAYERS`, `LSTM_DROPOUT` — Architecture constants matching the saved checkpoint, readable from environment variables.

---

#### `model.py`

Loads the trained checkpoint and exposes inference methods. Contains a self-contained re-declaration of the model architecture so the API has no dependency on the training codebase.

**Classes:**

- `AttentionLayer(nn.Module)` — Identical re-implementation of the attention mechanism from training: computes per-timestep softmax scores and returns a weighted context vector.
- `DDoSLSTMAttention(nn.Module)` — Identical re-implementation of the full LSTM + Attention classifier; must match the architecture of the saved `.pt` checkpoint exactly.
- `ModelWrapper` — Singleton class that owns the loaded model, scaler, and label encoder for the lifetime of the API process.
  - `load(path)` — Loads the checkpoint from disk, reconstructs model architecture from saved hyperparameters, and moves the model to GPU if available.
  - `is_loaded` — Property returning `True` if the model checkpoint was loaded successfully.
  - `_windows_from_df(df)` — Internal helper that strips non-feature columns, applies the saved `StandardScaler`, and builds sliding-window sequences from a DataFrame.
  - `predict_single(window)` — Runs inference on a single pre-built window (list of lists); used by the `/predict` JSON endpoint.
  - `predict_from_df(df, batch_size)` — Builds all sliding windows from a DataFrame and runs batched inference; used by the `/predict/file` endpoint.

The module exports `model_wrapper`, a global singleton instance shared across the application.

---

#### `main.py`

FastAPI application definition. Registers all routes, loads the model on startup via the `lifespan` context manager, and wires together the model wrapper and Ollama client.

**Startup:**
- `lifespan(app)` — Async context manager that calls `model_wrapper.load()` at startup and logs whether the model is ready; endpoints return HTTP 503 if the model failed to load.

**Route handlers:**
- `root()` — Returns a simple confirmation message that the API is running.
- `health()` — Returns model load status, window size, feature count, and class list as a `HealthResponse`.
- `attacks_info()` — New endpoint (`GET /attacks/info`) that reads `attacks.py` from disk, calls Ollama's `generate_attack_descriptions` to produce a structured JSON description of every attack type, and caches the result in memory (`_attack_descriptions_cache`) so Ollama is called only once per process lifetime.
- `predict(request)` — Accepts a `PredictRequest` JSON body (one full sliding window), calls `model_wrapper.predict_single`, and returns a `PredictResponse` with the predicted class, confidence, attack flag, and per-class probabilities.
- `predict_file(file)` — Accepts a CSV file upload, reads it into a DataFrame, calls `model_wrapper.predict_from_df`, and returns a `FileInferenceResponse` with per-window predictions and an attack window count.
- `simulate(request)` — Three-step pipeline: (1) calls Ollama to generate a synthetic traffic matrix for the requested attack type, (2) runs LSTM inference on it, (3) calls Ollama to generate a human-readable attack description and mitigation steps; returns a `SimulateResponse`.

---

#### `schemas.py`

Pydantic models for all request and response bodies. Enforces field counts and value ranges at the API boundary.

**Request schemas:**
- `TrafficSample` — A single traffic sample; validates that `features` contains exactly `NUM_FEATURES` floats.
- `PredictRequest` — A full sliding window; validates that `window` contains exactly `WINDOW_SIZE` `TrafficSample` objects.
- `SimulateRequest` — Attack type string for the `/simulate` endpoint.
- `AnalyzeRequest` — Full metrics payload for the `/analyze` endpoint, matching the format produced by `metrics_exporter.export_metrics_to_json`; contains `classification_report`, `confusion_matrix`, `mcc_score`, and `roc_auc_scores`.

**Response schemas:**
- `PredictResponse` — Predicted class, confidence (0–1), attack flag, and per-class probability dict.
- `WindowPrediction` — Single-window result within a batch file response.
- `FileInferenceResponse` — Total window count, attack window count, and list of `WindowPrediction` objects.
- `HealthResponse` — Model status, window size, feature count, class list.
- `AttackAnalysis` — LLM-generated attack description and list of mitigation steps.
- `SimulateResponse` — Full simulate result combining prediction fields and an `AttackAnalysis`.
- `ClassMetrics` — Per-class precision/recall/F1/support; supports both `f1-score` (sklearn) and `f1_score` field name variants via Pydantic alias.
- `AnalysisRecommendations` — Three lists of strings: `data_generation`, `model`, and `training` recommendations.
- `AnalyzeResponse` — Full LangGraph output: weak classes, per-class analysis text, confusion pattern text, recommendations, and summary.

---

#### `ollama_client.py`

Async HTTP client for communicating with the Ollama LLM engine. Uses a **shared persistent `httpx.AsyncClient`** (created once, reused across all requests) and **dynamic model detection** (checks the Ollama server for available models instead of requiring a hardcoded name).

**Module-level state:**
- `_http_client` — Shared `httpx.AsyncClient` instance; created lazily on first use and reused for the lifetime of the process.
- `_cached_model` — Cached model name; resolved once from env or Ollama `/api/tags`, then reused on every subsequent call to avoid repeated HTTP lookups.

**Functions:**
- `get_http_client()` — Returns the shared `httpx.AsyncClient`, creating a new one if it has been closed.
- `_extract_json(raw)` — Strips markdown code fences (` ```json ... ``` `) from Ollama responses before JSON parsing; Ollama frequently wraps structured output in markdown.
- `_get_active_model(client)` — Resolves the model to use: first checks `OLLAMA_MODEL` env var, then queries Ollama `/api/tags` for available models (preferring `llama3.1:8b`), with `llama3.1:8b` as a final fallback; result is cached in `_cached_model`.
- `ollama_generate(prompt, system)` — Generic async call to Ollama `/api/generate` using the shared client and dynamically resolved model; raises `RuntimeError` if Ollama is unreachable.
- `generate_attack_scenario(attack_type, window_size, feature_names)` — Prompts Ollama to generate a `window_size × len(feature_names)` float matrix simulating the requested attack type; handles Ollama returning too many or too few rows by slicing or randomly duplicating existing rows rather than raising an error.
- `generate_attack_analysis(predicted_class, confidence, is_attack, class_probabilities, attack_type_requested)` — Prompts Ollama to analyze LSTM prediction results and return a structured JSON with a technical `description` paragraph and a list of `mitigation_steps`; falls back to raw text if JSON parsing fails.
- `generate_attack_descriptions(attacks_source)` — Takes the full source code of `attacks.py` as a string and prompts Ollama to return a JSON object mapping each attack name to a `description` paragraph and a `characteristics` list; used by both `/attacks/info` and the LangGraph `node_load_attack_descriptions`.

---

#### `langraph.py`

Implements the full **LangGraph analysis and retraining pipeline**. The graph loads evaluation metrics, compares them against a previous run, runs LLM analysis, makes a deterministic decision about the next action, optionally scans the project codebase, proposes new hyperparameters, and — after user confirmation outside the graph — can trigger a full model retraining subprocess.

**Graph flow:**

```
load_and_compare ──► load_attack_descriptions ──► analyze_per_class
    ──► analyze_confusion ──► synthesize ──► decision
            ├── SUGGEST_ONLY ──► END
            ├── SCAN_FIRST   ──► scan_codebase ──► propose_hyperparams ──► END
            └── RETRAIN      ──► propose_hyperparams ──► END

[human_confirm and trigger_retrain are called manually after the graph finishes]
```

**Decision thresholds:**
- `DECISION_RETRAIN_MCC = 0.75` — MCC below this forces `RETRAIN`.
- `DECISION_SCAN_MCC = 0.85` — MCC below this (or any regression detected) triggers `SCAN_FIRST`.
- Above `DECISION_SCAN_MCC` with no regression → `SUGGEST_ONLY`.

**State (`MetricsState` TypedDict):**

| Field | Direction | Description |
|---|---|---|
| `classification_report`, `confusion_matrix`, `mcc_score`, `roc_auc_scores`, `class_labels` | Input | Current run metrics |
| `previous_metrics`, `metrics_delta`, `regression_detected` | Computed | Delta vs previous run |
| `per_class_analysis`, `confusion_analysis`, `attack_descriptions` | Intermediate | LLM outputs |
| `weak_classes`, `recommendations`, `summary` | Output | Analysis results |
| `decision`, `scanned_files`, `proposed_hyperparams` | Output | Action plan |
| `human_confirmed`, `retrain_triggered`, `retrain_command` | Output | Retrain status |
| `dataset_path`, `model_name` | Config | Set during human confirmation |

**Helper functions:**
- `_format_per_class_metrics(report, roc_auc, mcc)` — Formats per-class metrics into a readable string for LLM prompts.
- `_format_confusion_matrix(matrix, labels)` — Renders the confusion matrix as an ASCII table for LLM prompts.
- `_safe_parse_json(raw, fallback)` — Attempts to parse an LLM response as JSON after stripping markdown fences; returns a fallback dict if parsing fails.
- `_compute_delta(current, previous)` — Compares current metrics against a previous run; returns a delta dict and a `regression_detected` bool (True if MCC dropped, macro F1 dropped, or any class lost ≥ 0.03 F1).
- `_route_after_decision(state)` — Routing function for the conditional edge after `node_decision`; returns `"SUGGEST_ONLY"`, `"SCAN_FIRST"`, or `"RETRAIN"`.
- `_route_after_confirm(state)` — Routing function for the conditional edge after `node_human_confirm`; returns `"retrain"` or `"skip"`.
- `_find_file_recursive(filename, root)` — Recursively searches for a file by name under `root`, skipping directories in `SCAN_SKIP_DIRS` (`__pycache__`, `venv`, `.git`, etc.).
- `_validate_hyperparams(proposed)` — Validates LLM-proposed hyperparameters against `_HYPERPARAMETER_SPACE`; clamps out-of-range values to the nearest valid choice or range bound instead of rejecting them.
- `_patch_hyperparam_file(hp_path, proposed)` — Writes proposed hyperparameters into `hyperparam.py` using regex substitution; creates a `.bak` backup before modifying.
- `_prompt_user()` — Synchronous helper that reads user confirmation and dataset path from stdin; called inside `run_in_executor` so it doesn't block the async event loop.

**Graph nodes:**
- `node_load_and_compare_metrics(state)` — Loads `test_metrics_prev.json` if it exists, calls `_compute_delta`, and populates `previous_metrics`, `metrics_delta`, and `regression_detected`; no LLM call.
- `node_load_attack_descriptions(state)` — Finds `attacks.py` recursively, reads its source, and calls `generate_attack_descriptions` to produce LLM-generated technical descriptions of each attack type; populates `attack_descriptions`.
- `node_analyze_per_class(state)` — Sends per-class metrics (enriched with regression delta context and attack descriptions) to Ollama; identifies weak classes (F1 < 0.80, recall < 0.75, or ROC-AUC < 0.85) and explains why they underperform; populates `weak_classes` and `per_class_analysis`.
- `node_analyze_confusion(state)` — Sends the ASCII confusion matrix and previously identified weak classes to Ollama; identifies top misclassification pairs and their likely causes; populates `confusion_analysis`.
- `node_synthesize_recommendations(state)` — Synthesizes all prior analysis (including regression trend) into three recommendation lists (`data_generation`, `model`, `training`) and a summary; normalizes LLM responses that return dicts instead of plain strings; populates `recommendations` and `summary`.
- `node_decision(state)` — Pure Python; applies MCC thresholds and regression flag to set `decision` deterministically; no LLM call.
- `node_scan_codebase(state)` — Determines which source files to read based on `weak_classes` (using `CLASS_TO_FILES` mapping), reads each file recursively up to `SCAN_MAX_CHARS_PER_FILE` characters, and populates `scanned_files`; always includes `hyperparam.py` and `config.py`.
- `node_propose_hyperparams(state)` — Prompts Ollama with analysis results and scanned file excerpts to propose concrete hyperparameter values strictly within `_HYPERPARAMETER_SPACE`; runs the response through `_validate_hyperparams` before storing in `proposed_hyperparams`.
- `node_human_confirm(state)` — Blocks execution and waits for user confirmation in the terminal via `run_in_executor`; displays proposed hyperparameters and LLM reasoning; populates `human_confirmed` and `dataset_path`.
- `node_trigger_retrain(state)` — Finds `torch_nn.py` and `hyperparam.py` recursively, patches hyperparameters with `_patch_hyperparam_file`, generates a timestamped model output name, and launches `torch_nn.py` as an async subprocess with stdout streamed line by line; populates `retrain_triggered` and `retrain_command`.
- `build_analyzer_graph()` — Assembles and compiles the full LangGraph `StateGraph`; `node_human_confirm` and `node_trigger_retrain` are intentionally excluded from the graph and called manually after the graph finishes.

The module exports `analyzer_graph`, a compiled global graph instance invoked via `await analyzer_graph.ainvoke(state)`.

---

#### `langraph_test.py`

Interactive CLI runner for the full LangGraph analysis pipeline. Loads evaluation metrics from a JSON file, runs the compiled graph, prints a structured report, and then calls `node_human_confirm` and (if confirmed) `node_trigger_retrain` manually after the graph finishes. Saves the full analysis result to `langgraph_analysis.json` and backs up current metrics to `test_metrics_prev.json` for delta comparison on the next run.

**Functions:**
- `get_user_input()` — Prompts the user for the metrics file path and optional Ollama host URL before the graph starts.
- `separator(char, width)` — Prints a horizontal rule for terminal output formatting.
- `section(title)` — Prints a titled section header with surrounding separators.
- `print_list(items, indent)` — Prints a numbered list of recommendation strings with line wrapping at 90 characters; handles cases where the LLM returned dicts instead of plain strings.
- `print_attack_descriptions(descriptions)` — Prints a compact per-attack summary of description and key characteristics for the attack types section of the report.
- `print_results(state)` — Renders the full analysis report to stdout: regression delta, weak classes, attack descriptions, per-class analysis, confusion analysis, all three recommendation categories, decision outcome, proposed hyperparameters, scanned files, and executive summary.
- `check_ollama(host)` — Async function that pings Ollama `/api/tags` and prints available models; the runner exits if Ollama is unreachable.
- `run_test(metrics_path, ollama_url)` — Main async entry point: sets up env, loads the metrics JSON, builds the initial `MetricsState`, invokes the graph, backs up metrics, prints results, handles human confirmation and retrain, and saves output JSON.

**Usage:**
```bash
python langraph_test.py
# Enter path to metrics file (default: results/test_metrics.json):
# Enter Ollama host url (default: OLLAMA_HOST env var or http://localhost:11434):
```

---

#### `langraph_test_retrain.py`

Unit test suite for the retraining-related functions in `langraph.py`. Tests run entirely with mock state and patched subprocess calls — no Ollama connection or real model required.

**Test classes:**

- `TestValidateHyperparams` — Verifies that `_validate_hyperparams` passes valid choices through unchanged, clamps invalid choice values to the nearest valid option, clamps range values that exceed bounds, and silently skips parameters not present in the proposed dict.
- `TestNodeDecision` — Verifies the deterministic MCC threshold logic: `SUGGEST_ONLY` above 0.85 with no regression, `SCAN_FIRST` in the 0.75–0.85 range, `RETRAIN` below 0.75, and `SCAN_FIRST` when regression is detected even if MCC exceeds the scan threshold.
- `TestNodeHumanConfirm` — Verifies that `node_human_confirm` sets `human_confirmed=True` on `"y"` or `"yes"` input and `False` on `"n"`, empty string, or any unrecognized input.
- `TestRouteAfterConfirm` — Verifies that `_route_after_confirm` returns `"retrain"` when confirmed and `"skip"` when not.
- `TestPatchHyperparamFile` — Verifies that `_patch_hyperparam_file` creates a `.bak` backup before writing, correctly patches `choice`-type parameters (single-element list), correctly patches `range`-type parameters (collapsed to a point range), and leaves unrelated lines unchanged.
- `TestNodeTriggerRetrain` — Verifies (using mocked `subprocess.Popen` and `_find_file_recursive`) that `node_trigger_retrain` sets `retrain_triggered=True` on success, includes `torch_nn.py` in the retrain command, calls `Popen` exactly once, and sets `retrain_triggered=False` without calling `Popen` when `torch_nn.py` is not found.

**Usage:**
```bash
python langraph_test_retrain.py
```

---

#### `inference_script.py`

CLI tool for testing the `/predict/file` endpoint from the command line without writing client code.

**Usage:**
```bash
python inference_script.py traffic.csv
python inference_script.py traffic.csv --url http://localhost:8000
python inference_script.py traffic.csv --json   # raw JSON output
```

**Functions:**
- `inference_script()` — Parses CLI arguments, uploads the CSV file to `/predict/file`, and prints a formatted per-window summary (predicted class, confidence, attack status); the `--json` flag prints the raw response instead.

---

#### `ollama_test.py`

Interactive test suite for validating the Ollama integration and full API pipeline. Runs as a REPL menu or via CLI flags.

**Usage:**
```bash
python ollama_test.py                          # interactive menu (default)
python ollama_test.py --test all --verbose
python ollama_test.py --test simulate --attack syn_flood
```

**Helper functions:**
- `ok(msg)`, `fail(msg)`, `info(msg)`, `section(title)`, `subsection(title)` — Formatted console output helpers for test result display.
- `_prompt(text, default)` — Reads a line from stdin with a default fallback; handles `KeyboardInterrupt` cleanly.
- `_pick_attack_type()` — Interactive numbered menu for selecting an attack type from `CLASS_LABELS`.
- `_show_menu(session)` — Renders the main REPL menu with current connection config.

**Test functions:**
- `test_ping(ollama_host, verbose)` — Verifies Ollama server reachability at `/api/tags` and lists available models with sizes.
- `test_raw_generate(ollama_host, model, verbose)` — Sends a minimal single-word prompt to confirm the LLM responds correctly and measures latency.
- `test_generate_scenario(ollama_host, model, attack_type, verbose)` — Requests a traffic matrix from Ollama and validates that it returns a JSON array with the correct number of rows and columns.
- `test_generate_analysis(ollama_host, model, verbose)` — Requests an attack analysis JSON and validates that both required keys (`description`, `mitigation_steps`) are present.
- `test_simulate_endpoint(api_host, attack_type, verbose)` — Calls the `/simulate` FastAPI endpoint end-to-end and verifies that an `analysis` field is present in the response.
- `test_edge_cases(ollama_host, api_host, model, verbose)` — Sends an invalid attack type to `/simulate` and expects an HTTP 400 response.
- `interactive_mode(ollama_host, api_host, model)` — Runs the main REPL loop, dispatching to individual test functions based on menu selection.

---

### Deployment

**Prerequisites:** Docker, Docker Compose, NVIDIA GPU + drivers (for CUDA inference).

**Start the stack:**
```bash
docker compose up --build
```

This will:
1. Start `llama-engine` (Ollama) and wait for it to become healthy.
2. Run `ollama-init` once to pull `llama3.1:8b` if not already cached.
3. Start `api-service` (FastAPI on port 8000) once Ollama is healthy.

**Environment variables** (set in `docker-compose.yml` or a `.env` file):

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `ddos_lstm_attention.pt` | Path to the PyTorch checkpoint inside the container |
| `OLLAMA_HOST` | `http://llama-engine:11434` | Ollama service URL (internal Docker network) |
| `OLLAMA_MODEL` | *(auto-detected)* | LLM model name; if unset, the API queries Ollama for available models and prefers `llama3.1:8b` |
| `WINDOW_SIZE` | `20` | Sliding window length (must match training) |
| `LSTM_HIDDEN_SIZE` | `128` | Must match checkpoint |
| `LSTM_NUM_LAYERS` | `2` | Must match checkpoint |
| `LSTM_DROPOUT` | `0.3` | Must match checkpoint |

**Stop:**
```bash
docker compose down
```

**Dockerfile notes:**
- Base image: `python:3.11-slim`
- PyTorch is installed separately using the `+cu121` CUDA 12.1 wheel index; the standard PyPI index does not carry CUDA builds.