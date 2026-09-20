# AASIST with RawBoost Data Augmentation: State-of-the-Art Voice Deepfake Detection
## ASVspoof 2019 Logical Access Benchmark Research Study

---

## 1. Overview and Scientific Motivation

This research repository contains the end-to-end implementation of **AASIST** (*Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks*, Interspeech 2022) paired with **RawBoost** data augmentation (ICASSP 2022).

In our prior two benchmarks:
1. **SE-ResNet-18 (Log-Mel Spectrograms)**: Achieved a 0.064% Development EER, but failed on out-of-distribution neural vocoders during evaluation (Evaluation EER: 23.818%) due to the *Phase-Blindness Problem* of STFT magnitude representations.
2. **RawNet2-Mini (Raw Waveforms)**: Cut the Evaluation EER to 11.655% and boosted ROC-AUC to 0.9550, successfully detecting sample-level phase anomalies on WaveNet (A10) and Differential Formant synthesis (A13).
3. **The SOTA Target (AASIST + RawBoost)**: To push our defense system into the international top tier (EER < 1.5%), AASIST models both temporal nodes (across time frames) and spectral nodes (across frequency subbands) concurrently via graph attention networks, while RawBoost prevents time-domain overfitting through acoustic channel, non-linear quantization, and impulsive noise simulation.

---

## 2. Core Architectural Innovations

### 2.1 SincNet Parameterized Convolutional Frontend
Rather than employing unconstrained 1D convolutional weights that overfit to speaker-specific formant tracks, the frontend employs 128 parameterized bandpass filters derived from the continuous sinc function:
$$g(t, f_1, f_2) = 2f_2 \cdot \text{sinc}(2\pi f_2 t) - 2f_1 \cdot \text{sinc}(2\pi f_1 t)$$
where only low cutoff $f_1$ and bandwidth $f_2 - f_1$ are optimized via gradient descent. A symmetric Hamming window eliminates Gibbs truncation artifacts:
$$w[n] = 0.54 - 0.46 \cos\left(\frac{2\pi n}{L}\right)$$

### 2.2 Spectro-Temporal ResBlocks with Max-Feature-Map (MFM)
Feature representations are shaped into 2D spectro-temporal feature maps. Convolutions use Max-Feature-Map (MFM) activations, which select competitive maxima across channel pairs:
$$y_{i,j}^k = \max\left(x_{i,j}^{2k-1}, x_{i,j}^{2k}\right)$$
This non-linearity filters low-magnitude noise gradients and concentrates energy on sparse deepfake synthesis artifacts.

### 2.3 Heterogeneous Spectro-Temporal Graph Attention Network (HS-GAT)
From the 2D feature map $X \in \mathbb{R}^{B \times C \times F \times T}$:
- **Temporal Graph ($G_T$)**: Squeezes spectral subbands, producing $T$ temporal nodes. A temporal Graph Attention Network (GAT) computes dynamic inter-frame attention:
  $$\alpha_{i, j}^T = \frac{\exp\left(\text{LeakyReLU}\left(a_T^T [W_T v_i \,\|\, W_T v_j]\right)\right)}{\sum_{k \in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(a_T^T [W_T v_i \,\|\, W_T v_k]\right)\right)}$$
- **Spectral Graph ($G_S$)**: Squeezes time frames, producing $F$ spectral subband nodes. A spectral GAT computes attention across frequency bands.
- **Heterogeneous Readout Node**: A learnable master node executes cross-attention pooling over both temporal and spectral nodes, discovering localized spectro-temporal vocoder anomalies.

### 2.4 Official RawBoost Data Augmentation Engine
RawBoost applies three stochastic transformations directly in the raw 1D time domain:
1. **Algorithm 1 (Linear Convolutive Noise)**: Simulates transmission channels and acoustic impulse responses via random FIR filtering:
   $$y[n] = x[n] + \sum_{k=1}^K h[k] x[n-k]$$
2. **Algorithm 2 (Non-Linear Additive Coloured Noise)**: Simulates analog-to-digital non-linearities and quantization noise:
   $$y[n] = x[n] + \beta \cdot \text{sign}(x[n]) |x[n]|^\gamma + \eta_{\text{coloured}}[n]$$
3. **Algorithm 3 (Impulsive Signal-Dependent Noise)**: Simulates packet loss and burst dropouts:
   $$y[n] = x[n] \cdot (1 - m[n]) + \delta[n] \cdot \sigma_x$$

---

## 3. Tri-Modal Master Architectural Comparison

| Dimension | Experiment 1: SE-ResNet-18 | Experiment 2: RawNet2-Mini | Experiment 3: AASIST + RawBoost |
| :--- | :--- | :--- | :--- |
| **Input Domain** | 2D Log-Mel Spectrogram | 1D Raw Waveform | 1D Raw Continuous Waveform |
| **Frontend Method** | Fixed STFT + Mel Filters | SincConv1D + FMS | SincConv1D + Max-Feature-Map |
| **Time-Frequency Modeling** | 2D Residual Convolutions | 1D Convolutions + FMS | Heterogeneous Graph Attention (HS-GAT) |
| **Data Augmentation** | 2D SpecAugment (Time/Freq) | None (Standard Caching) | 3-Algorithm RawBoost (Time-Domain) |
| **Trainable Parameters** | 2,856,922 parameters | 4,566,082 parameters | ~297,000 parameters (Ultra-Lightweight) |
| **Development EER** | 0.064% | 0.466% | Sub-0.5% (Target) |
| **Evaluation EER** | 23.818% | 11.655% | **0.8% - 1.5% (Literature Benchmark)** |
| **Evaluation ROC-AUC** | 0.8440 | 0.9550 | **> 0.9850 (Target)** |

---

## 4. Kaggle Execution Instructions

1. Create a new notebook on Kaggle.
2. Under **Settings**, select **GPU T4 x2** or **GPU T4 x1** accelerator.
3. Add the official ASVspoof 2019 dataset to the notebook:
   - Search for `asvpoof-2019-dataset` or `awsaf49/asvpoof-2019-dataset`.
4. Upload or import `voice-deepfake-detection-aasist.ipynb`.
5. Execute **Run All**. The notebook is completely self-contained and runs deterministically from Cell 01 to Cell 30 in a single pass.

---

## 5. Artifacts Produced During Execution

Upon completion, all outputs are saved to `/kaggle/working/`:
- **Model Checkpoint**: `models/aasist_best.pth`
- **Training Trajectory**: `models/training_history.json`
- **Benchmark Summary**: `experiment_final_report.json`
- **Attack Vulnerability Table**: `attack_vulnerability_breakdown.csv`
- **15 Publication-Grade Figures (300 DPI)** in `figures/`:
  - `01_class_distribution_breakdown.png`
  - `02_attack_distribution_analysis.png`
  - `03_audio_duration_length_distribution.png`
  - `04_waveform_time_domain_glottal_inspection.png`
  - `05_rawboost_augmentation_transformations.png`
  - `06_sincnet_learned_filterbank_frequency_response.png`
  - `07_training_loss_and_dev_eer_trajectories.png`
  - `08_receiver_operating_characteristic_roc.png`
  - `09_detection_error_tradeoff_det.png`
  - `10_precision_recall_curves.png`
  - `11_normalized_confusion_matrix_eval.png`
  - `12_attack_by_attack_accuracy_barchart.png`
  - `13_tsne_latent_manifold_clusters.png`
  - `14_graph_attention_node_attribution.png`
  - `15_single_file_live_inference_verification.png`
