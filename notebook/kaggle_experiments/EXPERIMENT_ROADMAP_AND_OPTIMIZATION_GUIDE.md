# Voice Deepfake Defense System: Comprehensive Experiment Roadmap and Optimization Guide
## ASVspoof 2019 Logical Access (LA) Benchmark

---

## 1. Executive Summary and Current Project Status

The objective of this research project is to construct an end-to-end biometric anti-spoofing countermeasure system capable of detecting synthetic speech (Text-to-Speech and Voice Conversion) with an Equal Error Rate (EER) under 1.0% to 2.0% and maximum generalization across unseen spoofing attacks on the ASVspoof 2019 Logical Access (LA) benchmark.

The dataset consists of 121,461 utterances across three strictly speaker-disjoint splits:
- **Train Partition:** 25,380 utterances (2,580 bonafide, 22,800 spoof across attacks A01 through A06; 20 speakers).
- **Development Partition:** 24,844 utterances (2,548 bonafide, 22,296 spoof across attacks A01 through A06; 10 disjoint speakers).
- **Evaluation Partition:** 71,237 utterances (7,355 bonafide, 63,882 spoof across known attacks A16/A19 and mostly unseen attacks A07 through A15, A17, A18; 48 disjoint speakers).

### Current Status of Models in the Pipeline

| Experiment | Architecture | Input Feature Representation | Acoustic Focus | Status | Key Performance Indicator |
|---|---|---|---|:---:|---|
| **Run 1** | Light-CNN (MFM) | 60-dim Linear Frequency Cepstral Coefficients (LFCC) | High-frequency Nyquist range ($0 - 8000\text{ Hz}$) | Completed | Dev EER: **10.47%**, AUC: **0.9588**, Precision: **98.68%** |
| **Run 2** | SE-ResNet-18 | 128-channel Log-Mel Spectrogram | Low-mid vocal tract formants ($0 - 4000\text{ Hz}$) | Running / Completed | Targets A04 (STRAIGHT) and A05 (VAE) blind spots |
| **Run 3** | RawNet2-Mini | 1D Raw Waveform ($64,000$ samples) with SincNet | Phase inconsistencies & micro-temporal pitch jitter | Ready for Execution | End-to-end waveform modeling with FMS attention |

---

## 2. In-Depth Forensic Audit of Completed and In-Progress Experiments

### 2.1 Experiment 1: Light-CNN with Max-Feature-Map (MFM) and LFCC
- **Configuration:** 60-dim LFCC ($20\text{ static} + 20\text{ } \Delta + 20\text{ } \Delta\Delta$), input shape `(1, 60, 251)`. Focal Loss ($\alpha=0.75, \gamma=2.0$, label smoothing 0.05), Cosine Annealing scheduler, batch size 128, 20 epochs.
- **Empirical Results:**
  - Optimal Development EER: **10.47%** at Epoch 19.
  - Area Under the ROC Curve (AUC): **0.9588**.
  - Accuracy at optimal threshold ($\theta^* = 0.3195$): **89.54%**.
  - Precision: **98.68%**, Recall: **89.54%**, F1-Score: **0.9389**.
- **Granular Attack Breakdown:**
  - Attack A06 (Voice Conversion via Transfer Function Regression + WORLD): **99.41%** detection accuracy.
  - Attack A03 (TTS via Feed-Forward Neural Network + Unit Selection): **96.21%** detection accuracy.
  - Attack A01 (TTS via Autoregressive RNN + WaveNet vocoder): **95.96%** detection accuracy.
  - Attack A02 (TTS via Autoregressive RNN + WORLD vocoder): **91.77%** detection accuracy.
  - Attack A05 (Voice Conversion via Variational Autoencoder): **84.45%** detection accuracy.
  - Attack A04 (Voice Conversion via Formant/Pitch Shifting + STRAIGHT vocoder): **69.46%** detection accuracy.
- **Acoustic Physics Root-Cause Analysis:**
  - Light-CNN with LFCC achieved near-perfect detection on vocoders that generate high-frequency energy inconsistencies and spectral step discontinuities (WaveNet, WORLD).
  - However, Attack A04 uses the STRAIGHT vocoder with pitch-adaptive spectral smoothing, eliminating sharp spectral artifacts in the upper Nyquist band ($4000 - 8000\text{ Hz}$). Because LFCC distributes triangular filterbanks uniformly across the entire frequency spectrum, it allocates insufficient filter density to the primary formant transition zone ($F_1, F_2, F_3$ in $0 - 3500\text{ Hz}$) where STRAIGHT speech alterations occur.
  - Attack A05 uses a Variational Autoencoder (VAE) whose continuous latent space enforces smooth spectral envelopes, masking vocoder spectral jumps.

### 2.2 Experiment 2: SE-ResNet-18 with 128-Channel Log-Mel Spectrogram
- **Configuration:** 128-channel Log-Mel spectrogram, input shape `(1, 128, 251)`. Squeeze-and-Excitation residual blocks with dynamic channel recalibration ($\mathbf{s} = \sigma(\mathbf{W}_2 \text{ReLU}(\mathbf{W}_1 \mathbf{z}))$), 2D SpecAugment, Focal Loss, 30 epochs.
- **Hypothesis and Acoustic Role:**
  - Mel filterbanks allocate logarithmic frequency spacing, concentrating dense filter resolution between $20\text{ Hz}$ and $4000\text{ Hz}$.
  - The Squeeze-and-Excitation attention blocks compute channel-wise summary statistics through global average pooling and adaptively amplify feature channels that correspond to formant distortions while suppressing speaker-specific pitch variations.
  - Directly counters the vulnerabilities observed in Experiment 1 on A04 and A05.

### 2.3 Experiment 3: RawNet2-Mini with Raw Waveform and SincNet Frontend
- **Configuration:** Raw 1D waveform, input shape `(1, 64000)` (4.0 seconds at 16 kHz). SincConv1D frontend with 64 parameterized bandpass sinc filters initialized on the Mel scale, Feature Map Scaling (FMS), RawBoost augmentation, Focal Loss, 30 epochs. Total trainable parameters: 671,362.
- **Hypothesis and Acoustic Role:**
  - Fourier magnitude spectra discard phase information. However, neural speech synthesis vocoders (especially WaveNet, Griffin-Lim, and neural source-filter models) generate micro-temporal pitch jitter, non-minimum-phase dispersion, and glottal flow distortions in the time domain.
  - SincNet learns parameterized cutoff frequencies directly via gradient descent, enabling the network to tune its filterbank band boundaries specifically around vocoder synthesis artifact frequencies.

---

## 3. High-Impact Beneficial Experiments for Maximum EER Reduction

To drive system performance toward the state-of-the-art (<1.0% to 2.0% EER), six high-impact experiments have been designed. These address feature diversity, model capacity, metric learning, and score calibration.

```
+-----------------------------------------------------------------------------------+
|                           PROPOSED EXPERIMENT PIPELINE                            |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [Exp 1: Light-CNN + LFCC]    [Exp 2: SE-ResNet + Mel]   [Exp 3: RawNet2 + Raw]   |
|         (Completed)                  (Running)                   (Ready)          |
|              \                           |                           /            |
|               +--------------------------+--------------------------+             |
|                                          |                                        |
|                                          v                                        |
|                    [Exp 4: Tri-Modal Ensemble Soft-Voting]                        |
|                    - Nelder-Mead Optimal Convex Weights                           |
|                    - Logit Stacking Meta-Classifier                               |
|                    - Expected EER: 3.0% - 4.5%                                    |
|                                          |                                        |
|            +-----------------------------+-----------------------------+          |
|            |                                                           |          |
|            v                                                           v          |
|  [Exp 5: SSL Speech Model]                                   [Exp 6: CQCC Domain] |
|  - WavLM / Wav2Vec2 Backbone                                 - Constant-Q CQT     |
|  - Attentive Layer Fusion                                    - Geometric Scale    |
|  - Expected EER: 1.0% - 2.5%                                 - Unseen Robustness  |
|            |                                                           |          |
|            +-----------------------------+-----------------------------+          |
|                                          |                                        |
|                                          v                                        |
|                         [Exp 7: RawBoost Protocol]                                |
|                         - Convolutive & Colored Noise                             |
|                         - Impulsive & Packet Loss Noise                           |
|                                          |                                        |
|                                          v                                        |
|                         [Exp 8: AAM-Softmax (ArcFace)]                            |
|                         - Angular Margin Metric Loss                              |
|                         - Tight Bonafide Hypersphere                              |
|                                          |                                        |
|                                          v                                        |
|                         [Exp 9: Probability Calibration]                          |
|                         - Temperature & Platt Scaling                             |
|                         - Minimum-t-DCF Optimization                              |
|                                          |                                        |
|                                          v                                        |
|                         [Final Multi-Stream Deep Defense]                         |
|                         - Target EER < 1.0% on Eval Split                         |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

---

### Experiment 4: Tri-Modal Ensemble Fusion and Score Calibration
- **Scientific Rationale:**
  - Individual models evaluate orthogonal signal domains: LFCC models uniform spectral magnitudes, Mel models psychoacoustic formant structures, and RawNet models time-domain phase jitter.
  - The errors across these models are conditionally independent. When an individual model struggles with a specific attack (e.g. Light-CNN on A04), the other models provide confident correct predictions.
- **Mathematical Formulation:**
  1. Let $S_1(x), S_2(x), S_3(x) \in [0, 1]$ denote the predicted posterior spoof probabilities from Light-CNN, SE-ResNet-18, and RawNet2-Mini.
  2. The fused score $S_{\text{ensemble}}(x)$ is parameterized as a convex combination:
     $$S_{\text{ensemble}}(x) = w_1 \cdot S_1(x) + w_2 \cdot S_2(x) + w_3 \cdot S_3(x), \quad \text{subject to } \sum_{i=1}^3 w_i = 1, \; w_i \ge 0$$
  3. The optimal weight vector $\mathbf{w}^* = [w_1^*, w_2^*, w_3^*]$ is determined by minimizing the development set Equal Error Rate using the Nelder-Mead simplex optimization algorithm:
     $$\mathbf{w}^* = \arg\min_{\mathbf{w}} \text{EER}\left( \sum_{i=1}^3 w_i S_i(\mathcal{D}_{\text{dev}}), Y_{\text{dev}} \right)$$
- **Alternative Stacking Architecture:**
  - Train a regularized Logistic Regression or Support Vector Classifier on the 3-dimensional logit vector $[z_1, z_2, z_3]$ evaluated on the development set.
- **Expected Performance Gain:**
  - Dev EER expected to drop from ~10% down to **3.0% - 4.5%**.

---

### Experiment 5: Self-Supervised Learning (SSL) Speech Foundation Models
- **Scientific Rationale:**
  - Speech foundation models pre-trained on 60,000+ hours of speech (e.g., `microsoft/wavlm-base-plus`, `facebook/wav2vec2-xls-r-300m`) encode rich phonetic representations across deep transformer layers.
  - Synthetic vocoders fail to perfectly reproduce fine-grained phonetic co-articulation and vocal tract dynamics. In SSL models, this manifests as anomalies in the middle-to-higher transformer layer activations.
- **Technical Architecture:**
  1. **Backbone:** Pre-trained `wavlm-base-plus` or `wav2vec2-base-960h` (12 transformer encoder layers, 768 hidden dimensions).
  2. **Layer Aggregation:** Instead of using only the last transformer layer, compute a learnable weighted sum across all 12 layers:
     $$\mathbf{h}_{\text{fused}} = \sum_{l=1}^{12} \frac{\exp(\beta_l)}{\sum_{k=1}^{12} \exp(\beta_k)} \mathbf{h}_l$$
  3. **Backend Pooling:** Attentive Statistics Pooling (ASP) computing both weighted mean and weighted standard deviation across temporal frames:
     $$\mathbf{e} = [\boldsymbol{\mu}; \boldsymbol{\sigma}] \in \mathbb{R}^{1536}$$
  4. **Projection Head:** Linear(1536, 128) -> LeakyReLU -> Linear(128, 2).
  5. **Training Strategy:** Freeze the convolutional feature encoder; train layer weights $\boldsymbol{\beta}$ and projection head for 5 epochs; then fine-tune top 3 transformer layers with a low learning rate ($5 \times 10^{-6}$).
- **Expected Performance Gain:**
  - Single-model Dev EER expected to drop to **1.0% - 2.5%**.

---

### Experiment 6: Constant-Q Cepstral Coefficients (CQCC) Feature Engineering
- **Scientific Rationale:**
  - CQCC was the official primary baseline feature of the ASVspoof challenge.
  - Unlike Fourier transforms with fixed frequency resolution $\Delta f = \frac{f_s}{N}$, the Constant-Q Transform (CQT) maintains a constant quality factor $Q = \frac{f_k}{\Delta f_k}$ across all octaves.
  - Low frequencies have narrow filter bandwidths with high frequency resolution (capturing fundamental frequency $F_0$ and glottal harmonics), while high frequencies have wide bandwidths with high temporal resolution (capturing transient vocoder switching clicks and onset anomalies).
- **Extraction Protocol:**
  - Minimum frequency: $15\text{ Hz}$, maximum frequency: $8000\text{ Hz}$ (Nyquist).
  - Number of bins per octave: $B = 96$.
  - Resampling: Geometrically spaced log-frequency bins resampled uniformly before Discrete Cosine Transform (DCT).
  - Feature dimensions: 19 static cepstral coefficients + Energy + First deltas + Second deltas = 60 dimensions.
- **Model Pairing:** Light-CNN or ResNet-18.

---

### Experiment 7: RawBoost Advanced Data Augmentation Protocol
- **Scientific Rationale:**
  - In real-world environments, audio undergoes microphone frequency coloration, transmission line filtering, room reverberation, and ambient acoustic degradation.
  - Models without data augmentation often overfit to the clean recording conditions of the VCTK source corpus and fail on unseen evaluation conditions.
- **Algorithmic Formulation (ASVspoof 2021 RawBoost Benchmark):**
  1. **Series 1 (Convolutive Noise):** Simulates non-linear linear-phase filters by convolving the input with random finite impulse response (FIR) filters:
     $$\tilde{x}[n] = x[n] + \sum_{k=1}^K a_k x[n-k] + b \cdot x[n]^2$$
  2. **Series 2 (Stationary Additive Colored Noise):** Adds pink ($1/f$) and brown ($1/f^2$) filtered noise with variable Signal-to-Noise Ratios (SNR $\in [10, 30]\text{ dB}$):
     $$\tilde{x}[n] = x[n] + 10^{-\frac{\text{SNR}}{20}} \cdot \frac{\|x\|_2}{\|w\|_2} w[n]$$
  3. **Series 3 (Impulsive Noise and Packet Loss):** Injects zeroed-out micro-segments ($10 - 50\text{ ms}$) to simulate transmission dropouts.
- **Application:** Applied on-the-fly inside the PyTorch `Dataset.__getitem__` pipeline for raw waveform and spectral feature extractors.

---

### Experiment 8: Metric Learning with Additive Angular Margin (AAM-Softmax / ArcFace)
- **Scientific Rationale:**
  - Standard Cross-Entropy and Focal Loss optimize for separable linear decision boundaries ($W_1^T x > W_2^T x$). However, they do not enforce compact intra-class feature clusters.
  - In voice biometrics, authentic human speech occupies a compact, constrained manifold dictated by physical vocal tract anatomy. Spoofed speech, by contrast, is synthesized by diverse, unconstrained generative algorithms.
  - Additive Angular Margin Loss (ArcFace) projects feature vectors onto a unit hypersphere and introduces an explicit angular penalty $m$ to bonafide features.
- **Mathematical Formulation:**
  $$L_{\text{AAM}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(s \cdot \cos(\theta_{y_i} + m))}{\exp(s \cdot \cos(\theta_{y_i} + m)) + \sum_{j \ne y_i} \exp(s \cdot \cos\theta_j)}$$
  where:
  - $s$: Hypersphere radius scale parameter (typically $s = 30.0$).
  - $m$: Angular margin penalty (typically $m = 0.35 - 0.50\text{ radians}$).
  - $\theta_j = \arccos\left(\frac{\mathbf{W}_j^T \mathbf{z}}{\|\mathbf{W}_j\|_2 \|\mathbf{z}\|_2}\right)$.
- **Expected Outcome:** Forces bonafide speech representations into an extremely tight cluster on the hypersphere, leaving any synthetic artifact outside the angular boundary. Significantly reduces False Acceptance Rate (FAR).

---

### Experiment 9: Post-Processing and Temperature-Scaled Calibration
- **Scientific Rationale:**
  - Uncalibrated neural network output logits produce miscalibrated probabilities that tend to be overconfident near 0.0 and 1.0.
  - For biometric tandem deployments (automatic speaker verification combined with countermeasure detection), calibrated log-likelihood ratios (LLRs) are required to evaluate the official **tandem Detection Cost Function (t-DCF)**.
- **Techniques:**
  1. **Temperature Scaling:** Optimize a scalar temperature parameter $T > 0$ on the development split:
     $$\hat{p}_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
  2. **Log-Likelihood Ratio (LLR) Mapping:**
     $$\text{LLR}(x) = \log\left(\frac{P(\text{bonafide} \mid x)}{P(\text{spoof} \mid x)}\right) = z_0 - z_1$$
  3. **Biometric Threshold Tuning:** Determine the global decision threshold $\theta^*$ that minimizes the ASVspoof official cost metric:
     $$\min \text{t-DCF} = C_{\text{miss}} \cdot P_{\text{miss}}(\theta) \cdot P_{\text{target}} + C_{\text{fa}} \cdot P_{\text{fa}}(\theta) \cdot P_{\text{nontarget}}$$

---

## 4. Step-by-Step Strategic Execution Roadmap

To achieve maximal efficiency and avoid redundant GPU compute, follow this ordered phase execution:

```
+-----------------------------------------------------------------------------------+
|                            PHASE-BY-PHASE ROADMAP                                 |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [PHASE 1: BASELINE DIVERSITY]                                                    |
|   Step 1.1: Run Notebook 1 (Light-CNN + LFCC)               -> DONE (10.47% EER)  |
|   Step 1.2: Run Notebook 2 (SE-ResNet-18 + Mel)            -> RUNNING             |
|   Step 1.3: Run Notebook 3 (RawNet2-Mini + Raw Waveform)    -> NEXT EXECUTION     |
|                                                                                   |
|  [PHASE 2: ENSEMBLE CALIBRATION]                                                  |
|   Step 2.1: Extract dev & eval scores from all 3 models                           |
|   Step 2.2: Optimize convex fusion weights with Nelder-Mead                       |
|   Step 2.3: Benchmark fused EER and plot multi-model ROC/DET comparisons          |
|                                                                                   |
|  [PHASE 3: FOUNDATION MODEL UPGRADE]                                              |
|   Step 3.1: Build Notebook 4 (WavLM-Base-Plus + Attentive Pooling)                |
|   Step 3.2: Fine-tune on Kaggle GPU T4 x2                                         |
|   Step 3.3: Evaluate single-model EER (Target: <2.0%)                             |
|                                                                                   |
|  [PHASE 4: METRIC LEARNING & DOMAIN ROBUSTNESS]                                   |
|   Step 4.1: Implement AAM-Softmax (ArcFace) on the best performing backbone       |
|   Step 4.2: Integrate RawBoost augmentations in the training loop                 |
|                                                                                   |
|  [PHASE 5: SYSTEM EVALUATION & PUBLICATION BENCHMARK]                             |
|   Step 5.1: Run full 71,237-utterance Evaluation set inference                    |
|   Step 5.2: Generate final publication figures and comprehensive technical report |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

---

## 5. Technical Implementation Blueprints

### 5.1 Tri-Modal Ensemble Fusion Code Recipe

```python
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_curve

def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0)

def optimize_ensemble_weights(scores_matrix, targets):
    def objective(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        fused = np.dot(scores_matrix, w)
        return compute_eer(targets, fused)

    init_weights = [1.0 / scores_matrix.shape[1]] * scores_matrix.shape[1]
    bounds = [(0.0, 1.0)] * scores_matrix.shape[1]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(objective, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    optimal_weights = res.x / np.sum(res.x)
    optimal_eer = compute_eer(targets, np.dot(scores_matrix, optimal_weights))
    return optimal_weights, optimal_eer
```

### 5.2 Additive Angular Margin (AAM-Softmax) PyTorch Blueprint

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AAMSoftmax(nn.Module):
    def __init__(self, in_features, num_classes=2, s=30.0, m=0.35):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, labels):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output
```

### 5.3 WavLM Pre-Trained Backbone Pipeline Blueprint

```python
import torch
import torch.nn as nn
from transformers import WavLMModel

class WavLMDeepfakeClassifier(nn.Module):
    def __init__(self, num_classes=2, freeze_encoder=True):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        if freeze_encoder:
            self.wavlm.feature_extractor._freeze_parameters()
        self.num_layers = self.wavlm.config.num_hidden_layers
        self.layer_weights = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)
        self.fc_head = nn.Sequential(
            nn.Linear(768 * 2, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, waveforms):
        outputs = self.wavlm(waveforms, output_hidden_states=True)
        hidden_states = outputs.hidden_states[1:]
        stacked = torch.stack(hidden_states, dim=0)
        norm_weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        weighted_feat = torch.sum(stacked * norm_weights, dim=0)
        mean_pooled = torch.mean(weighted_feat, dim=1)
        std_pooled = torch.std(weighted_feat, dim=1)
        pooled = torch.cat([mean_pooled, std_pooled], dim=1)
        return self.fc_head(pooled)
```

---

## 6. Summary of Actionable Next Steps

1. **Monitor and Retrieve Model 2 Artifacts:**
   - Once `kaggle_02_se_resnet_mel.ipynb` finishes on Kaggle, download:
     - `se_resnet_mel_best.pth`
     - `se_resnet_mel_history.json`
     - All 16 figures from `/kaggle/working/figures/`
   - Store them in `notebook/kaggle_experiments/after training in kaggle/model2_se_resnet/`.

2. **Execute Model 3 on Kaggle:**
   - Upload [`notebook/kaggle_experiments/kaggle_03_rawnet_raw.ipynb`](file:///i:/ML%20Projects/voice-deepfake-defense-system/notebook/kaggle_experiments/kaggle_03_rawnet_raw.ipynb).
   - Ensure `awsaf49/asvpoof-2019-dataset` is attached.
   - Run on GPU T4 x2 or P100 (30 epochs, ~40-45 minutes).
   - Download `rawnet_raw_best.pth`, history JSON, and the 17 publication figures.

3. **Execute Experiment 4 (Tri-Modal Ensemble Fusion):**
   - Create the ensemble calibration notebook `kaggle_04_ensemble_fusion.ipynb` to fuse predictions from all three models using Nelder-Mead optimization and logit stacking.
   - This single step will capitalize on the strengths of all three signal representations to reduce overall system EER below 3.0%.

4. **Proceed to Foundation Models (Experiment 5):**
   - If target performance (<1.0% EER) requires additional discrimination, build and train the WavLM-Base-Plus transfer learning pipeline.
