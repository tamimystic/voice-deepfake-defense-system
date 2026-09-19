# End-to-End Voice Deepfake Detection with Squeeze-and-Excitation ResNet-18
## Full Tri-Partition Evaluation on ASVspoof 2019 Logical Access (Train, Dev, and Eval)

---

## 1. Executive Summary

This study formulates an end-to-end deep learning framework based on **SE-ResNet-18 (Squeeze-and-Excitation Residual Network)** to overcome the fundamental generalization bottleneck in voice deepfake detection. 

### Resolving Previous Experimental Limitations
* **The Closed-World Limitation**: Previous experiments trained acoustic models on the ASVspoof 2019 LA Training partition (A01-A06) and evaluated solely on the Development partition (A01-A06). Because the Development set shares the identical generative attack algorithms as the training partition (differing only by speaker identities), the reported sub-0.1% EER scores reflected closed-world memorization rather than true out-of-distribution generalization.
* **The Open-World Solution**: This notebook executes a complete, uninterrupted end-to-end workflow in a single Kaggle session. In addition to Development partition tracking, it executes exhaustive batch inference over **all 71,237 utterances of the official ASVspoof 2019 Evaluation partition**, rigorously auditing performance against thirteen distinct attack mechanisms (**A07 through A19**), of which eleven are completely unseen out-of-distribution generative architectures.

---

## 2. Architectural Formulation

### 2.1 GPU-Accelerated Front-End Transformation
* **Input Signal**: 1D raw audio waveform ($x \in \mathbb{R}^{64000}$, corresponding to 4.00 seconds of 16 kHz mono speech).
* **Spectrogram Projection**: 80-channel Mel filterbanks extracted directly on GPU tensor streams via `torchaudio.transforms.MelSpectrogram` (FFT size: 512, window length: 400 samples, hop length: 160 samples, frequency range: 20 Hz - 8,000 Hz).
* **Log Dynamic Range Compression & Instance Normalization**:
  $$\mathbf{X}_{\text{Log-Mel}} = \log(\mathbf{X}_{\text{Mel}} + 10^{-6})$$
  $$\widetilde{\mathbf{X}} = \frac{\mathbf{X}_{\text{Log-Mel}} - \mu(\mathbf{X}_{\text{Log-Mel}})}{\sigma(\mathbf{X}_{\text{Log-Mel}}) + 10^{-5}}$$
* **SpecAugment Regularization**: On-the-fly random frequency masking (up to 8 bins) and time masking (up to 24 frames) during training.

### 2.2 Squeeze-and-Excitation Residual Backbone
* **Layer Depth**: 18 parameterized convolutional layers arranged into four residual stages (64, 128, 256, and 512 feature channels).
* **Channel Recalibration (SE Blocks)**:
  * **Squeeze**: Global average pooling compresses spatial/temporal dimensions into a channel descriptor vector $z \in \mathbb{R}^C$:
    $$z_c = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W X_c(i, j)$$
  * **Excitation**: Two-layer bottleneck MLP with dimensionality reduction factor $r=16$, parameterized by weights $\mathbf{W}_1 \in \mathbb{R}^{\frac{C}{r} \times C}$ and $\mathbf{W}_2 \in \mathbb{R}^{C \times \frac{C}{r}}$:
    $$\mathbf{s} = \sigma(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \mathbf{z}))$$
  * **Scale**: Channel-wise multiplication $\widetilde{X}_c = s_c \cdot X_c$ amplifies discriminative artifact bands while suppressing uninformative vocal tract resonances.
* **Latent Space Projection**: Global adaptive average pooling coupled with a 128-dimensional linear latent embedding layer with Batch Normalization.

### 2.3 Loss Optimization & Sampling
* **Focal Loss with Label Smoothing ($\alpha=0.75, \gamma=2.0, \epsilon=0.05$)**:
  $$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
  Prevents the network from over-fitting to easily classified authentic utterances while penalizing ambiguous synthetic boundaries.
* **WeightedRandomSampler**: Dynamically oversamples authentic human speech to counterbalance the 1:9 class imbalance inherent in the ASVspoof protocol.
* **Cosine Annealing Optimizer**: AdamW with base learning rate $1 \times 10^{-3}$, minimum learning rate $1 \times 10^{-6}$, and weight decay $1 \times 10^{-4}$.

---

## 3. Dataset Audit & Tri-Partition Topology

| Partition | Role in Pipeline | Utterance Count | Authentic (Bonafide) | Synthetic (Spoof) | Imbalance Ratio | Attack IDs Present |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | Model Parameter Optimization | 25,380 | 2,580 | 22,800 | 8.84 : 1 | A01, A02, A03, A04, A05, A06 |
| **Dev** | Checkpoint Selection & Tuning | 24,844 | 2,548 | 22,296 | 8.75 : 1 | A01, A02, A03, A04, A05, A06 |
| **Eval** | Out-of-Distribution Acid Test | 71,237 | 7,355 | 63,882 | 8.69 : 1 | A07 - A19 (11 unseen architectures) |
| **Total** | Unified Database Manifest | **121,461** | **12,483** | **108,978** | **8.73 : 1** | **All 19 Attack Families** |

---

## 4. Instructions for Kaggle Deployment

### Step 1: Create a New Kaggle Notebook
1. Navigate to Kaggle and click **New Notebook**.
2. Set the Notebook Runtime to **GPU** (Accelerator: GPU P100 or GPU T4 x2).
3. Ensure Internet access is turned on in the Settings panel.

### Step 2: Attach the ASVspoof 2019 Dataset
1. In the right sidebar, click **+ Add Input**.
2. Search for `asvpoof-2019-dataset` (by awsaf49) or `asvspoof-2019-dataset`.
3. Click the **+** button to attach the dataset.
4. Verify that `/kaggle/input/asvpoof-2019-dataset/` is listed under input data.

### Step 3: Upload and Execute the Notebook
1. Click **File -> Upload Notebook** and select `kaggle_01_se_resnet_mel_end_to_end.ipynb`.
2. Click **Run All** or click **Save Version -> Save & Run All (Commit)**.
3. Execution will proceed automatically from start to finish. Estimated runtime is approximately 45-60 minutes on GPU T4 or P100.

---

## 5. Output Deliverables & Diagnostic Figures

The entire pipeline saves all artifacts into `/kaggle/working/`:

### 5.1 Trained Model Checkpoints & Serialization
* `/kaggle/working/models/se_resnet18_best.pth`: Optimal model weights saved at the lowest Development EER operating point.
* `/kaggle/working/models/training_history.json`: Epoch-by-epoch loss, Dev EER, AUC, and threshold logs.
* `/kaggle/working/experiment_final_report.json`: Comprehensive machine-readable benchmark summary.
* `/kaggle/working/attack_vulnerability_breakdown.csv`: Tabular accuracy and mean spoof scores across all nineteen individual attacks (A01 through A19).

### 5.2 Standalone Publication Figures (Saved at 300 DPI in `/kaggle/working/figures/`)
1. `01_class_distribution_breakdown.png`: Bar chart comparison of authentic vs synthetic utterances across Train, Dev, and Eval.
2. `02_attack_distribution_analysis.png`: Utterance distribution of known training attacks (A01-A06) vs unseen evaluation attacks (A07-A19).
3. `03_audio_duration_length_distribution.png`: Histogram of raw audio durations and 16 kHz sample rate verification.
4. `04_waveform_time_domain_comparison.png`: Dual-panel time-domain waveform and 100ms glottal pulse comparison between authentic and synthetic voice.
5. `05_power_spectral_density_frequency_rolloff.png`: Welch power spectral density curves before and after length normalization and peak scaling.
6. `06_log_mel_spectrogram_representations.png`: 80-bin Log-Mel spectrograms comparing authentic speech, neural TTS, and voice conversion.
7. `07_audio_preprocessing_pipeline.png`: Visual progression from raw FLAC audio to normalized Log-Mel spectrogram.
8. `08_training_loss_and_eer_curves.png`: Training loss convergence and Development EER trajectory across all epochs.
9. `09_receiver_operating_characteristic_roc.png`: Benchmark ROC curves comparing Development vs Evaluation partitions.
10. `10_detection_error_tradeoff_det.png`: Normal deviate scale DET curves illustrating False Alarm Rate vs Miss Rate.
11. `11_precision_recall_curves.png`: Precision-Recall curves highlighting positive predictive performance under severe class imbalance.
12. `12_normalized_confusion_matrix_eval.png`: Normalized confusion matrix evaluated over all 71,237 utterances of the Evaluation partition.
13. `13_attack_by_attack_accuracy_barchart.png`: Granular detection accuracy bar chart across all nineteen attack algorithms.
14. `14_tsne_latent_manifold_clusters.png`: 2D t-SNE projection of the 128-dimensional latent manifold showing separation of authentic, known spoof, and unseen spoof clusters.
15. `15_gradcam_spectro_temporal_explainability.png`: Grad-CAM saliency heatmaps highlighting spectro-temporal regions driving model classification.
16. `16_single_file_inference_verification.png`: Confidence and posterior probability benchmark for single-file live inference demo.
