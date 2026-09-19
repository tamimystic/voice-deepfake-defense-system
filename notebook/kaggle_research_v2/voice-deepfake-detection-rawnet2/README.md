# Voice Deepfake Detection with RawNet2-Mini and Raw Audio Waveforms
## ASVspoof 2019 Logical Access Benchmark: Full Tri-Partition Evaluation (Train, Dev, and Eval)

---

## 1. Executive Summary

This study formulates an end-to-end deep learning framework based on **RawNet2-Mini** operating directly on the raw 1D acoustic waveform to capture phase inconsistencies, pitch micro-jitter, and vocoder timing irregularities that are discarded by standard Fourier magnitude spectrograms.

### 1.1 Resolution of Previous Experimental Limitations
* **The Closed-World Limitation**: Previous raw-waveform experiments evaluated solely on the Development partition (A01-A06). Because the Development partition shares the identical generative attack algorithms as the training partition (differing only by speaker identities), reported error rates reflected closed-world memorization rather than true out-of-distribution generalization.
* **The Open-World Solution**: This notebook executes an automated, single-click end-to-end workflow in Kaggle. In addition to Development partition validation, it executes exhaustive batch inference over **all 71,237 utterances of the official ASVspoof 2019 Evaluation partition**, rigorously auditing performance against thirteen distinct attack mechanisms (**A07 through A19**), of which eleven are completely unseen out-of-distribution generative architectures.

---

## 2. Technical Architecture

### 2.1 Acoustic Signal Processing Pipeline
* **Input Signal**: 16 kHz, 16-bit linear PCM FLAC audio.
* **Length Normalization**: Fixed 4.00-second window (64,000 samples) with random cropping for training and center cropping for evaluation.
* **Peak Scaling**: Signal amplitude normalized to $[-1.0, 1.0]$ via $x / (\max(|x|) + 10^{-7})$.
* **Data Augmentation**: On-the-fly random amplitude scaling ($[0.7, 1.2]$) and additive stationary Gaussian noise ($SNR \sim [25, 45]\text{ dB}$) during training.

### 2.2 Parameterized Sinc-Convolutions (SincNet)
* **First Layer**: 128 parameterized bandpass sinc filters (kernel length $L=251$) initialized along the Mel frequency scale between 30 Hz and 7,900 Hz:
  $$g(t, f_1, f_2) = 2f_2 \text{sinc}(2\pi f_2 t) - 2f_1 \text{sinc}(2\pi f_1 t)$$
  $$w(t) = 0.54 - 0.46 \cos(2\pi t / L)$$
* **Envelope Detection**: Instantaneous full-wave absolute value rectification $|x|$ captures high-frequency vocoder excitation boundaries.

### 2.3 Feature Map Scaling (FMS) Residual Backbone
* **Residual Depth**: Six 1D residual blocks arranged across channel widths of 128, 256, and 512 channels.
* **Feature Map Scaling (FMS)**: Squeeze operation via Global Average Pooling coupled with a Sigmoid excitation layer that dynamically recalibrates channel activations:
  $$\mathbf{s} = \sigma(\mathbf{W} \cdot \text{GAP}(\mathbf{X}))$$
  $$\widetilde{\mathbf{X}}_c = s_c \cdot \mathbf{X}_c$$
* **Dual Pooling**: Concatenated Global Average Pooling and Global Max Pooling yielding a 1,024-dimensional feature vector, projected into a 64-dimensional latent embedding.

### 2.4 Optimization & Sampling
* **Focal Loss with Label Smoothing ($\alpha=0.75, \gamma=2.0, \epsilon=0.05$)**:
  $$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
* **WeightedRandomSampler**: Counteracts the 1:8.7 class imbalance by oversampling authentic human speech.
* **Optimizer**: AdamW ($lr=1 \times 10^{-4}, \text{weight\_decay}=10^{-4}$) with CosineAnnealingWarmRestarts.
* **Mixed Precision**: Accelerated with `torch.amp.autocast`.

---

## 3. Dataset Audit & Tri-Partition Topology

| Partition | Role in Pipeline | Utterance Count | Authentic (Bonafide) | Synthetic (Spoof) | Imbalance Ratio | Attack IDs Present |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | Model Parameter Optimization | 25,380 | 2,580 | 22,800 | 8.84 : 1 | A01, A02, A03, A04, A05, A06 |
| **Dev** | Checkpoint Selection & Tuning | 24,844 | 2,548 | 22,296 | 8.75 : 1 | A01, A02, A03, A04, A05, A06 |
| **Eval** | Out-of-Distribution Acid Test | 71,237 | 7,355 | 63,882 | 8.69 : 1 | A04, A06, A07 - A19 (11 unseen architectures) |
| **Total** | Unified Database Manifest | **121,461** | **12,483** | **108,978** | **8.73 : 1** | **All 19 Attack Families** |

---

## 4. Instructions for Kaggle Deployment

### Step 1: Create a New Kaggle Notebook
1. Go to Kaggle and select **New Notebook**.
2. Title the notebook: `voice-deepfake-detection-rawnet2`.
3. In the right panel, set **Accelerator** to **GPU P100** or **GPU T4 x2**.
4. Enable **Internet** in the notebook settings.

### Step 2: Attach the Dataset
1. Click **+ Add Input** in the top right sidebar.
2. Search for `asvpoof-2019-dataset` (by awsaf49).
3. Click the **+** button to attach the dataset to `/kaggle/input/`.

### Step 3: Upload and Execute
1. Click **File -> Upload Notebook** and select `voice-deepfake-detection-rawnet2.ipynb`.
2. Click **Save Version -> Save & Run All (Commit)**.
3. Execution proceeds automatically from start to finish. Estimated runtime is approximately 45 to 55 minutes on GPU.

---

## 5. Deliverables & Diagnostic Figure Inventory

All artifacts are saved into `/kaggle/working/`:

### 5.1 Serialized Models and Data Records
* `/kaggle/working/models/rawnet2_best.pth`: Optimal checkpoint weights saved at lowest Development EER.
* `/kaggle/working/models/training_history.json`: Per-epoch loss, Dev EER, AUC, and threshold logs.
* `/kaggle/working/experiment_final_report.json`: Structured benchmark report.
* `/kaggle/working/attack_vulnerability_breakdown.csv`: Forensic detection accuracy across all 19 attacks (A01 through A19).

### 5.2 16 Standalone Publication Figures (Saved at 300 DPI in `/kaggle/working/figures/`)
1. `01_class_distribution_breakdown.png`: Utterance distribution and class imbalance ratio across Train, Dev, and Eval.
2. `02_attack_distribution_analysis.png`: Attack distribution comparing known training attacks (A01-A06) vs unseen evaluation attacks (A07-A19).
3. `03_audio_duration_length_distribution.png`: Speech duration histogram and 16 kHz sampling rate audit.
4. `04_waveform_time_domain_glottal_inspection.png`: Time-domain waveforms and 100ms glottal pulse comparisons (Authentic vs WaveNet vs STRAIGHT).
5. `05_sincnet_learned_filterbank_frequency_response.png`: Parameterized SincNet bandpass filter frequency responses across the 0-8000 Hz spectrum.
6. `06_instantaneous_energy_envelope_detection.png`: Full-wave rectified instantaneous energy envelopes across authentic vs synthetic speech.
7. `07_time_domain_signal_preprocessing_pipeline.png`: Visual transformation from raw audio to normalized 64,000-sample tensor and SincNet activation map.
8. `08_training_loss_and_dev_eer_trajectories.png`: Focal loss convergence and Development EER validation trajectory across epochs.
9. `09_receiver_operating_characteristic_roc.png`: ROC curves comparing Development vs Evaluation performance with EER operating points.
10. `10_detection_error_tradeoff_det.png`: Normal deviate scale DET curves (False Alarm Rate vs Miss Rate).
11. `11_precision_recall_curves.png`: Precision-Recall curves highlighting positive predictive performance.
12. `12_normalized_confusion_matrix_eval.png`: Normalized confusion matrix evaluated over all 71,237 utterances of the Evaluation partition.
13. `13_attack_by_attack_accuracy_barchart.png`: Granular detection accuracy bar chart across all nineteen attack algorithms.
14. `14_tsne_latent_manifold_clusters.png`: 2D t-SNE projection of the 64-dimensional latent manifold showing separation of authentic, known spoof, and unseen spoof clusters.
15. `15_temporal_saliency_gradient_attribution.png`: 1D time-domain gradient saliency attribution highlighting exact temporal locations of vocoder artifacts.
16. `16_single_file_live_inference_verification.png`: Confidence and posterior probability benchmark for live single-file inference verification.
