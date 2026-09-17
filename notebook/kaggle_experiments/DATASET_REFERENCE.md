# ASVspoof 2019 Dataset — Complete Reference

**Kaggle Dataset:** https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset
**Official Paper:** Todisco et al., "ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection", INTERSPEECH 2019 (arXiv:1911.01601)
**Dataset Version:** 1 | **Total Size:** 25.3 GB | **License:** ODC Attribution License (ODC-By)

---

## 1. Kaggle Input Path

```
/kaggle/input/asvpoof-2019-dataset/
```

Dataset add করার পর এই path কাজ করবে। যদি "linked" dataset হয় তাহলে path আলাদা হতে পারে — notebooks-এ `find_la_root()` function সেটা automatically handle করে।

---

## 2. Complete Folder Structure

```
/kaggle/input/asvpoof-2019-dataset/
├── README.txt
├── asvspoof2019_evaluation_plan.pdf
├── asvspoof2019_Interspeech2019_submission.pdf
│
├── LA/                                              ← Logical Access (TTS + VC attacks)
│   ├── README.LA.txt
│   │
│   ├── ASVspoof2019_LA_cm_protocols/                ← Countermeasure protocol files
│   │   ├── ASVspoof2019.LA.cm.train.trn.txt         ← 25,380 lines
│   │   ├── ASVspoof2019.LA.cm.dev.trl.txt           ← 24,844 lines
│   │   └── ASVspoof2019.LA.cm.eval.trl.txt          ← 71,237 lines
│   │
│   ├── ASVspoof2019_LA_asv_protocols/               ← ASV protocols (not needed for CM task)
│   │   ├── ASVspoof2019.LA.asv.train.male.trn.txt
│   │   ├── ASVspoof2019.LA.asv.train.female.trn.txt
│   │   ├── ASVspoof2019.LA.asv.dev.trl.txt
│   │   └── ASVspoof2019.LA.asv.eval.trl.txt
│   │
│   ├── ASVspoof2019_LA_asv_scores/                  ← Pre-computed ASV backend scores
│   │   ├── ASVspoof2019.LA.asv.dev.gi.trl.scores.txt
│   │   └── ASVspoof2019.LA.asv.eval.gi.trl.scores.txt
│   │
│   ├── ASVspoof2019_LA_train/
│   │   └── flac/                                    ← 25,380 .flac files
│   │       ├── LA_T_1000137.flac
│   │       ├── LA_T_1000265.flac
│   │       └── ...   (format: LA_T_xxxxxxx.flac)
│   │
│   ├── ASVspoof2019_LA_dev/
│   │   └── flac/                                    ← 24,844 .flac files
│   │       └── ...   (format: LA_D_xxxxxxx.flac)
│   │
│   └── ASVspoof2019_LA_eval/
│       └── flac/                                    ← 71,237 .flac files
│           └── ...   (format: LA_E_xxxxxxx.flac)
│
└── PA/                                              ← Physical Access (replay attacks)
    ├── README.PA.txt
    ├── ASVspoof2019_PA_cm_protocols/
    │   ├── ASVspoof2019.PA.cm.train.trn.txt
    │   ├── ASVspoof2019.PA.cm.dev.trl.txt
    │   └── ASVspoof2019.PA.cm.eval.trl.txt
    ├── ASVspoof2019_PA_asv_protocols/
    ├── ASVspoof2019_PA_asv_scores/
    ├── ASVspoof2019_PA_train/flac/
    ├── ASVspoof2019_PA_dev/flac/
    └── ASVspoof2019_PA_eval/flac/
```

> এই project-এ শুধু **LA** (Logical Access) partition ব্যবহার করা হচ্ছে।

---

## 3. Audio Properties

| Property | Value |
|----------|-------|
| Container Format | FLAC (lossless compressed) |
| Sample Rate | **16,000 Hz** |
| Bit Depth | **16-bit** |
| Channels | **Mono** |
| Duration | 1–10 seconds (variable) |
| Source Corpus | VCTK (Voice Cloning Toolkit) |

**File naming convention:**

| Partition | Pattern | Example |
|-----------|---------|---------|
| Train | `LA_T_xxxxxxx.flac` | `LA_T_1138215.flac` |
| Dev | `LA_D_xxxxxxx.flac` | `LA_D_6349315.flac` |
| Eval | `LA_E_xxxxxxx.flac` | `LA_E_1001265.flac` |

---

## 4. Protocol File Format

**Space-separated, 5 columns per line:**

```
SPEAKER_ID    AUDIO_ID         ENV_ID   ATTACK_ID   KEY
LA_0079       LA_T_1138215     -        A01         spoof
LA_0069       LA_T_2436815     -        -           bonafide
```

| Position | Column Name | Description | Bonafide | Spoof |
|----------|-------------|-------------|----------|-------|
| `p[0]` | SPEAKER_ID | Speaker identifier | `LA_0069` | `LA_0079` |
| `p[1]` | AUDIO_ID | Audio filename (without `.flac`) | `LA_T_2436815` | `LA_T_1138215` |
| `p[2]` | ENV_ID | Environment ID — always `-` for LA | `-` | `-` |
| `p[3]` | ATTACK_ID | Spoofing system ID | `-` | `A01`–`A19` |
| `p[4]` | KEY | Ground truth label | `bonafide` | `spoof` |

**File path construction:**
```python
file_path = os.path.join(FLAC_DIRS[partition], audio_id + ".flac")
```

---

## 5. Utterance Counts

### Per Partition

| Partition | Bonafide | Spoof | Total | Spoof:Bonafide Ratio |
|-----------|----------|-------|-------|----------------------|
| Train | 2,580 | 22,800 | **25,380** | 8.8 : 1 |
| Dev | 2,548 | 22,296 | **24,844** | 8.7 : 1 |
| Eval | 7,355 | 63,882 | **71,237** | 8.7 : 1 |
| **Total** | **12,483** | **108,978** | **121,461** | 8.7 : 1 |

### Train Spoof — Per Attack ID

| Attack ID | Type | Algorithm | Utterances |
|-----------|------|-----------|------------|
| A01 | TTS | AR RNN + WaveNet vocoder | 3,800 |
| A02 | TTS | AR RNN + WORLD vocoder | 3,800 |
| A03 | TTS | Feed-Forward NN + WORLD vocoder | 3,800 |
| A04 | TTS | CART + Waveform concatenation | 3,800 |
| A05 | VC | Voice Conversion | 3,800 |
| A06 | VC | Voice Conversion | 3,800 |
| **Total spoof** | | | **22,800** |

> প্রতিটি attack-এ ঠিক 3,800 utterance — perfectly balanced across attack types.

---

## 6. Eval Set Attack Algorithms (A07–A19)

Training-এ A01–A06 দেখা হয়, Eval-এ A07–A19 mostly unseen:

| Attack | Type | Acoustic Model | Waveform Generator | Status |
|--------|------|----------------|--------------------|--------|
| A07 | TTS | RNN | WORLD + GAN | Unknown |
| A08 | TTS | AR RNN | Neural source-filter | Unknown |
| A09 | TTS | RNN | Vocaine vocoder | Unknown |
| A10 | TTS | AR RNN + CNN | WaveRNN | Unknown |
| A11 | TTS | AR RNN + CNN | Griffin-Lim | Unknown |
| A12 | TTS | RNN | WaveNet | Unknown |
| A13 | VC | — | Waveform filtering (Moment matching) | Unknown |
| A14 | VC | RNN | STRAIGHT vocoder | Unknown |
| A15 | VC | RNN | WaveNet | Unknown |
| A16 | TTS | CART | Waveform concatenation | **Known** (= A04 algorithm) |
| A17 | VC | VAE | Waveform filtering | Unknown |
| A18 | VC | Linear PLDA | MFCC vocoder | Unknown |
| A19 | VC | GMM-UBM | LPC Spectral filtering + OLA | **Known** (= A06 algorithm) |

> A16 ≈ A04, A19 ≈ A06 — same algorithm, different recording conditions.
> এই unseen attack generalization-ই মূল challenge।

---

## 7. Speaker Distribution

| Partition | Total Speakers | Male | Female |
|-----------|---------------|------|--------|
| Train | 20 | 8 | 12 |
| Dev | 10 | 4 | 6 |
| Eval | 48 | 21 | 27 |
| **Total** | **78 (disjoint)** | **33** | **45** |

> **Zero speaker overlap** across Train/Dev/Eval — speaker-independent evaluation guaranteed.
> Source: VCTK corpus — 107 total speakers (46 male, 61 female).

---

## 8. Evaluation Metrics

| Metric | Description | Goal |
|--------|-------------|------|
| **EER** (Equal Error Rate) | Point where False Accept Rate = False Reject Rate | Minimize |
| **min-t-DCF** | Tandem Detection Cost Function — official challenge metric combining CM + ASV | Minimize |
| **AUC-ROC** | Area under ROC curve | Maximize |

**Published baseline results (LA track):**

| System | Dev EER | Eval EER |
|--------|---------|----------|
| LFCC-GMM (official baseline) | 8.09% | 13.54% |
| CQCC-GMM (official baseline) | 9.87% | 11.04% |
| Top challenge systems | < 1.0% | 1–3% |

---

## 9. Key Modeling Challenges

1. **Class imbalance (8.8:1)** — Needs `WeightedRandomSampler` or `FocalLoss`
2. **Unseen attack generalization** — Train on A01–A06, test on A07–A19
3. **Speaker-independent** — Completely disjoint speakers across splits
4. **Variable-length audio** — Must pad/crop to fixed length (4.0s = 64,000 samples @ 16kHz)
5. **Subtle artifacts** — Vocoder fingerprints, unnatural prosody, phase discontinuities

---

## 10. Recommended Hyperparameters (GPU — Kaggle T4/P100)

| Setting | Light-CNN + LFCC | SE-ResNet-18 + Mel | RawNet-Mini + Raw |
|---------|------------------|--------------------|-------------------|
| Input shape | `(1, 60, 251)` | `(1, 128, 251)` | `(1, 64000)` |
| Batch size | 128 | 64 | 32 |
| Learning rate | 1e-3 | 5e-4 | 1e-4 |
| LR min | 1e-6 | 1e-6 | 1e-6 |
| Weight decay | 1e-4 | 1e-4 | 1e-4 |
| Epochs | 30 | 30 | 30 |
| Scheduler | CosineAnnealingWarmRestarts(T_0=10, T_mult=2) | same | same |
| Augmentation | SpecAugment (time+freq mask) | SpecAugment | none |
| Loss | FocalLoss(α=0.75, γ=2.0, label_smooth=0.05) | same | same |
| Mixed precision (AMP) | Yes (GPU only) | Yes | Yes |
| DataLoader workers | 2 | 2 | 2 |

**Output files per notebook (saved to `/kaggle/working/`):**
- `{name}_best.pth` — best model checkpoint (lowest dev EER)
- `{name}_history.json` — per-epoch train loss, dev EER, dev AUC
- `{name}_curve.png` — training curves plot
