import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest, torch, numpy as np
from src.features.preprocessor import SignalPreprocessor
from src.features.spectral import SpectralFeatureExtractor
from src.features.cepstral import CepstralFeatureExtractor
from src.models.light_cnn import LightCNN
from src.models.resnet_spec import SEResNetSpectrogram
from src.models.rawnet_mini import RawNetMini
from src.inference.predictor import ForensicPredictor
from src.utils.config import path_config
import pandas as pd

class TestAudioForensicsPipeline(unittest.TestCase):
    def setUp(self):
        self.preprocessor = SignalPreprocessor()
        self.spectral = SpectralFeatureExtractor()
        self.cepstral = CepstralFeatureExtractor()
        self.predictor = ForensicPredictor()
        manifest_path = os.path.join(path_config.data_processed, "protocol_manifest.parquet")
        self.df = pd.read_parquet(manifest_path)

    def test_preprocessor_shape(self):
        dummy_audio = np.random.randn(80000).astype(np.float32)
        out = self.preprocessor.process_pipeline(dummy_audio, is_training=False)
        self.assertEqual(len(out), 64000)
        self.assertTrue(np.max(np.abs(out)) <= 1.0)

    def test_spectral_extractor_shape(self):
        dummy_audio = np.random.randn(64000).astype(np.float32)
        mel = self.spectral.extract_log_mel(dummy_audio, sr=16000)
        self.assertEqual(mel.shape[0], 128)

    def test_cepstral_extractor_shape(self):
        dummy_audio = np.random.randn(64000).astype(np.float32)
        lfcc = self.cepstral.extract_lfcc(dummy_audio, sr=16000)
        self.assertEqual(lfcc.shape[0], 60)

    def test_model_forward_passes(self):
        m_light = LightCNN(in_channels=1, num_classes=2)
        m_resnet = SEResNetSpectrogram(in_channels=1, num_classes=2)
        m_raw = RawNetMini(in_channels=1, num_classes=2)

        t_lfcc = torch.randn(2, 1, 60, 251)
        t_mel = torch.randn(2, 1, 128, 251)
        t_raw = torch.randn(2, 1, 64000)

        self.assertEqual(m_light(t_lfcc).shape, (2, 2))
        self.assertEqual(m_resnet(t_mel).shape, (2, 2))
        self.assertEqual(m_raw(t_raw).shape, (2, 2))

    def test_end_to_end_file_prediction(self):
        sample_path = self.df.iloc[0]["file_path"]
        res = self.predictor.predict_from_file(sample_path)
        self.assertIn("prediction", res)
        self.assertIn("spoof_probability", res)
        self.assertIn("model_breakdown", res)
        self.assertIn(res["prediction"], ["bonafide", "spoof"])

if __name__ == "__main__":
    unittest.main()
