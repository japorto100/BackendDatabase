"""
Tests für Tensor-Operationen der Fusion-Strategien
"""

import unittest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from models_app.fusion.tensor_ops import FusionTensorOps

class TestFusionTensorOps(unittest.TestCase):
    """Test-Klasse für FusionTensorOps"""

    def setUp(self):
        """Setup für Tests"""
        # Erstelle eine Test-Instanz
        self.tensor_ops = FusionTensorOps(device='cpu')
        
        # Erstelle Test-Tensoren
        self.visual_features = torch.rand(2, 5, 768)
        self.text_features = torch.rand(2, 10, 512)
    
    def test_early_fusion(self):
        """Test für early_fusion Methode"""
        # Teste mit 3D-Tensoren
        result = self.tensor_ops.early_fusion(
            self.visual_features, 
            self.text_features,
            projection_dim=1024
        )
        
        # Überprüfe die Ausgabe
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 2)  # Batch-Size beibehalten
        self.assertEqual(result.shape[1], 1024)  # Projektion auf 1024
        
        # Teste mit 2D-Tensoren
        v_features_2d = torch.rand(2, 768)
        t_features_2d = torch.rand(2, 512)
        result = self.tensor_ops.early_fusion(
            v_features_2d, 
            t_features_2d,
            projection_dim=1024
        )
        
        # Überprüfe die Ausgabe
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.shape[1], 1024)
        
        # Teste mit NumPy-Arrays
        v_features_np = np.random.rand(2, 768)
        t_features_np = np.random.rand(2, 512)
        result = self.tensor_ops.early_fusion(
            v_features_np, 
            t_features_np,
            projection_dim=1024
        )
        
        # Überprüfe die Ausgabe
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.shape[1], 1024)
    
    def test_late_fusion(self):
        """Test für late_fusion Methode"""
        # Teste mit 3D-Tensoren
        result = self.tensor_ops.late_fusion(
            self.visual_features, 
            self.text_features,
            visual_weight=0.3,
            text_weight=0.7,
            projection_dim=1024
        )
        
        # Überprüfe die Ausgabe
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 2)  # Batch-Size beibehalten
        self.assertEqual(result.shape[1], 1024)  # Projektion auf 1024
    
    def test_attention_fusion(self):
        """Test für attention_fusion Methode"""
        # Teste mit 3D-Tensoren
        result = self.tensor_ops.attention_fusion(
            self.visual_features, 
            self.text_features,
            num_heads=4,
            dropout=0.1
        )
        
        # Überprüfe die Ausgabe
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 2)  # Batch-Size beibehalten
    
    def test_compute_fusion_confidence(self):
        """Test für compute_fusion_confidence Methode"""
        # Erstelle Test-Tensoren
        v_features = torch.rand(2, 768)
        t_features = torch.rand(2, 512)
        f_features = torch.rand(2, 1024)
        
        # Teste die Methode
        confidence = self.tensor_ops.compute_fusion_confidence(
            v_features, t_features, f_features
        )
        
        # Überprüfe die Ausgabe
        self.assertIsInstance(confidence, float)
        self.assertTrue(0 <= confidence <= 1)
    
    def test_to_tensor(self):
        """Test für _to_tensor Methode"""
        # Teste mit Torch-Tensor
        tensor = torch.rand(2, 768)
        result = self.tensor_ops._to_tensor(tensor)
        self.assertIs(result.device.type, 'cpu')
        
        # Teste mit NumPy-Array
        np_array = np.random.rand(2, 768)
        result = self.tensor_ops._to_tensor(np_array)
        self.assertIsInstance(result, torch.Tensor)
        self.assertIs(result.device.type, 'cpu')
        
        # Teste mit nicht unterstütztem Typ
        with self.assertRaises(TypeError):
            self.tensor_ops._to_tensor([1, 2, 3])
    
    def test_ensure_2d(self):
        """Test für _ensure_2d Methode"""
        # Teste mit 1D-Tensor
        tensor_1d = torch.rand(768)
        result = self.tensor_ops._ensure_2d(tensor_1d)
        self.assertEqual(len(result.shape), 2)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], 768)
        
        # Teste mit 2D-Tensor
        tensor_2d = torch.rand(2, 768)
        result = self.tensor_ops._ensure_2d(tensor_2d)
        self.assertEqual(len(result.shape), 2)
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.shape[1], 768)
        
        # Teste mit 3D-Tensor
        tensor_3d = torch.rand(2, 5, 768)
        result = self.tensor_ops._ensure_2d(tensor_3d)
        self.assertEqual(len(result.shape), 2)
        self.assertEqual(result.shape[0], 2)

if __name__ == '__main__':
    unittest.main()