"""
Tests für die HybridFusion-Klasse.
"""

import unittest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from models_app.fusion.hybrid_fusion import HybridFusion
from models_app.fusion.base import FusionStrategy, EarlyFusion, LateFusion, AttentionFusion

class TestHybridFusion(unittest.TestCase):
    """
    Testklasse für HybridFusion.
    """
    
    def setUp(self):
        """
        Initialisiert die Testumgebung.
        """
        self.hybrid_fusion = HybridFusion()
        
        # Erstelle Mock-Features für Tests
        self.visual_features = np.random.rand(10, 512).astype(np.float32)
        self.text_features = np.random.rand(5, 768).astype(np.float32)
        
        # Erstelle Metadaten für Tests
        self.metadata = {
            "document_type": "scientific",
            "language": "en",
            "confidence": 0.9
        }
    
    def test_initialization(self):
        """
        Testet die korrekte Initialisierung der HybridFusion-Klasse.
        """
        # Überprüfe, ob alle Strategien initialisiert wurden
        self.assertIn("early", self.hybrid_fusion.fusion_strategies)
        self.assertIn("late", self.hybrid_fusion.fusion_strategies)
        self.assertIn("attention", self.hybrid_fusion.fusion_strategies)
        
        # Überprüfe, ob die Performance-Tracking-Strukturen initialisiert wurden
        self.assertIn("early", self.hybrid_fusion.performance_history)
        self.assertIn("late", self.hybrid_fusion.performance_history)
        self.assertIn("attention", self.hybrid_fusion.performance_history)
        
        # Überprüfe, ob die Performance-Metriken initialisiert wurden
        self.assertIn("early", self.hybrid_fusion.performance_metrics)
        self.assertIn("late", self.hybrid_fusion.performance_metrics)
        self.assertIn("attention", self.hybrid_fusion.performance_metrics)
    
    def test_predict_best_strategy(self):
        """
        Testet die Vorhersage der besten Fusionsstrategie.
        """
        # Patch die get_confidence-Methode, um kontrollierte Werte zurückzugeben
        with patch.object(EarlyFusion, 'get_confidence', return_value=0.7), \
             patch.object(LateFusion, 'get_confidence', return_value=0.8), \
             patch.object(AttentionFusion, 'get_confidence', return_value=0.6):
            
            # Ohne Metadaten sollte Late Fusion die beste sein (0.8)
            best_strategy, confidence = self.hybrid_fusion.predict_best_strategy(
                self.visual_features, self.text_features
            )
            self.assertEqual(best_strategy, "late")
            self.assertAlmostEqual(confidence, 0.8)
            
            # Mit scientific document_type sollte Attention Fusion bevorzugt werden
            # (0.6 * 1.2 = 0.72, was immer noch kleiner als Late Fusion mit 0.8 ist)
            best_strategy, confidence = self.hybrid_fusion.predict_best_strategy(
                self.visual_features, self.text_features, {"document_type": "scientific"}
            )
            self.assertEqual(best_strategy, "late")
            
            # Mit business document_type sollte Late Fusion noch stärker bevorzugt werden
            # (0.8 * 1.2 = 0.96)
            best_strategy, confidence = self.hybrid_fusion.predict_best_strategy(
                self.visual_features, self.text_features, {"document_type": "business"}
            )
            self.assertEqual(best_strategy, "late")
            self.assertAlmostEqual(confidence, 0.96)
    
    def test_fuse_with_best_strategy(self):
        """
        Testet die Fusion mit der besten Strategie.
        """
        # Mock für die Fusion-Methode
        mock_fused_features = np.random.rand(10, 1024).astype(np.float32)
        
        # Patch die relevanten Methoden
        with patch.object(self.hybrid_fusion, 'predict_best_strategy', 
                         return_value=("early", 0.9)), \
             patch.object(EarlyFusion, 'fuse', return_value=mock_fused_features):
            
            # Führe die Fusion durch
            fused, strategy, conf = self.hybrid_fusion.fuse_with_best_strategy(
                self.visual_features, self.text_features, self.metadata
            )
            
            # Überprüfe die Ergebnisse
            self.assertEqual(strategy, "early")
            self.assertEqual(conf, 0.9)
            np.testing.assert_array_equal(fused, mock_fused_features)
            
            # Überprüfe, ob Performance-Metriken aktualisiert wurden
            self.assertEqual(len(self.hybrid_fusion.performance_metrics["early"]["time"]), 1)
            self.assertEqual(len(self.hybrid_fusion.performance_metrics["early"]["memory"]), 1)
            self.assertEqual(len(self.hybrid_fusion.performance_metrics["early"]["quality"]), 1)
            
            # Überprüfe, ob Performance-History aktualisiert wurde
            self.assertEqual(len(self.hybrid_fusion.performance_history["early"]), 1)
            self.assertIn("time", self.hybrid_fusion.performance_history["early"][0])
            self.assertIn("memory", self.hybrid_fusion.performance_history["early"][0])
            self.assertIn("quality", self.hybrid_fusion.performance_history["early"][0])
            self.assertIn("document_type", self.hybrid_fusion.performance_history["early"][0])
    
    def test_calculate_quality_metric(self):
        """
        Testet die Berechnung der Qualitätsmetrik.
        """
        # Erstelle ein Feature-Array mit bekannter Varianz
        features = np.ones((10, 10)) * 5
        features[0, 0] = 10  # Füge etwas Varianz hinzu
        
        # Berechne die Qualitätsmetrik
        quality = self.hybrid_fusion._calculate_quality_metric(features, None, None)
        
        # Die Varianz sollte klein, aber nicht Null sein
        self.assertGreater(quality, 0)
        self.assertLess(quality, 1)
        
        # Teste mit einem Feature-Dictionary
        feature_dict = {"features": features}
        quality = self.hybrid_fusion._calculate_quality_metric(feature_dict, None, None)
        self.assertGreater(quality, 0)
        self.assertLess(quality, 1)
        
        # Teste Fehlerbehandlung
        quality = self.hybrid_fusion._calculate_quality_metric("not_a_feature", None, None)
        self.assertEqual(quality, 0.7)  # Sollte den Fallback-Wert zurückgeben
    
    def test_fuse_with_ensemble(self):
        """
        Testet die Ensemble-Fusion.
        """
        # Mock für die Fusion-Ergebnisse
        mock_results = {
            "early": np.random.rand(10, 512).astype(np.float32),
            "late": np.random.rand(10, 512).astype(np.float32),
            "attention": np.random.rand(10, 512).astype(np.float32)
        }
        
        # Mock für die Gewichte
        mock_weights = {
            "early": 0.3,
            "late": 0.5,
            "attention": 0.2
        }
        
        # Mock für das Ensemble-Ergebnis
        mock_ensemble = np.random.rand(10, 512).astype(np.float32)
        
        # Patch die relevanten Methoden
        with patch.object(EarlyFusion, 'fuse', return_value=mock_results["early"]), \
             patch.object(LateFusion, 'fuse', return_value=mock_results["late"]), \
             patch.object(AttentionFusion, 'fuse', return_value=mock_results["attention"]), \
             patch.object(EarlyFusion, 'get_confidence', return_value=0.3), \
             patch.object(LateFusion, 'get_confidence', return_value=0.5), \
             patch.object(AttentionFusion, 'get_confidence', return_value=0.2), \
             patch('models_app.fusion.tensor_ops.FusionTensorOps.weighted_combine', 
                  return_value=mock_ensemble):
            
            # Führe die Ensemble-Fusion durch
            result, weights = self.hybrid_fusion.fuse_with_ensemble(
                self.visual_features, self.text_features, self.metadata
            )
            
            # Überprüfe die Ergebnisse
            np.testing.assert_array_equal(result, mock_ensemble)
            self.assertEqual(weights["early"], 0.3)
            self.assertEqual(weights["late"], 0.5)
            self.assertEqual(weights["attention"], 0.2)
            
            # Überprüfe, ob Performance-Metriken aktualisiert wurden
            for strategy in ["early", "late", "attention"]:
                self.assertEqual(len(self.hybrid_fusion.performance_metrics[strategy]["time"]), 1)
                self.assertEqual(len(self.hybrid_fusion.performance_metrics[strategy]["memory"]), 1)
    
    def test_get_performance_statistics(self):
        """
        Testet die Generierung von Performance-Statistiken.
        """
        # Füge einige Testdaten hinzu
        for strategy in ["early", "late", "attention"]:
            self.hybrid_fusion.performance_metrics[strategy]["time"] = [0.1, 0.2, 0.3]
            self.hybrid_fusion.performance_metrics[strategy]["memory"] = [10, 20, 30]
            self.hybrid_fusion.performance_metrics[strategy]["quality"] = [0.7, 0.8, 0.9]
        
        # Hole die Statistiken
        stats = self.hybrid_fusion.get_performance_statistics()
        
        # Überprüfe die Struktur und Werte
        for strategy in ["early", "late", "attention"]:
            self.assertIn(strategy, stats)
            self.assertIn("time", stats[strategy])
            self.assertIn("memory", stats[strategy])
            self.assertIn("quality", stats[strategy])
            
            self.assertAlmostEqual(stats[strategy]["time"]["mean"], 0.2)
            self.assertAlmostEqual(stats[strategy]["memory"]["mean"], 20)
            self.assertAlmostEqual(stats[strategy]["quality"]["mean"], 0.8)
            
            self.assertEqual(stats[strategy]["time"]["count"], 3)
            self.assertEqual(stats[strategy]["memory"]["count"], 3)
            self.assertEqual(stats[strategy]["quality"]["count"], 3)
    
    def test_get_performance_by_document_type(self):
        """
        Testet die Generierung von Performance-Statistiken nach Dokumenttyp.
        """
        # Füge einige Testdaten hinzu
        self.hybrid_fusion.performance_history["early"] = [
            {"time": 0.1, "memory": 10, "quality": 0.7, "document_type": "scientific"},
            {"time": 0.2, "memory": 20, "quality": 0.8, "document_type": "business"}
        ]
        self.hybrid_fusion.performance_history["late"] = [
            {"time": 0.3, "memory": 30, "quality": 0.9, "document_type": "scientific"},
            {"time": 0.4, "memory": 40, "quality": 0.6, "document_type": "business"}
        ]
        
        # Hole die Statistiken
        stats = self.hybrid_fusion.get_performance_by_document_type()
        
        # Überprüfe die Struktur und Werte
        self.assertIn("scientific", stats)
        self.assertIn("business", stats)
        
        self.assertIn("early", stats["scientific"])
        self.assertIn("late", stats["scientific"])
        
        self.assertAlmostEqual(stats["scientific"]["early"]["time"]["mean"], 0.1)
        self.assertAlmostEqual(stats["scientific"]["late"]["time"]["mean"], 0.3)
        
        self.assertAlmostEqual(stats["business"]["early"]["quality"]["mean"], 0.8)
        self.assertAlmostEqual(stats["business"]["late"]["quality"]["mean"], 0.6)
    
    def test_update_strategy_weights(self):
        """
        Testet die Aktualisierung der Strategiegewichtungen.
        """
        # Füge einige Testdaten hinzu
        self.hybrid_fusion.performance_metrics["early"]["quality"] = [0.6, 0.7]
        self.hybrid_fusion.performance_metrics["late"]["quality"] = [0.8, 0.9]
        self.hybrid_fusion.performance_metrics["attention"]["quality"] = [0.4, 0.5]
        
        # Aktualisiere die Gewichtungen
        self.hybrid_fusion.update_strategy_weights()
        
        # Überprüfe, ob strategy_weights gesetzt wurde
        self.assertTrue(hasattr(self.hybrid_fusion, "strategy_weights"))
        
        # Überprüfe die Werte
        total = 0.65 + 0.85 + 0.45  # Durchschnittliche Qualitäten
        self.assertAlmostEqual(self.hybrid_fusion.strategy_weights["early"], 0.65 / total)
        self.assertAlmostEqual(self.hybrid_fusion.strategy_weights["late"], 0.85 / total)
        self.assertAlmostEqual(self.hybrid_fusion.strategy_weights["attention"], 0.45 / total)
    
    def test_analyze_document_type_performance(self):
        """
        Testet die Analyse der Performance nach Dokumenttyp.
        """
        # Mock für get_performance_by_document_type
        mock_stats = {
            "scientific": {
                "early": {"quality": {"mean": 0.7, "count": 5}},
                "late": {"quality": {"mean": 0.8, "count": 3}},
                "attention": {"quality": {"mean": 0.9, "count": 2}}
            },
            "business": {
                "early": {"quality": {"mean": 0.6, "count": 4}},
                "late": {"quality": {"mean": 0.9, "count": 6}},
                "attention": {"quality": {"mean": 0.5, "count": 1}}
            }
        }
        
        with patch.object(self.hybrid_fusion, 'get_performance_by_document_type', 
                         return_value=mock_stats):
            
            # Analysiere die Performance
            recommendations = self.hybrid_fusion.analyze_document_type_performance()
            
            # Überprüfe die Empfehlungen
            self.assertIn("scientific", recommendations)
            self.assertIn("business", recommendations)
            
            self.assertEqual(recommendations["scientific"]["recommended_strategy"], "attention")
            self.assertEqual(recommendations["business"]["recommended_strategy"], "late")
            
            self.assertAlmostEqual(recommendations["scientific"]["confidence"], 0.9)
            self.assertAlmostEqual(recommendations["business"]["confidence"], 0.9)
            
            self.assertEqual(recommendations["scientific"]["sample_size"], 2)
            self.assertEqual(recommendations["business"]["sample_size"], 6)

if __name__ == '__main__':
    unittest.main() 