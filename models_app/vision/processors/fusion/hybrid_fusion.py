"""
Hybride Fusion Implementierung

Kombiniert verschiedene Fusionsstrategien und wählt die beste
basierend auf Dokumententyp und Eigenschaften aus.
"""

import logging
import time
import psutil
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import torch
import os

from .base import FusionStrategy, EarlyFusion, LateFusion, AttentionFusion
from .tensor_ops import FusionTensorOps
from analytics_app.utils import monitor_fusion_performance
from error_handlers.models_app_errors.vision_errors.vision_errors import handle_fusion_errors, FusionError, StrategyError, DataQualityError, InconsistentResultError, ProcessingTimeoutError, ResourceExhaustedError

logger = logging.getLogger(__name__)

class HybridFusion:
    """
    Hybride Fusion: Kombiniert verschiedene Fusionsstrategien und wählt die beste aus.
    """
    
    def __init__(self):
        """
        Initialisiert die Hybride Fusion mit verschiedenen Fusionsstrategien.
        """
        # Erstelle Instanzen aller Fusionsstrategien
        self.fusion_strategies = {
            "early": EarlyFusion(),
            "late": LateFusion(),
            "attention": AttentionFusion()
        }
        
        # Leistungs-Tracking
        self.performance_history = {
            "early": [],
            "late": [],
            "attention": []
        }
        
        # Performance-Metriken
        self.performance_metrics = {
            "early": {"time": [], "memory": [], "quality": []},
            "late": {"time": [], "memory": [], "quality": []},
            "attention": {"time": [], "memory": [], "quality": []}
        }
        
        # Initialize tensor operations
        self.tensor_ops = FusionTensorOps()
    
    def cleanup(self) -> None:
        """Clean up fusion resources."""
        try:
            # Clean up strategies
            for strategy in self.fusion_strategies.values():
                if hasattr(strategy, 'cleanup'):
                    strategy.cleanup()
            
            # Clean up tensor operations
            if hasattr(self.tensor_ops, 'cleanup'):
                self.tensor_ops.cleanup()
            
            # Clear performance history and metrics
            self.performance_history.clear()
            self.performance_metrics.clear()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("HybridFusion cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during HybridFusion cleanup: {str(e)}")
            
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Error during HybridFusion destruction: {str(e)}")
    
    def predict_best_strategy(self, visual_features: Any, text_features: Any, 
                             document_metadata: Optional[Dict] = None) -> Tuple[str, float]:
        """
        Sagt die beste Fusionsstrategie für die gegebenen Features voraus.
        
        Args:
            visual_features: Features aus dem visuellen Modell
            text_features: Features aus dem OCR-Modell
            document_metadata: Metadaten zum Dokument (optional)
            
        Returns:
            Tuple[str, float]: Die beste Strategie und ihre Konfidenz
        """
        # Berechne Konfidenzen für alle Strategien
        confidences = {}
        for name, strategy in self.fusion_strategies.items():
            confidences[name] = strategy.get_confidence(visual_features, text_features)
        
        # Berücksichtige Dokumententyp, falls vorhanden
        if document_metadata and "document_type" in document_metadata:
            doc_type = document_metadata["document_type"].lower()
            
            # Anpassungen basierend auf Dokumententyp
            if doc_type == "academic" or doc_type == "scientific":
                # Akademische Dokumente profitieren von der Attention-basierten Fusion
                confidences["attention"] *= 1.2
            elif doc_type == "business" or doc_type == "form":
                # Geschäftsdokumente profitieren von der Late Fusion
                confidences["late"] *= 1.2
            elif doc_type == "general" or doc_type == "article":
                # Allgemeine Dokumente profitieren von der Early Fusion
                confidences["early"] *= 1.2
        
        # Berücksichtige historische Performance-Metriken, falls vorhanden
        for strategy_name in self.fusion_strategies.keys():
            if (len(self.performance_metrics[strategy_name]["time"]) > 0 and
                len(self.performance_metrics[strategy_name]["quality"]) > 0):
                
                # Berechne durchschnittliche Qualität
                avg_quality = np.mean(self.performance_metrics[strategy_name]["quality"])
                
                # Passe Konfidenz basierend auf historischer Qualität an
                quality_factor = min(max(avg_quality, 0.5), 1.5)  # Begrenze zwischen 0.5 und 1.5
                confidences[strategy_name] *= quality_factor
        
        # Wähle die Strategie mit der höchsten Konfidenz
        best_strategy = max(confidences, key=confidences.get)
        best_confidence = confidences[best_strategy]
        
        logger.info(f"Beste Fusionsstrategie: {best_strategy} (Konfidenz: {best_confidence:.2f})")
        return best_strategy, best_confidence
    
    @monitor_fusion_performance
    @handle_fusion_errors
    def fuse_with_best_strategy(self, visual_features: Any, text_features: Any, 
                              document_metadata: Optional[Dict] = None) -> Tuple[Any, str, float]:
        """
        Führt die Fusion mit der besten Strategie durch.
        
        Args:
            visual_features: Features aus dem visuellen Modell
            text_features: Features aus dem OCR-Modell
            document_metadata: Metadaten zum Dokument (optional)
            
        Returns:
            Tuple[Any, str, float]: Die fusionierten Features, die verwendete Strategie und die Konfidenz
        """
        # Vorhersage der besten Strategie
        best_strategy, confidence = self.predict_best_strategy(
            visual_features, text_features, document_metadata
        )
        
        # Durchführung der Fusion mit der besten Strategie und Performance-Tracking
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        fused_features = self.fusion_strategies[best_strategy].fuse(
            visual_features, text_features, document_metadata
        )
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        # Berechne Performance-Metriken
        processing_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        # Speichere Metriken
        self.performance_metrics[best_strategy]["time"].append(processing_time)
        self.performance_metrics[best_strategy]["memory"].append(memory_usage)
        
        # Berechne Qualitätsmetrik (hier als Beispiel eine einfache Metrik)
        # In der Praxis würde hier eine komplexere Berechnung stattfinden
        quality_metric = self._calculate_quality_metric(fused_features, visual_features, text_features)
        self.performance_metrics[best_strategy]["quality"].append(quality_metric)
        
        # Aktualisiere Performance-History
        self.performance_history[best_strategy].append({
            "time": processing_time,
            "memory": memory_usage,
            "quality": quality_metric,
            "document_type": document_metadata.get("document_type", "unknown") if document_metadata else "unknown"
        })
        
        logger.info(f"Fusion mit {best_strategy} abgeschlossen: Zeit={processing_time:.4f}s, Speicher={memory_usage:.2f}MB, Qualität={quality_metric:.4f}")
        
        return fused_features, best_strategy, confidence
    
    def _calculate_quality_metric(self, fused_features: Any, visual_features: Any, text_features: Any) -> float:
        """
        Berechnet eine Qualitätsmetrik für die fusionierten Features.
        
        Args:
            fused_features: Die fusionierten Features
            visual_features: Die visuellen Features
            text_features: Die Text-Features
            
        Returns:
            float: Eine Qualitätsmetrik zwischen 0 und 1
        """
        try:
            # Extrahiere die relevanten Features
            if hasattr(fused_features, "get") and callable(getattr(fused_features, "get")):
                f_features = fused_features.get("features", fused_features)
            else:
                f_features = fused_features
                
            # Berechne eine einfache Qualitätsmetrik basierend auf der Varianz der Features
            # In der Praxis würde hier eine komplexere Berechnung stattfinden
            if hasattr(f_features, "var") and callable(getattr(f_features, "var")):
                variance = float(f_features.var())
                # Normalisiere die Varianz auf einen Wert zwischen 0 und 1
                quality = min(max(variance / 10.0, 0), 1)
            else:
                # Fallback, wenn keine Varianz berechnet werden kann
                quality = 0.8
                
            return quality
        except Exception as e:
            logger.warning(f"Fehler bei der Berechnung der Qualitätsmetrik: {str(e)}")
            return 0.7  # Fallback-Wert
    
    @monitor_fusion_performance
    @handle_fusion_errors
    def fuse_with_ensemble(self, visual_features: Any, text_features: Any, 
                          document_metadata: Optional[Dict] = None) -> Tuple[Any, Dict[str, float]]:
        """
        Führt die Fusion mit einem Ensemble aller Strategien durch.
        
        Args:
            visual_features: Features aus dem visuellen Modell
            text_features: Features aus dem OCR-Modell
            document_metadata: Metadaten zum Dokument (optional)
            
        Returns:
            Tuple[Any, Dict[str, float]]: Die fusionierten Features und die Gewichte der Strategien
        """
        # Führe alle Fusionsstrategien aus und tracke die Performance
        results = {}
        weights = {}
        performance_data = {}
        
        for name, strategy in self.fusion_strategies.items():
            # Messe Performance
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            
            # Führe Fusion durch
            result = strategy.fuse(visual_features, text_features, document_metadata)
            
            # Berechne Performance-Metriken
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            
            processing_time = end_time - start_time
            memory_usage = end_memory - start_memory
            
            # Speichere Ergebnis und Performance-Daten
            results[name] = result
            performance_data[name] = {
                "time": processing_time,
                "memory": memory_usage
            }
            
            # Berechne Gewicht basierend auf Konfidenz
            weights[name] = strategy.get_confidence(visual_features, text_features)
            
            # Speichere Metriken
            self.performance_metrics[name]["time"].append(processing_time)
            self.performance_metrics[name]["memory"].append(memory_usage)
        
        # Normalisiere Gewichte
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Kombiniere die Ergebnisse basierend auf den Gewichten
        # Hier müsste eine spezifische Implementierung für die Kombination der Features erfolgen
        # Als Beispiel verwenden wir eine gewichtete Summe, falls die Features NumPy-Arrays sind
        try:
            tensor_ops = FusionTensorOps()
            ensemble_result = tensor_ops.weighted_combine(results, weights)
        except Exception as e:
            logger.warning(f"Fehler bei der Ensemble-Fusion: {str(e)}")
            # Fallback: Verwende das Ergebnis der Strategie mit dem höchsten Gewicht
            best_strategy = max(weights, key=weights.get)
            ensemble_result = results[best_strategy]
        
        logger.info(f"Ensemble-Fusion abgeschlossen mit Gewichten: {weights}")
        
        return ensemble_result, weights
    
    def get_performance_statistics(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Gibt Statistiken über die Performance der Fusionsstrategien zurück.
        
        Returns:
            Dict: Performance-Statistiken für jede Strategie
        """
        stats = {}
        
        for strategy_name, metrics in self.performance_metrics.items():
            stats[strategy_name] = {
                "time": {
                    "mean": np.mean(metrics["time"]) if metrics["time"] else 0,
                    "min": np.min(metrics["time"]) if metrics["time"] else 0,
                    "max": np.max(metrics["time"]) if metrics["time"] else 0,
                    "std": np.std(metrics["time"]) if metrics["time"] else 0,
                    "count": len(metrics["time"])
                },
                "memory": {
                    "mean": np.mean(metrics["memory"]) if metrics["memory"] else 0,
                    "min": np.min(metrics["memory"]) if metrics["memory"] else 0,
                    "max": np.max(metrics["memory"]) if metrics["memory"] else 0,
                    "std": np.std(metrics["memory"]) if metrics["memory"] else 0,
                    "count": len(metrics["memory"])
                },
                "quality": {
                    "mean": np.mean(metrics["quality"]) if metrics["quality"] else 0,
                    "min": np.min(metrics["quality"]) if metrics["quality"] else 0,
                    "max": np.max(metrics["quality"]) if metrics["quality"] else 0,
                    "std": np.std(metrics["quality"]) if metrics["quality"] else 0,
                    "count": len(metrics["quality"])
                }
            }
        
        return stats
    
    def get_performance_by_document_type(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Gibt Performance-Statistiken gruppiert nach Dokumententyp zurück.
        
        Returns:
            Dict: Performance-Statistiken für jede Strategie, gruppiert nach Dokumententyp
        """
        doc_type_stats = {}
        
        for strategy_name, history in self.performance_history.items():
            for entry in history:
                doc_type = entry.get("document_type", "unknown")
                
                if doc_type not in doc_type_stats:
                    doc_type_stats[doc_type] = {}
                
                if strategy_name not in doc_type_stats[doc_type]:
                    doc_type_stats[doc_type][strategy_name] = {
                        "time": [], "memory": [], "quality": []
                    }
                
                doc_type_stats[doc_type][strategy_name]["time"].append(entry["time"])
                doc_type_stats[doc_type][strategy_name]["memory"].append(entry["memory"])
                doc_type_stats[doc_type][strategy_name]["quality"].append(entry["quality"])
        
        # Berechne Statistiken für jeden Dokumententyp und jede Strategie
        result = {}
        for doc_type, strategies in doc_type_stats.items():
            result[doc_type] = {}
            
            for strategy_name, metrics in strategies.items():
                result[doc_type][strategy_name] = {
                    "time": {
                        "mean": np.mean(metrics["time"]) if metrics["time"] else 0,
                        "count": len(metrics["time"])
                    },
                    "memory": {
                        "mean": np.mean(metrics["memory"]) if metrics["memory"] else 0,
                        "count": len(metrics["memory"])
                    },
                    "quality": {
                        "mean": np.mean(metrics["quality"]) if metrics["quality"] else 0,
                        "count": len(metrics["quality"])
                    }
                }
        
        return result
    
    def fuse_with_strategy(self, strategy: FusionStrategy, visual_features: Any, text_features: Any, 
                         document_metadata: Optional[Dict] = None) -> Any:
        """
        Führt die Fusion mit einer bestimmten Strategie durch.
        
        Args:
            strategy: Die zu verwendende Fusionsstrategie
            visual_features: Features aus dem visuellen Modell
            text_features: Features aus dem OCR-Modell
            document_metadata: Metadaten zum Dokument (optional)
            
        Returns:
            Die fusionierten Features
        """
        return strategy.fuse(visual_features, text_features, document_metadata)

    def update_strategy_weights(self):
        """
        Aktualisiert die Gewichtungen der Strategien basierend auf historischer Performance.
        Dies ermöglicht adaptives Lernen über Zeit.
        """
        if not any(self.performance_metrics[strategy]["quality"] for strategy in self.fusion_strategies.keys()):
            logger.info("Nicht genügend Daten für Gewichtungsaktualisierung")
            return
        
        # Berechne durchschnittliche Qualität für jede Strategie
        avg_quality = {}
        for strategy in self.fusion_strategies.keys():
            if self.performance_metrics[strategy]["quality"]:
                avg_quality[strategy] = np.mean(self.performance_metrics[strategy]["quality"])
            else:
                avg_quality[strategy] = 0.5  # Standardwert
        
        # Normalisiere zu Gewichtungen
        total_quality = sum(avg_quality.values())
        if total_quality > 0:
            self.strategy_weights = {k: v / total_quality for k, v in avg_quality.items()}
            logger.info(f"Strategie-Gewichtungen aktualisiert: {self.strategy_weights}")
    
    def analyze_document_type_performance(self):
        """
        Analysiert, welche Fusionsstrategie für welchen Dokumenttyp am besten funktioniert.
        Gibt eine Empfehlung für zukünftige Dokumente.
        """
        doc_type_stats = self.get_performance_by_document_type()
        recommendations = {}
        
        for doc_type, strategies in doc_type_stats.items():
            if not strategies:
                continue
            
            # Finde die beste Strategie für diesen Dokumenttyp
            best_strategy = None
            best_quality = -1
            
            for strategy, metrics in strategies.items():
                avg_quality = metrics.get("quality", {}).get("mean", 0)
                if avg_quality > best_quality:
                    best_quality = avg_quality
                    best_strategy = strategy
            
            if best_strategy:
                recommendations[doc_type] = {
                    "recommended_strategy": best_strategy,
                    "confidence": best_quality,
                    "sample_size": strategies[best_strategy]["quality"]["count"]
                }
        
        return recommendations

    def evaluate_confidence_accuracy(self):
        """
        Evaluiert, wie gut die vorhergesagte Konfidenz mit der tatsächlichen Qualität korreliert.
        Ein wichtiger Indikator für die Zuverlässigkeit des Auswahlmechanismus.
        """
        correlation_data = {
            "early": {"predicted": [], "actual": []},
            "late": {"predicted": [], "actual": []},
            "attention": {"predicted": [], "actual": []}
        }
        
        # Sammle Daten aus der Performance-History
        for strategy, history in self.performance_history.items():
            for entry in history:
                # Die vorhergesagte Konfidenz müsste hier gespeichert werden
                # Dies ist ein Vorschlag für eine Erweiterung
                if "predicted_confidence" in entry and "quality" in entry:
                    correlation_data[strategy]["predicted"].append(entry["predicted_confidence"])
                    correlation_data[strategy]["actual"].append(entry["quality"])
        
        # Berechne Korrelation für jede Strategie
        results = {}
        for strategy, data in correlation_data.items():
            if len(data["predicted"]) > 5:  # Mindestens 5 Datenpunkte für sinnvolle Korrelation
                try:
                    from scipy.stats import pearsonr
                    correlation, p_value = pearsonr(data["predicted"], data["actual"])
                    results[strategy] = {
                        "correlation": correlation,
                        "p_value": p_value,
                        "sample_size": len(data["predicted"])
                    }
                except:
                    # Fallback ohne scipy
                    results[strategy] = {
                        "correlation": "scipy nicht verfügbar",
                        "sample_size": len(data["predicted"])
                    }
        
        return results

    @monitor_fusion_performance
    @handle_fusion_errors
    def fuse_features(self, ocr_features: Dict[str, Any], 
                      vision_features: Dict[str, Any], 
                      document_type: str = "generic") -> Dict[str, Any]:
        """
        Führt die Fusion von OCR- und Vision-Features durch.
        
        Args:
            ocr_features: Features aus dem OCR-Prozess
            vision_features: Features aus der visuellen Verarbeitung
            document_type: Typ des Dokuments für strategiebasierte Fusion
            
        Returns:
            Dictionary mit fusionierten Features und Metadaten
        
        Raises:
            FusionError: Bei allgemeinen Fusionsfehlern
            StrategyError: Bei fehlgeschlagener Strategieauswahl
            InconsistentResultError: Bei inkonsistenten Ergebnissen
            ProcessingTimeoutError: Bei Überschreitung des Zeitlimits
            ResourceExhaustedError: Bei Überschreitung der Ressourcenlimits
        """
        start_time = time.time()
        
        try:
            # Ressourcennutzung überwachen
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
            
            # Ein Timeout für die Operation setzen
            max_processing_time = 120  # 2 Minuten
            
            # Überprüfen, ob die Eingabe-Features gültig sind
            if not ocr_features or not vision_features:
                logger.error("Leere Features für die Fusion übergeben")
                raise DataQualityError("Empty features provided for fusion", 
                                      error_code="empty_fusion_features")
            
            # Bestimme die beste Fusionstrategie basierend auf dem Dokumenttyp und den Features
            strategy_name = self._predict_best_strategy(document_type, ocr_features, vision_features)
            logger.info(f"Verwende Fusionstrategie: {strategy_name} für Dokumenttyp: {document_type}")
            
            # Strategie abrufen
            if strategy_name not in self.fusion_strategies:
                raise StrategyError(f"Unknown fusion strategy: {strategy_name}", 
                                  error_code="unknown_strategy")
                
            strategy = self.fusion_strategies[strategy_name]
            
            # Ressourcennutzung prüfen
            current_memory = process.memory_info().rss
            memory_used = (current_memory - initial_memory) / (1024*1024)  # MB
            if memory_used > 500:  # über 500MB Speichernutzung
                raise ResourceExhaustedError("Memory usage exceeded during strategy selection", 
                                           error_code="memory_limit_exceeded")
            
            # Fusion durchführen mit der gewählten Strategie
            fused_data = strategy.fuse(ocr_features, vision_features)
            
            # Überprüfe auf Zeitüberschreitung
            elapsed_time = time.time() - start_time
            if elapsed_time > max_processing_time:
                raise ProcessingTimeoutError(f"Fusion processing exceeded time limit of {max_processing_time}s", 
                                           error_code="fusion_timeout")
            
            # Konsistenzprüfung der Ergebnisse
            if not self._validate_fusion_result(fused_data):
                raise InconsistentResultError("Fusion produced inconsistent or incomplete results", 
                                            error_code="invalid_fusion_result")
            
            # Metadaten hinzufügen
            result = {
                "fused_data": fused_data,
                "strategy_used": strategy_name,
                "document_type": document_type,
                "fusion_time": time.time() - start_time,
                "ocr_contribution": self._calculate_contribution(fused_data, "ocr"),
                "vision_contribution": self._calculate_contribution(fused_data, "vision")
            }
            
            return result
            
        except (StrategyError, InconsistentResultError, ProcessingTimeoutError, 
                ResourceExhaustedError, DataQualityError) as e:
            # Diese Exceptions werden weitergereicht und vom Decorator behandelt
            raise
        except Exception as e:
            # Allgemeine Fehler in FusionError umwandeln
            raise FusionError(f"Failed to fuse features: {str(e)}", 
                            error_code="fusion_failed")
    
    def _validate_fusion_result(self, fused_data: Dict[str, Any]) -> bool:
        """
        Überprüft die Konsistenz und Vollständigkeit der Fusionsergebnisse.
        
        Args:
            fused_data: Die fusionierten Daten
            
        Returns:
            bool: True wenn die Daten konsistent sind, sonst False
        """
        # Mindestens diese Schlüssel sollten in den fusionierten Daten vorhanden sein
        required_keys = ["text", "embeddings", "layout_info"]
        
        # Überprüfe, ob alle erforderlichen Schlüssel vorhanden sind
        if not all(key in fused_data for key in required_keys):
            logger.error(f"Fusion result is missing required keys. Found: {list(fused_data.keys())}")
            return False
            
        # Überprüfe auf Nullwerte oder leere Arrays in wichtigen Feldern
        if not fused_data["text"] or len(fused_data["text"]) == 0:
            logger.error("Fusion result has empty text field")
            return False
            
        # Überprüfe, ob die Embeddings vorhanden und korrekt dimensioniert sind
        if "embeddings" in fused_data:
            embeddings = fused_data["embeddings"]
            if embeddings is None or (isinstance(embeddings, np.ndarray) and embeddings.size == 0):
                logger.error("Fusion result has empty embeddings")
                return False
        
        # Weitere spezifische Validierungen können hier hinzugefügt werden
        
        return True
        
    def _calculate_contribution(self, fused_data: Dict[str, Any], source: str) -> float:
        """
        Berechnet den Beitrag einer bestimmten Quelle (OCR oder Vision) zum Fusionsergebnis.
        
        Args:
            fused_data: Die fusionierten Daten
            source: Die Quelle ('ocr' oder 'vision')
            
        Returns:
            float: Prozentualer Beitrag der Quelle (0.0 - 1.0)
        """
        # Diese Funktion würde eine komplexere Logik enthalten, um den Beitrag zu berechnen
        # Hier eine vereinfachte Implementierung:
        if "source_contributions" in fused_data:
            contributions = fused_data["source_contributions"]
            if source in contributions:
                return contributions[source]
                
        # Fallback: Schätze basierend auf den Metadaten
        if source == "ocr" and "text_from_ocr" in fused_data:
            return 0.7  # OCR trägt in diesem Fall mehr bei
        elif source == "vision" and "layout_from_vision" in fused_data:
            return 0.6  # Vision trägt in diesem Fall mehr bei
            
        # Standard-Fallback
        return 0.5  # Gleicher Beitrag 