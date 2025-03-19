"""
Tensor-Operationen für Fusion

Implementiert mathematische Operationen für die multimodale Fusion
von visuellen (ColPali) und textuellen (OCR) Features.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union

logger = logging.getLogger(__name__)

class FusionTensorOps:
    """
    Klasse für Tensor-Operationen bei der Feature-Fusion.
    """
    
    def __init__(self, device=None):
        """
        Initialisiert die Tensor-Operationen für die Fusion.
        
        Args:
            device: Torch-Device (CPU/GPU)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def early_fusion(self, 
                    visual_features: Union[torch.Tensor, np.ndarray], 
                    text_features: Union[torch.Tensor, np.ndarray],
                    projection_dim: int = 512) -> torch.Tensor:
        """
        Führt eine Early-Fusion der Features durch.
        
        Bei der Early-Fusion werden die Features konkateniert und durch
        eine lineare Projektion auf einen gemeinsamen Raum abgebildet.
        
        Args:
            visual_features: Visuelle Features
            text_features: Textuelle Features
            projection_dim: Dimension des Ausgabevektors
            
        Returns:
            torch.Tensor: Fusionierte Features
        """
        # Konvertiere zu Torch-Tensoren falls nötig
        v_features = self._to_tensor(visual_features)
        t_features = self._to_tensor(text_features)
        
        # Stell sicher, dass die Tensoren 2D sind (batch_size, feature_dim)
        v_features = self._ensure_2d(v_features)
        t_features = self._ensure_2d(t_features)
        
        # Konkateniere die Features
        concatenated = torch.cat([v_features, t_features], dim=-1)
        
        # Erstelle eine lineare Projektion
        input_dim = concatenated.shape[-1]
        projection = nn.Linear(input_dim, projection_dim).to(self.device)
        
        # Wende die Projektion an
        fused_features = projection(concatenated)
        
        # Normalisiere die Features
        fused_features = F.normalize(fused_features, p=2, dim=-1)
        
        return fused_features
    
    def late_fusion(self,
                  visual_features: Union[torch.Tensor, np.ndarray],
                  text_features: Union[torch.Tensor, np.ndarray],
                  visual_weight: float = 0.5) -> torch.Tensor:
        """
        Führt eine Late-Fusion der Features durch.
        
        Bei der Late-Fusion werden die Features separat verarbeitet und
        die Ergebnisse gewichtet kombiniert.
        
        Args:
            visual_features: Visuelle Features
            text_features: Textuelle Features
            visual_weight: Gewichtung der visuellen Features
            
        Returns:
            torch.Tensor: Fusionierte Features
        """
        # Konvertiere zu Torch-Tensoren falls nötig
        v_features = self._to_tensor(visual_features)
        t_features = self._to_tensor(text_features)
        
        # Stell sicher, dass die Tensoren 2D sind (batch_size, feature_dim)
        v_features = self._ensure_2d(v_features)
        t_features = self._ensure_2d(t_features)
        
        # Normalisiere beide Feature-Vektoren
        v_features = F.normalize(v_features, p=2, dim=-1)
        t_features = F.normalize(t_features, p=2, dim=-1)
        
        # Berechne die gewichtete Summe
        text_weight = 1.0 - visual_weight
        fused_features = visual_weight * v_features + text_weight * t_features
        
        # Normalisiere die fusionierten Features
        fused_features = F.normalize(fused_features, p=2, dim=-1)
        
        return fused_features
    
    def attention_fusion(self,
                       visual_features: Union[torch.Tensor, np.ndarray],
                       text_features: Union[torch.Tensor, np.ndarray],
                       num_heads: int = 4,
                       hidden_size: int = 512) -> torch.Tensor:
        """
        Führt eine Attention-basierte Fusion der Features durch.
        
        Bei der Attention-Fusion werden Kreuz-Attention-Mechanismen verwendet,
        um relevante Teile der verschiedenen Modalitäten zu fokussieren.
        
        Args:
            visual_features: Visuelle Features
            text_features: Textuelle Features
            num_heads: Anzahl der Attention Heads
            hidden_size: Dimension des Hidden States
            
        Returns:
            torch.Tensor: Fusionierte Features
        """
        # Konvertiere zu Torch-Tensoren falls nötig
        v_features = self._to_tensor(visual_features)
        t_features = self._to_tensor(text_features)
        
        # Ermittle Feature-Dimensionen
        v_dim = v_features.shape[-1]
        t_dim = t_features.shape[-1]
        
        # Erstelle Multi-Head Attention
        mha = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        ).to(self.device)
        
        # Projektionen für visuelle und textuelle Features
        v_projection = nn.Linear(v_dim, hidden_size).to(self.device)
        t_projection = nn.Linear(t_dim, hidden_size).to(self.device)
        
        # Projiziere Features in den gemeinsamen Raum
        v_proj = v_projection(v_features)
        t_proj = t_projection(t_features)
        
        # Füge Sequenzdimension hinzu, falls nötig
        if len(v_proj.shape) == 2:
            v_proj = v_proj.unsqueeze(1)  # [batch_size, 1, hidden_size]
        if len(t_proj.shape) == 2:
            t_proj = t_proj.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # Kreuz-Attention: Text als Query, Visual als Key/Value
        attn_output, _ = mha(t_proj, v_proj, v_proj)
        
        # Kombiniere die Attention-Ausgabe mit den Text-Features
        fused_features = attn_output + t_proj
        
        # Komprimiere die Sequenzdimension durch Mittelwertbildung
        fused_features = fused_features.mean(dim=1)
        
        # Normalisiere die Features
        fused_features = F.normalize(fused_features, p=2, dim=-1)
        
        return fused_features
    
    def compute_fusion_confidence(self,
                              visual_features: Union[torch.Tensor, np.ndarray],
                              text_features: Union[torch.Tensor, np.ndarray],
                              fused_features: Union[torch.Tensor, np.ndarray]) -> float:
        """
        Berechnet einen Konfidenzwert für die Fusion.
        
        Die Konfidenz basiert auf der Kosinus-Ähnlichkeit zwischen
        den Eingabe-Features und dem fusionierten Ergebnis.
        
        Args:
            visual_features: Visuelle Features
            text_features: Textuelle Features
            fused_features: Fusionierte Features
            
        Returns:
            float: Konfidenzwert zwischen 0 und 1
        """
        # Konvertiere zu Torch-Tensoren falls nötig
        v_features = self._to_tensor(visual_features)
        t_features = self._to_tensor(text_features)
        f_features = self._to_tensor(fused_features)
        
        # Stell sicher, dass die Tensoren 2D sind (batch_size, feature_dim)
        v_features = self._ensure_2d(v_features)
        t_features = self._ensure_2d(t_features)
        f_features = self._ensure_2d(f_features)
        
        # Normalisiere die Features
        v_features = F.normalize(v_features, p=2, dim=-1)
        t_features = F.normalize(t_features, p=2, dim=-1)
        f_features = F.normalize(f_features, p=2, dim=-1)
        
        # Berechne Kosinus-Ähnlichkeit zwischen fusionierten Features und Eingabe-Features
        v_sim = F.cosine_similarity(f_features, v_features, dim=-1)
        t_sim = F.cosine_similarity(f_features, t_features, dim=-1)
        
        # Mittelwert der Ähnlichkeiten als Konfidenz
        confidence = (v_sim.mean() + t_sim.mean()) / 2.0
        
        return float(confidence)
    
    def _to_tensor(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Konvertiert Daten zu einem Torch-Tensor.
        
        Args:
            data: Eingabedaten als Torch-Tensor oder NumPy-Array
            
        Returns:
            torch.Tensor: Daten als Torch-Tensor
        """
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self.device)
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            raise TypeError(f"Nicht unterstützter Datentyp: {type(data)}")
    
    def _ensure_2d(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Stellt sicher, dass der Tensor 2D ist (batch_size, feature_dim).
        
        Args:
            tensor: Eingabe-Tensor
            
        Returns:
            torch.Tensor: 2D-Tensor
        """
        if len(tensor.shape) == 1:
            # Einzelner Vektor, füge Batch-Dimension hinzu
            return tensor.unsqueeze(0)
        elif len(tensor.shape) > 2:
            # Tensor mit mehr als 2 Dimensionen, komprimiere zu 2D
            batch_size = tensor.shape[0]
            return tensor.view(batch_size, -1)
        else:
            # Bereits 2D
            return tensor 