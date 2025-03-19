# Kostenvergleich: Lokale vs. Cloud-KI-Modelle

Diese Übersicht vergleicht die Kosten für den Betrieb von KI-Modellen lokal versus in der Cloud.

## 1. Kostenkomponenten

### 1.1 Lokale Modelle (Einmalige Hardware + Stromkosten)

| Komponente | Kosten |
|------------|--------|
| Hardware-Anschaffung | Einmalig |
| Stromkosten | Laufend |
| Wartung/Updates | Minimal |
| Speicher | Einmalig |

### 1.2 Cloud-Modelle (Pay-per-Use)

| Komponente | Kosten |
|------------|--------|
| API-Anfragen | Pro Token/Anfrage |
| Speicherung | Pro GB/Monat |
| Bandbreite | Pro GB Transfer |
| Zusatzfunktionen | Variabel |

## 2. Hardwarekosten für lokale Modelle

| Modell | Minimale Hardware | Kostenschätzung |
|--------|-------------------|-----------------|
| **DeepSeek-R1/V3** | NVIDIA A100 80GB oder mehrere A6000 48GB, 128GB+ RAM, 300GB+ SSD | 10.000-25.000€ |
| **DeepSeek-R1 (8-bit)** | NVIDIA RTX 4090 24GB, 64GB RAM, 100GB+ SSD | 3.000-4.000€ |
| **DeepSeek-R1 (4-bit)** | NVIDIA RTX 3090 24GB, 32GB RAM, 50GB SSD | 2.000-2.500€ |
| **QwQ-32B (4-bit)** | NVIDIA RTX 4090 24GB, 32GB RAM, 30GB+ SSD | 2.500-3.000€ |
| **QwQ-32B (8-bit)** | NVIDIA RTX 3090 24GB, 32GB RAM, 40GB+ SSD | 2.000-2.500€ |
| **Phi-3-mini/Gemma-2B** | Standard-PC mit 16GB RAM (keine dedizierte GPU erforderlich) | 800-1.200€ |
| **Mistral-7B (4-bit)** | NVIDIA RTX 3060 12GB, 32GB RAM, 20GB+ SSD | 1.200-1.500€ |
| **Llama-3-8B (4-bit)** | NVIDIA RTX 3070 8GB, 32GB RAM, 20GB+ SSD | 1.300-1.600€ |
| **Phi-3-medium (4-bit)** | NVIDIA RTX 3070 8GB, 16GB RAM, 15GB+ SSD | 1.300-1.600€ |
| **Mistral-Large (8-bit)** | NVIDIA RTX 4090 24GB, 64GB RAM, 70GB+ SSD | 2.500-3.500€ |

## 3. Laufende Betriebskosten

### 3.1 Stromkosten für lokale Modelle (pro Monat)

| Modell | Stromverbrauch | Kosten (CH: 0.25 CHF/kWh) |
|--------|----------------|----------------------------|
| Server mit A100 | ~1.200W (24/7) | ~216 CHF/Monat |
| PC mit RTX 4090 | ~500W (12h/Tag) | ~45 CHF/Monat |
| PC mit RTX 3090 | ~450W (12h/Tag) | ~41 CHF/Monat |
| PC mit RTX 3070 | ~300W (12h/Tag) | ~27 CHF/Monat |
| PC mit RTX 3060 | ~250W (12h/Tag) | ~23 CHF/Monat |
| Standard-PC | ~150W (12h/Tag) | ~14 CHF/Monat |

### 3.2 API-Kosten für Cloud-Modelle

| Modell | Input-Kosten | Output-Kosten | 1M Tokens (ca.) |
|--------|--------------|---------------|-----------------|
| GPT-4-Turbo | $10/1M Tokens | $30/1M Tokens | $20 (gemischt) |
| GPT-4o | $5/1M Tokens | $15/1M Tokens | $10 (gemischt) |
| Claude-3 Opus | $15/1M Tokens | $75/1M Tokens | $45 (gemischt) |
| Claude-3 Sonnet | $3/1M Tokens | $15/1M Tokens | $9 (gemischt) |
| Claude-3 Haiku | $0.25/1M Tokens | $1.25/1M Tokens | $0.75 (gemischt) |
| GPT-3.5-Turbo | $0.5/1M Tokens | $1.5/1M Tokens | $1 (gemischt) |
| Mistral Large | $2/1M Tokens | $6/1M Tokens | $4 (gemischt) |
| Mistral Small | $1/1M Tokens | $3/1M Tokens | $2 (gemischt) |
| Mistral Embed | $0.1/1M Tokens | - | $0.1 (nur Input) |
| Cohere Embed | $0.1/1M Tokens | - | $0.1 (nur Input) |

## 4. Break-Even-Analyse

| Modellnutzung | Break-Even-Punkt (lokale vs. Cloud) |
|---------------|-------------------------------------|
| DeepSeek-R1 vs. GPT-4 | ~100-150M Tokens (ca. 3-4 Monate bei 1M Tokens/Tag) |
| QwQ-32B vs. Claude-3 Sonnet | ~250-300M Tokens (ca. 8-10 Monate bei 1M Tokens/Tag) |
| Mistral-7B vs. Mistral Small (Cloud) | ~600-750M Tokens (ca. 20-25 Monate bei 1M Tokens/Tag) |
| Phi-3-mini vs. GPT-3.5 | ~800-1.000M Tokens (ca. 2-3 Jahre bei 1M Tokens/Tag) |

## 5. Infomaniak Cloud-Kosten (monatlich)

| Service | Spezifikation | Kosten (CHF) |
|---------|--------------|--------------|
| Public Cloud mit A100 | 24GB GPU, 32 vCPUs, 128GB RAM | ~800-1.000 CHF |
| Public Cloud mit A10G | 24GB GPU, 24 vCPUs, 112GB RAM | ~600-800 CHF |
| Public Cloud mit T4 | 16GB GPU, 16 vCPUs, 60GB RAM | ~400-600 CHF |
| Public Cloud mit L4 | 24GB GPU, 16 vCPUs, 60GB RAM | ~450-650 CHF |
| Standard VM (ohne GPU) | 8 vCPUs, 32GB RAM | ~100-150 CHF |

## 6. Kosteneinsparungen durch Hybrid-Modell

Ein Hybrid-Ansatz kann erhebliche Einsparungen erzielen:
- Leichte Anfragen lokal mit Phi-3-mini: ~0,01€ pro 1M Tokens
- Komplexe Anfragen mit Cloud-APIs: ~10-20€ pro 1M Tokens
- Optimale Kosteneffizienz bei 80% lokale/20% Cloud-Verteilung

## 7. Empfehlung für durchschnittliche Business-PCs

Für Standard-Businessgeräte ohne dedizierte GPU:

1. **Lightweight Provider mit Phi-3-mini oder Gemma-2B**:
   - ONNX-optimiert für CPU-Ausführung
   - Geringe Hardware-Anforderungen (8-16GB RAM)
   - Stromkosten von nur ~10-15 CHF pro Monat
   - Ausreichende Qualität für einfache bis mittlere Anfragen

2. **Hybrid-Ansatz für optimale Kosten-Leistung**:
   - Lokale Modelle für Standardanfragen (~80% des Volumens)
   - Cloud-APIs für komplexe Aufgaben (~20% des Volumens)
   - Kosteneinsparung von 70-80% gegenüber reiner Cloud-Nutzung

3. **Skalierbare Hardware-Aufrüstung je nach Bedarf**:
   - Beginnen mit leichtgewichtigen Modellen
   - Schrittweise Aufrüstung zu leistungsfähigerer Hardware
   - Kontinuierliche Kosten-Nutzen-Analyse

## 8. Vergleich der verwendeten LLM Provider-Typen

| Provider | Vorteile | Nachteile | Geeignet für |
|----------|----------|-----------|--------------|
| **OpenAI** | Höchste Qualität, regelmäßige Updates, umfassende Dokumentation | Hohe Kosten, Datenschutzbedenken, API-Abhängigkeit | Kritische Anwendungen mit hohen Qualitätsanforderungen |
| **Anthropic** | Hervorragende Qualität, bessere Instruktionsbefolgung, längere Kontexte | Höhere Kosten als OpenAI, weniger Modellvarianten | Komplexe Dokumentenanalyse, Reasoning-Aufgaben |
| **DeepSeek** | Lokal ausführbar, lange Kontexte bis 128K, spezialisiert auf Dokumentenanalyse | Hohe Hardwareanforderungen, komplexe Konfiguration | Dokumentenanalyse, Langtext-Zusammenfassungen |
| **QwQ** | Lokal ausführbar, sehr gute Reasoning-Fähigkeiten, handhabbare Größe | Höhere Hardwareanforderungen als kleinere Modelle | Reasoning, komplexe Analysen, Beratungssysteme |
| **Lightweight** | Minimale Hardwareanforderungen, hohe Geschwindigkeit, ONNX-optimiert | Geringere Fähigkeiten bei komplexen Aufgaben | Einfache Assistenten, erste Antwortebene, Frontend-Integration |
| **Generic Local** | Flexibel konfigurierbar, unterstützt verschiedene Modellgrößen | Weniger spezialisierte Optimierungen | Testumgebungen, Prototyping, verschiedene Experimente |

## 9. Jährliche Kostenrechnung (Beispiel: 10M Tokens/Monat)

| Szenario | Jahr 1 | Jahr 2 | Jahr 3 | Gesamtkosten 3 Jahre |
|----------|--------|--------|--------|----------------------|
| **Nur Cloud (GPT-4o)** | 1.200€ | 1.200€ | 1.200€ | 3.600€ |
| **Nur Cloud (Claude-3 Sonnet)** | 1.080€ | 1.080€ | 1.080€ | 3.240€ |
| **Lightweight-Lösung (Phi-3)** | 1.000€* + 120€ | 120€ | 120€ | 1.240€ |
| **QwQ-32B-Lösung** | 2.500€* + 240€ | 240€ | 240€ | 3.220€ |
| **Hybrid-Lösung (80/20)** | 1.000€* + 240€ | 240€ | 240€ | 1.720€ |

*Einmalige Hardware-Anschaffungskosten
