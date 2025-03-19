from typing import Dict, List, Optional
import aiohttp
from pathlib import Path
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
from django.conf import settings
from django.core.cache import cache
from .base import BaseSearchProvider

class LocalDocsSearchProvider(BaseSearchProvider):
    """
    Provider für lokale Dokumentensuche mit dualer Verarbeitungskette:
    1. ColPali für OCR-freie Analyse
    2. Tesseract OCR als Backup/Validation
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.colpali_url = settings.COLPALI_API_URL
        self.tesseract_enabled = settings.ENABLE_TESSERACT_BACKUP
        self.cache_duration = settings.LOCAL_DOCS_CACHE_DURATION
        self.max_file_size = settings.MAX_DOCUMENT_SIZE
        
        # OCR Konfiguration
        self.tesseract_config = {
            'lang': 'eng+deu',  # Sprachen
            'config': '--oem 1 --psm 3',  # OCR Engine Mode & Page Segmentation Mode
        }
        
        # Cache Prefixes
        self.COLPALI_CACHE_PREFIX = "local_doc_colpali:"
        self.OCR_CACHE_PREFIX = "local_doc_ocr:"
        self.RESULT_CACHE_PREFIX = "local_doc_result:"

    async def search(self, query: str, filters: Dict = None) -> List[Dict]:
        """Durchsucht lokale Dokumente"""
        results = []
        search_path = Path(settings.LOCAL_DOCUMENTS_ROOT)
        
        for doc_path in search_path.rglob('*'):
            if not self._is_valid_document(doc_path):
                continue
                
            # Versuche gecachte Ergebnisse zu holen
            cache_key = f"{self.RESULT_CACHE_PREFIX}{doc_path}"
            cached_result = cache.get(cache_key)
            
            if cached_result:
                if self._matches_query(cached_result, query):
                    results.append(cached_result)
                continue
            
            # Parallel Processing mit ColPali und OCR
            colpali_result = await self._process_with_colpali(doc_path)
            ocr_result = await self._process_with_ocr(doc_path) if self.tesseract_enabled else None
            
            # Vergleich und Zusammenführung der Ergebnisse
            final_result = self._merge_results(colpali_result, ocr_result)
            
            # Cache das Ergebnis
            cache.set(cache_key, final_result, self.cache_duration)
            
            if self._matches_query(final_result, query):
                results.append({
                    'title': doc_path.name,
                    'path': str(doc_path),
                    'content': final_result.get('content'),
                    'metadata': final_result.get('metadata'),
                    'confidence': final_result.get('confidence'),
                    'processing_method': final_result.get('method')
                })
        
        return sorted(results, key=lambda x: x.get('confidence', 0), reverse=True)

    async def _process_with_colpali(self, doc_path: Path) -> Optional[Dict]:
        """Verarbeitet Dokument mit ColPali"""
        cache_key = f"{self.COLPALI_CACHE_PREFIX}{doc_path}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result

        try:
            with open(doc_path, 'rb') as f:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.colpali_url}/process",
                        data={'file': f}
                    ) as response:
                        result = await response.json()
                        if result:
                            cache.set(cache_key, result, self.cache_duration)
                        return result
        except Exception as e:
            print(f"ColPali processing failed: {str(e)}")
            return None

    async def _process_with_ocr(self, doc_path: Path) -> Optional[Dict]:
        """Verarbeitet Dokument mit Tesseract OCR"""
        cache_key = f"{self.OCR_CACHE_PREFIX}{doc_path}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result

        try:
            # PDF zu Bildern konvertieren
            if doc_path.suffix.lower() == '.pdf':
                text_blocks = []
                confidence_scores = []
                
                pdf_document = fitz.open(doc_path)
                for page_num in range(pdf_document.page_count):
                    page = pdf_document[page_num]
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x Zoom für bessere Qualität
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # OCR für jede Seite
                    result = pytesseract.image_to_data(
                        img, 
                        lang=self.tesseract_config['lang'],
                        config=self.tesseract_config['config'],
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # Extrahiere Text und Confidence pro Block
                    page_blocks = []
                    current_block = []
                    current_conf = []
                    
                    for i in range(len(result['text'])):
                        if result['block_num'][i] != 0:  # Ignoriere leere Blöcke
                            current_block.append(result['text'][i])
                            current_conf.append(result['conf'][i])
                            
                            if i + 1 == len(result['text']) or result['block_num'][i+1] != result['block_num'][i]:
                                if current_block:
                                    block_text = ' '.join(current_block)
                                    avg_conf = sum(current_conf) / len(current_conf)
                                    text_blocks.append(block_text)
                                    confidence_scores.append(avg_conf)
                                    current_block = []
                                    current_conf = []
                
                pdf_document.close()
                
                # Berechne Gesamtconfidence
                overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
                
                result = {
                    'content': '\n'.join(text_blocks),
                    'confidence': overall_confidence,
                    'method': 'tesseract_ocr',
                    'metadata': {
                        'page_count': len(text_blocks),
                        'confidence_per_page': confidence_scores,
                        'processing_details': self.tesseract_config
                    }
                }
                
                # Cache das Ergebnis
                cache.set(cache_key, result, self.cache_duration)
                return result
                
        except Exception as e:
            print(f"OCR processing failed: {str(e)}")
            return None

    def _merge_results(self, colpali_result: Dict, ocr_result: Dict) -> Dict:
        """Verbesserte Zusammenführung der Ergebnisse von ColPali und OCR"""
        if not colpali_result and not ocr_result:
            return {}
            
        if not ocr_result:
            return colpali_result
            
        if not colpali_result:
            return ocr_result
        
        # Berechne Übereinstimmung zwischen beiden Ergebnissen
        similarity = self._calculate_text_similarity(
            colpali_result.get('content', ''),
            ocr_result.get('content', '')
        )
        
        # Wenn hohe Übereinstimmung, nutze das Ergebnis mit höherer Confidence
        if similarity > 0.8:
            return colpali_result if colpali_result.get('confidence', 0) >= ocr_result.get('confidence', 0) else ocr_result
            
        # Bei geringer Übereinstimmung, kombiniere die Ergebnisse
        return {
            'content': self._combine_texts(
                colpali_result.get('content', ''),
                ocr_result.get('content', '')
            ),
            'confidence': (colpali_result.get('confidence', 0) + ocr_result.get('confidence', 0)) / 2,
            'method': 'hybrid',
            'metadata': {
                'colpali_meta': colpali_result.get('metadata', {}),
                'ocr_meta': ocr_result.get('metadata', {}),
                'similarity_score': similarity
            }
        }

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Berechnet die Ähnlichkeit zwischen zwei Texten"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1, text2).ratio()

    def _combine_texts(self, text1: str, text2: str) -> str:
        """Kombiniert zwei Texte intelligent"""
        # TODO: Implementiere sophistiziertere Kombination
        return f"{text1}\n\n=== Alternative Extraction ===\n\n{text2}"

    def _is_valid_document(self, path: Path) -> bool:
        """Prüft ob Dokument valid und sicher ist"""
        return (
            path.suffix.lower() in ['.pdf', '.doc', '.docx', '.txt'] and
            path.stat().st_size <= self.max_file_size
        )

    def _matches_query(self, result: Dict, query: str) -> bool:
        """Prüft ob ein Dokument zur Suchanfrage passt"""
        content = result.get('content', '').lower()
        return query.lower() in content 