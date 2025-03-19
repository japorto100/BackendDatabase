"""Zentrale Implementierung von Dummy-Modellen für Tests und Offline-Betrieb."""
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image

class DummyTensorOutput:
    def __init__(self, shape=(1, 512)):
        import numpy as np
        self.logits = np.random.rand(*shape)
        self.last_hidden_state = np.random.rand(1, 32, 512)

class DummyProcessor:
    def __init__(self, task="ocr"):
        self.task = task
        self.tokenizer = self
        self.feature_extractor = self
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        
    def __call__(self, images, return_tensors="pt", **kwargs):
        return {"pixel_values": torch.randn(1, 3, 224, 224), "attention_mask": torch.ones(1, 196)}
    
    def batch_decode(self, ids, skip_special_tokens=True):
        return ["Dummy text output for OCR result"]
    
    def decode(self, ids, skip_special_tokens=True):
        return "Dummy text output for OCR result"
        
    def tokenizer(self, text, **kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]]), "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])}

class DummyModel:
    def __init__(self, model_type="ocr"):
        self.model_type = model_type
    
    def to(self, device):
        return self
        
    def eval(self):
        return self
        
    def __call__(self, **kwargs):
        return DummyTensorOutput()
        
    def generate(self, **kwargs):
        class DummySequences:
            def __init__(self):
                self.sequences = torch.tensor([[101, 2054, 2003, 1037, 4937, 102]])
        return DummySequences()

class DummyModelFactory:
    """Factory-Klasse für die Erstellung von Dummy-Modellen für verschiedene Frameworks."""
    
    @staticmethod
    def create_ocr_dummy(model_type: str = "generic") -> Any:
        """Erzeugt ein OCR-Dummy-Modell des angegebenen Typs."""
        if model_type == "easyocr":
            return DummyModelFactory._create_easyocr_dummy()
        elif model_type == "paddleocr":
            return DummyModelFactory._create_paddleocr_dummy()
        elif model_type == "doctr":
            return DummyModelFactory._create_doctr_dummy()
        elif model_type == "layoutlmv3":
            return DummyModelFactory._create_layoutlmv3_dummy()
        elif model_type == "donut":
            return DummyModelFactory._create_donut_dummy()
        elif model_type == "microsoft":
            return DummyModelFactory._create_microsoft_dummy()
        elif model_type == "formula":
            return DummyModelFactory._create_formula_dummy()
        elif model_type == "tesseract":
            return DummyModelFactory._create_tesseract_dummy()
        elif model_type == "table_extraction":
            return DummyModelFactory._create_table_extraction_dummy()
        elif model_type == "nougat":
            return DummyModelFactory._create_nougat_dummy()
        else:
            return DummyModelFactory._create_generic_ocr_dummy()
    
    @staticmethod
    def _create_easyocr_dummy() -> Any:
        """Erzeugt ein Dummy-Modell für EasyOCR."""
        class DummyReader:
            def readtext(self, img_path, detail=1, paragraph=False, **kwargs):
                # Ein Dummy-Ergebnis zurückgeben
                return [
                    ([[10, 10], [100, 10], [100, 40], [10, 40]], "Dummy Text 1", 0.95),
                    ([[10, 50], [150, 50], [150, 80], [10, 80]], "Dummy Text 2", 0.92)
                ]
        return DummyReader()
    
    @staticmethod
    def _create_paddleocr_dummy() -> Any:
        """Erzeugt ein Dummy-Modell für PaddleOCR."""
        class DummyPaddleOCR:
            def __call__(self, img_path, cls=True):
                # Gibt ein standardisiertes Dummy-Ergebnis zurück
                return [
                    [
                        [[10, 10], [100, 10], [100, 40], [10, 40]], 
                        ('Dummy Text PaddleOCR', 0.95)
                    ],
                    [
                        [[10, 50], [200, 50], [200, 80], [10, 80]], 
                        ('Another Dummy Text', 0.88)
                    ]
                ]
        return DummyPaddleOCR()
    
    @staticmethod
    def _create_doctr_dummy() -> Any:
        """Erzeugt ein Dummy-Modell für DocTR."""
        class DummyDoc:
            def __init__(self):
                self.pages = [DummyPage()]
                
            def render(self):
                return np.zeros((500, 500, 3), dtype=np.uint8)
                
            def export(self):
                return {
                    "pages": [
                        {
                            "blocks": [
                                {
                                    "lines": [
                                        {
                                            "words": [
                                                {"value": "DocTR", "confidence": 0.95, "geometry": [0.1, 0.1, 0.2, 0.15]}
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
        
        class DummyPage:
            def __init__(self):
                self.blocks = [DummyBlock()]
                
        class DummyBlock:
            def __init__(self):
                self.lines = [DummyLine()]
                
        class DummyLine:
            def __init__(self):
                self.words = [DummyWord()]
                
        class DummyWord:
            def __init__(self):
                self.value = "DocTR"
                self.confidence = 0.95
                self.geometry = (0.1, 0.1, 0.2, 0.15)
        
        class DummyDocTR:
            def __call__(self, img, **kwargs):
                return DummyDoc()
        
        return DummyDocTR()
    
    @staticmethod
    def _create_layoutlmv3_dummy() -> Dict[str, Any]:
        """Erzeugt ein Dummy-Modell für LayoutLMv3."""
        class DummyOutput:
            def __init__(self, task):
                self.task = task
                self.logits = torch.randn(1, 10, 100)
                if task == "token_classification":
                    self.logits = torch.randn(1, 50, 9)  # 9 NER-Klassen
                elif task == "question_answering":
                    self.start_logits = torch.randn(1, 50)
                    self.end_logits = torch.randn(1, 50)
                elif task == "document_classification":
                    self.logits = torch.randn(1, 16)  # 16 Dokumentklassen
                
        class DummyLayoutLMModel:
            def __init__(self):
                pass
                
            def to(self, device):
                return self
                
            def eval(self):
                return self
                
            def __call__(self, **kwargs):
                # Je nach Aufgabe verschiedene Ausgabe zurückgeben
                task = kwargs.get("task", "token_classification")
                return DummyOutput(task)
        
        class DummyLayoutLMProcessor:
            def __init__(self, apply_ocr=True, ocr_lang=None, tesseract_config=''):
                self.apply_ocr = apply_ocr
                self.ocr_lang = ocr_lang
                self.tesseract_config = tesseract_config
                
            def __call__(self, images, padding=True, return_tensors="pt", **kwargs):
                class DummyEncoding:
                    def __init__(self):
                        self.input_ids = torch.randint(0, 1000, (1, 50))
                        self.attention_mask = torch.ones(1, 50)
                        self.token_type_ids = torch.zeros(1, 50)
                        self.bbox = torch.randint(0, 1000, (1, 50, 4))
                        self.pixel_values = torch.randn(1, 3, 224, 224)
                return DummyEncoding()
                
            def tokenizer(self, text, **kwargs):
                return {
                    "input_ids": torch.randint(0, 1000, (1, 50)),
                    "attention_mask": torch.ones(1, 50),
                    "token_type_ids": torch.zeros(1, 50)
                }
                
            def decode(self, ids, skip_special_tokens=True):
                return "Dummy LayoutLM decoded text"
                
            def batch_decode(self, ids, skip_special_tokens=True):
                return ["Dummy LayoutLM decoded text"]
        
        return {
            "processor": DummyLayoutLMProcessor(),
            "model": DummyLayoutLMModel()
        }
    
    @staticmethod
    def _create_donut_dummy() -> Dict[str, Any]:
        """Erzeugt ein Dummy-Modell für Donut."""
        class DummyDonutModel:
            def __init__(self):
                pass
                
            def to(self, device):
                return self
                
            def eval(self):
                return self
                
            def generate(self, **kwargs):
                return torch.tensor([[101, 2054, 2003, 1037, 4937, 102]])
        
        class DummyDonutProcessor:
            def __init__(self):
                self.tokenizer = self
                self.pad_token_id = 0
                self.eos_token_id = 1
                
            def __call__(self, images, return_tensors="pt"):
                return {"pixel_values": torch.randn(1, 3, 224, 224)}
                
            def tokenizer(self, prompt, return_tensors="pt", add_special_tokens=True):
                return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
                
            def decode(self, ids, skip_special_tokens=True):
                return "<fim_name>Dummy Name</fim_name><fim_address>123 Dummy St</fim_address>"
                
            def batch_decode(self, sequences, skip_special_tokens=True):
                return ["<fim_name>Dummy Name</fim_name><fim_address>123 Dummy St</fim_address>"]
        
        return {
            "processor": DummyDonutProcessor(),
            "model": DummyDonutModel()
        }
    
    @staticmethod
    def _create_microsoft_dummy() -> Any:
        """Erzeugt ein Dummy-Modell für Microsoft Read API."""
        class DummyReadOperationResult:
            def __init__(self):
                self.operation_id = "dummy-operation-123"
                
        class DummyReadOperationDetailResult:
            def __init__(self):
                self.status = "succeeded"
                self.analyze_result = DummyAnalyzeResult()
                
        class DummyAnalyzeResult:
            def __init__(self):
                self.read_results = [DummyReadResult()]
                
        class DummyReadResult:
            def __init__(self):
                self.page = 1
                self.angle = 0.0
                self.width = 800
                self.height = 1000
                self.unit = "pixel"
                self.lines = [DummyLine(), DummyLine()]
                
        class DummyLine:
            def __init__(self):
                self.text = "Dummy Microsoft Read API Text"
                self.bounding_box = [10, 10, 100, 10, 100, 40, 10, 40]
                self.words = [DummyWord(), DummyWord()]
                
        class DummyWord:
            def __init__(self):
                self.text = "Dummy"
                self.bounding_box = [10, 10, 50, 10, 50, 40, 10, 40]
                self.confidence = 0.95
        
        class DummyReadAPI:
            def read_in_stream(self, image, language=None, raw=False, **kwargs):
                return DummyReadOperationResult()
                
            def get_read_result(self, operation_id):
                return DummyReadOperationDetailResult()
        
        return DummyReadAPI()
    
    @staticmethod
    def _create_formula_dummy() -> Dict[str, Any]:
        """Erzeugt ein Dummy-Modell für FormulaRecognition."""
        class DummyLatexOCR:
            def __call__(self, img):
                return "E = mc^2"
        
        class DummyTrOCRProcessor:
            def __call__(self, images, return_tensors="pt"):
                return {"pixel_values": torch.randn(1, 3, 224, 224)}
                
            def batch_decode(self, ids, skip_special_tokens=True):
                return ["\\frac{a}{b} + \\sqrt{c}"]
        
        class DummyTrOCRModel:
            def __init__(self):
                pass
                
            def to(self, device):
                return self
                
            def eval(self):
                return self
                
            def generate(self, **kwargs):
                return torch.tensor([[101, 2054, 2003, 1037, 4937, 102]])
        
        class DummyNougatProcessor:
            def __call__(self, images, return_tensors="pt"):
                return {"pixel_values": torch.randn(1, 3, 224, 224)}
                
            def batch_decode(self, ids, skip_special_tokens=True):
                return ["$$\\int_a^b f(x) dx = F(b) - F(a)$$"]
        
        class DummyNougatModel:
            def __init__(self):
                pass
                
            def to(self, device):
                return self
                
            def eval(self):
                return self
                
            def generate(self, **kwargs):
                return torch.tensor([[101, 2054, 2003, 1037, 4937, 102]])
        
        return {
            "pix2tex": DummyLatexOCR(),
            "trocr": DummyTrOCRModel(),
            "trocr_processor": DummyTrOCRProcessor(),
            "nougat": DummyNougatModel(),
            "nougat_processor": DummyNougatProcessor()
        }
    
    @staticmethod
    def _create_tesseract_dummy() -> Any:
        """Erzeugt ein Dummy-Modell für Tesseract."""
        class DummyTesseract:
            def image_to_string(self, image, lang='eng', config=''):
                return "Dummy Tesseract OCR Text"
                
            def image_to_data(self, image, lang='eng', config='', output_type=None):
                return ("level page_num block_num par_num line_num word_num left top width height conf text\n"
                       "1 1 0 0 0 0 0 0 500 500 -1 \n"
                       "2 1 1 1 1 0 10 10 100 30 95 Dummy\n"
                       "2 1 1 1 1 1 120 10 150 30 92 Tesseract")
                
            def image_to_boxes(self, image, lang='eng', config=''):
                return "D 10 10 20 30 0\nU 25 10 35 30 0"
                
            def get_languages(self, config=''):
                return ['eng', 'deu', 'fra', 'spa']
        
        return DummyTesseract()
    
    @staticmethod
    def _create_table_extraction_dummy() -> Dict[str, Any]:
        """Erzeugt ein Dummy-Modell für TableExtraction."""
        class DummyFeatureExtractor:
            def __call__(self, images, return_tensors="pt"):
                return {
                    "pixel_values": torch.randn(1, 3, 224, 224),
                    "pixel_mask": torch.ones(1, 224, 224)
                }
        
        class DummyTableOutput:
            def __init__(self):
                self.logits = torch.randn(1, 3, 224, 224)  # Für Bounding Boxes
                self.pred_boxes = torch.tensor([[[0.1, 0.2, 0.5, 0.6]]])  # [x1, y1, x2, y2] in relativen Koordinaten
                self.pred_classes = torch.tensor([[0.1, 0.9, 0.0]])  # [no_object, table, figure]
        
        class DummyTableTransformer:
            def __init__(self):
                pass
                
            def to(self, device):
                return self
                
            def eval(self):
                return self
                
            def __call__(self, pixel_values, pixel_mask=None):
                return DummyTableOutput()
        
        return {
            "feature_extractor": DummyFeatureExtractor(),
            "model": DummyTableTransformer()
        }
    
    @staticmethod
    def _create_nougat_dummy() -> Dict[str, Any]:
        """Erzeugt ein Dummy-Modell für Nougat."""
        class DummyNougatProcessor:
            def __call__(self, images, return_tensors="pt"):
                return {
                    "pixel_values": torch.randn(1, 3, 224, 224),
                    "attention_mask": torch.ones(1, 224, 224)
                }
        
        class DummyNougatModel:
            def __init__(self):
                pass
                
            def to(self, device):
                return self
                
            def eval(self):
                return self
                
            def generate(self, **kwargs):
                return torch.tensor([[101, 2054, 2003, 1037, 4937, 102]])
        
        return {
            "processor": DummyNougatProcessor(),
            "model": DummyNougatModel()
        }
    
    @staticmethod
    def _create_generic_ocr_dummy() -> Dict[str, Any]:
        """Erzeugt ein generisches OCR-Dummy-Modell."""
        class DummyOCRModel:
            def process_image(self, image_path, options=None):
                return {
                    "text": "Generic OCR dummy text",
                    "blocks": [
                        {
                            "text": "Generic OCR dummy text",
                            "bbox": [10, 10, 100, 40],
                            "conf": 0.9
                        }
                    ],
                    "confidence": 0.9,
                    "language": "en"
                }
        
        return DummyOCRModel()
    
    @staticmethod
    def create_document_dummy(model_type: str = "generic") -> Dict[str, Any]:
        """Erzeugt ein Document-Dummy-Modell des angegebenen Typs."""
        if model_type == "excel":
            return DummyModelFactory._create_excel_document_dummy()
        elif model_type == "word":
            return DummyModelFactory._create_word_document_dummy()
        elif model_type == "hybrid":
            return DummyModelFactory._create_hybrid_document_dummy()
        elif model_type == "image":
            return DummyModelFactory._create_image_document_dummy()
        elif model_type == "universal":
            return DummyModelFactory._create_universal_document_dummy()
        else:
            return DummyModelFactory._create_generic_document_dummy()

    @staticmethod
    def _create_excel_document_dummy() -> Dict[str, Any]:
        """Erzeugt Dummy-Komponenten für Excel-Dokumente."""
        # Pandas DataFrame Dummy
        class DummyPandas:
            def read_excel(self, *args, **kwargs):
                import numpy as np
                class DummyDataFrame:
                    def __init__(self):
                        self.columns = ["A", "B", "C"]
                        self.values = np.random.rand(5, 3)
                        self.shape = (5, 3)
                    def to_dict(self, *args, **kwargs):
                        return {"A": {0: "Wert1", 1: "Wert2"}, "B": {0: 10, 1: 20}}
                    def to_html(self, *args, **kwargs):
                        return "<table><tr><td>Dummy</td></tr></table>"
                return DummyDataFrame()
            
            def read_csv(self, *args, **kwargs):
                return self.read_excel()
        
        # OpenPyXL Dummy
        class DummyWorkbook:
            def __init__(self):
                self.active = DummyWorksheet()
                self.sheetnames = ["Sheet1", "Sheet2"]
            def get_sheet_by_name(self, name):
                return DummyWorksheet()
        
        class DummyWorksheet:
            def __init__(self):
                self.title = "Sheet1"
                self.max_row = 10
                self.max_column = 5
            def cell(self, row, column):
                class DummyCell:
                    def __init__(self):
                        self.value = f"Zelle {row},{column}"
                return DummyCell()
        
        class DummyOpenpyxl:
            def load_workbook(self, *args, **kwargs):
                return DummyWorkbook()
        
        return {
            "pandas": DummyPandas(),
            "openpyxl": DummyOpenpyxl()
        }
    
    @staticmethod
    def _create_word_document_dummy() -> Dict[str, Any]:
        """Erzeugt Dummy-Komponenten für Word-Dokumente."""
        class DummyParagraph:
            def __init__(self, text="Dummy paragraph text", style=None):
                self.text = text
                self.style = style or "Normal"
                self.runs = [DummyRun("Dummy "), DummyRun("paragraph "), DummyRun("text")]
        
        class DummyRun:
            def __init__(self, text="Dummy run", bold=False, italic=False):
                self.text = text
                self.bold = bold
                self.italic = italic
                self.font = DummyFont()
        
        class DummyFont:
            def __init__(self):
                self.name = "Calibri"
                self.size = 11
                self.color = DummyColor()
        
        class DummyColor:
            def __init__(self):
                self.rgb = "000000"
        
        class DummyTable:
            def __init__(self, rows=3, cols=3):
                self.rows = [DummyRow() for _ in range(rows)]
                self._cells = [[DummyCell() for _ in range(cols)] for _ in range(rows)]
                
            def cell(self, row, col):
                return self._cells[row][col]
        
        class DummyRow:
            def __init__(self):
                self.cells = [DummyCell(), DummyCell(), DummyCell()]
        
        class DummyCell:
            def __init__(self, text="Cell text"):
                self.text = text
                self.paragraphs = [DummyParagraph(text)]
        
        class DummyDocument:
            def __init__(self):
                self.paragraphs = [
                    DummyParagraph("Heading 1", "Heading1"),
                    DummyParagraph("Regular text paragraph."),
                    DummyParagraph("Another paragraph with some content.")
                ]
                self.tables = [DummyTable()]
                self.styles = DummyStyles()
                self.core_properties = DummyDocumentProperties()
        
        class DummyStyles:
            def __init__(self):
                self.styles = ["Normal", "Heading1", "Heading2", "Title"]
        
        class DummyDocumentProperties:
            def __init__(self):
                self.author = "Dummy Author"
                self.title = "Dummy Document"
                self.created = "2023-01-01"
                self.modified = "2023-01-02"
        
        class DummyDocx:
            @staticmethod
            def Document(docx_file=None):
                return DummyDocument()
        
        return {
            "docx": DummyDocx(),
            "doc_parser": DummyDocument()
        }
    
    @staticmethod
    def _create_hybrid_document_dummy() -> Dict[str, Any]:
        """Erzeugt Dummy-Komponenten für Hybrid-Dokumente."""
        class DummyTextAdapter:
            def process_document(self, file_path, **kwargs):
                return {
                    "document_type": "text",
                    "content": "Dummy text content from text adapter",
                    "metadata": {
                        "author": "Dummy Author",
                        "created": "2023-01-01"
                    },
                    "sections": [
                        {"type": "paragraph", "content": "Section 1 content"},
                        {"type": "paragraph", "content": "Section 2 content"}
                    ]
                }
        
        class DummyImageAdapter:
            def process_document(self, file_path, **kwargs):
                return {
                    "document_type": "image",
                    "ocr_text": "Dummy OCR text from image adapter",
                    "visual_elements": [
                        {"type": "text", "bbox": [10, 10, 100, 40], "text": "Visual text 1"},
                        {"type": "table", "bbox": [10, 50, 200, 150], "rows": 3, "cols": 3}
                    ],
                    "metadata": {
                        "dimensions": "800x600",
                        "color_mode": "RGB"
                    }
                }
        
        class DummyNLP:
            def extract_entities(self, text):
                return [
                    {"type": "PERSON", "text": "John Doe", "start": 10, "end": 18},
                    {"type": "ORG", "text": "Dummy Corp", "start": 25, "end": 35}
                ]
                
            def extract_keywords(self, text):
                return ["dummy", "hybrid", "document", "processing"]
                
            def summarize(self, text):
                return "This is a dummy summary of the hybrid document content."
                
            def chunk_text(self, text, strategy="fixed"):
                if strategy == "fixed":
                    return [
                        {"content": "Chunk 1 content", "index": 0},
                        {"content": "Chunk 2 content", "index": 1}
                    ]
                elif strategy == "semantic":
                    return [
                        {"content": "Semantic chunk 1", "index": 0, "topic": "Topic A"},
                        {"content": "Semantic chunk 2", "index": 1, "topic": "Topic B"}
                    ]
                else:
                    return [{"content": text, "index": 0}]
        
        return {
            "text_adapter": DummyTextAdapter(),
            "image_adapter": DummyImageAdapter(),
            "nlp": DummyNLP()
        }
    
    @staticmethod
    def _create_image_document_adapter() -> Dict[str, Any]:
        """Erzeugt Dummy-Komponenten für Image-Dokument-Adapter."""
        class DummyOCRProcessor:
            def process_image(self, image_path, options=None):
                return {
                    "text": "Dummy OCR text for the image document",
                    "blocks": [
                        {
                            "text": "Dummy OCR block 1",
                            "bbox": [10, 10, 100, 40],
                            "conf": 0.95
                        },
                        {
                            "text": "Dummy OCR block 2",
                            "bbox": [10, 50, 200, 80],
                            "conf": 0.92
                        }
                    ],
                    "confidence": 0.94,
                    "language": "en",
                    "metadata": {
                        "page_size": "A4",
                        "orientation": "portrait"
                    }
                }
        
        class DummyColPaliProcessor:
            def process_image(self, image, query=None):
                result = {
                    "embeddings": torch.randn(1, 768),
                    "attention_weights": torch.randn(1, 12, 16, 16),
                    "visual_features": torch.randn(1, 256, 7, 7),
                    "elements": [
                        {
                            "type": "text",
                            "bbox": [0.1, 0.1, 0.5, 0.2],
                            "text": "Dummy ColPali text element",
                            "confidence": 0.92
                        },
                        {
                            "type": "image",
                            "bbox": [0.2, 0.3, 0.7, 0.6],
                            "confidence": 0.88
                        }
                    ],
                    "layout": {
                        "regions": [
                            {"type": "header", "bbox": [0.1, 0.05, 0.9, 0.15]},
                            {"type": "content", "bbox": [0.1, 0.2, 0.9, 0.8]},
                            {"type": "footer", "bbox": [0.1, 0.85, 0.9, 0.95]}
                        ]
                    }
                }
                
                if query:
                    result["query_result"] = {
                        "matched_elements": [0],
                        "relevance_score": 0.87
                    }
                
                return result
        
        class DummyFusionEngine:
            def fuse_features(self, ocr_features, vision_features, document_type="generic"):
                return {
                    "text": "Fused text content from OCR and vision",
                    "elements": [
                        {
                            "type": "paragraph",
                            "content": "Fused paragraph content",
                            "bbox": [0.1, 0.1, 0.9, 0.2],
                            "source": "hybrid",
                            "confidence": 0.94
                        },
                        {
                            "type": "table",
                            "content": "Table content here",
                            "bbox": [0.1, 0.3, 0.9, 0.6],
                            "source": "vision",
                            "confidence": 0.89,
                            "structure": {
                                "rows": 3,
                                "cols": 3,
                                "cells": [
                                    {"text": "Cell 1,1", "row": 0, "col": 0},
                                    {"text": "Cell 1,2", "row": 0, "col": 1}
                                ]
                            }
                        }
                    ],
                    "metadata": {
                        "fusion_strategy": "attention",
                        "confidence": 0.91,
                        "document_type": document_type
                    }
                }
                
            def predict_best_strategy(self, visual_features, text_features, document_metadata=None):
                return "attention", 0.87
        
        return {
            "ocr_processor": DummyOCRProcessor(),
            "colpali_processor": DummyColPaliProcessor(),
            "fusion_engine": DummyFusionEngine()
        }
    
    @staticmethod
    def _create_universal_document_dummy() -> Dict[str, Any]:
        """Erzeugt Dummy-Komponenten für Universal-Dokument-Adapter."""
        class DummyFormatHandler:
            def process_document(self, document_path, options=None):
                return {
                    "text": "Dummy document content",
                    "metadata": {
                        "author": "Dummy Author",
                        "created": "2023-01-01",
                        "modified": "2023-01-02"
                    },
                    "structure": {
                        "sections": [
                            {
                                "type": "heading",
                                "level": 1,
                                "content": "Dummy Heading 1"
                            },
                            {
                                "type": "paragraph",
                                "content": "Dummy paragraph content."
                            },
                            {
                                "type": "table",
                                "rows": 3,
                                "cols": 3,
                                "data": [
                                    ["Header 1", "Header 2", "Header 3"],
                                    ["Row1 Col1", "Row1 Col2", "Row1 Col3"],
                                    ["Row2 Col1", "Row2 Col2", "Row2 Col3"]
                                ]
                            }
                        ]
                    }
                }
                
            def extract_text(self, document_path):
                return "Dummy document content text extracted by format handler."
                
            def extract_structure(self, document_path):
                return {
                    "sections": [
                        {"type": "heading", "level": 1, "content": "Dummy Heading 1"},
                        {"type": "paragraph", "content": "Dummy paragraph content."},
                        {"type": "table", "rows": 3, "cols": 3}
                    ]
                }
                
            def extract_metadata(self, document_path):
                return {
                    "author": "Dummy Author",
                    "created": "2023-01-01",
                    "modified": "2023-01-02",
                    "title": "Dummy Document",
                    "subject": "Testing",
                    "pages": 5
                }
        
        class DummyFormatHandlers:
            def __init__(self):
                self.handlers = {
                    "docx": DummyFormatHandler(),
                    "pdf": DummyFormatHandler(),
                    "xlsx": DummyFormatHandler(),
                    "pptx": DummyFormatHandler(),
                    "txt": DummyFormatHandler(),
                    "html": DummyFormatHandler(),
                    "json": DummyFormatHandler(),
                    "md": DummyFormatHandler(),
                    "rtf": DummyFormatHandler(),
                    "xml": DummyFormatHandler(),
                    "csv": DummyFormatHandler()
                }
                
            def get(self, format_key, default=None):
                return self.handlers.get(format_key, default or self.handlers["txt"])
        
        return {
            "format_handlers": DummyFormatHandlers()
        }
    
    @staticmethod
    def _create_generic_document_dummy() -> Dict[str, Any]:
        """Erzeugt ein generisches Dokument-Dummy-Modell."""
        class DummyGenericDocumentProcessor:
            def process_document(self, document_path, options=None):
                return {
                    "text": "Generic document dummy text content",
                    "metadata": {
                        "file_name": os.path.basename(document_path) if document_path else "dummy.txt",
                        "file_size": 12345,
                        "created": "2023-01-01",
                        "modified": "2023-01-02"
                    },
                    "content_type": "text/plain"
                }
                
            def extract_text(self, document_path):
                return "Generic document dummy text content"
                
            def extract_metadata(self, document_path):
                return {
                    "file_name": os.path.basename(document_path) if document_path else "dummy.txt",
                    "file_size": 12345,
                    "created": "2023-01-01",
                    "modified": "2023-01-02"
                }
        
        return {
            "document_processor": DummyGenericDocumentProcessor()
        }
    
    @staticmethod
    def create_colpali_dummy() -> Dict[str, Any]:
        """Erzeugt ein Dummy für ColPali."""
        class DummyColPaliModel:
            def __init__(self):
                pass
            
            def __call__(self, pixel_values, **kwargs):
                # Dummy-Ausgaben erzeugen
                class DummyOutput:
                    def __init__(self):
                        self.last_hidden_state = torch.randn(1, 50, 768)
                        self.attentions = [torch.randn(1, 8, 64, 64) for _ in range(12)]
                        self.embeddings = torch.randn(1, 768)
                return DummyOutput()
            
            def eval(self):
                return self
            
            def to(self, device):
                return self
                
        class DummyProcessor:
            def __call__(self, images, **kwargs):
                return {
                    "pixel_values": torch.randn(1, 3, 224, 224),
                    "attention_mask": torch.ones(1, 196)
                }
        
        class DummyColPaliProcessor:
            def __init__(self):
                self.model = DummyColPaliModel()
                self.processor = DummyProcessor()
                
            def process_image(self, image: Any, query: Optional[str] = None) -> Dict[str, Any]:
                # Grundlegende Funktionalität simulieren
                features = {
                    "embeddings": torch.randn(1, 768).numpy(),
                    "attention_weights": torch.randn(1, 12, 16, 16).numpy(),
                    "elements": [
                        {
                            "type": "text", 
                            "bbox": [0.1, 0.1, 0.5, 0.2], 
                            "text": "Dummy ColPali text",
                            "confidence": 0.95
                        },
                        {
                            "type": "image", 
                            "bbox": [0.3, 0.4, 0.8, 0.7],
                            "confidence": 0.92
                        }
                    ],
                    "layout": {
                        "type": "document",
                        "orientation": "portrait",
                        "regions": [
                            {"type": "header", "bbox": [0.1, 0.05, 0.9, 0.15]},
                            {"type": "body", "bbox": [0.1, 0.15, 0.9, 0.85]},
                            {"type": "footer", "bbox": [0.1, 0.85, 0.9, 0.95]}
                        ]
                    }
                }
                
                # Wenn eine Anfrage vorhanden ist, Abfrageergebnisse hinzufügen
                if query:
                    features["query_result"] = {
                        "matches": [0],  # Index des passenden Elements
                        "scores": [0.87],
                        "relevance": 0.87
                    }
                
                return features
                
            def _generate_attention_map(self, outputs, input_shape):
                h, w = input_shape[0] // 16, input_shape[1] // 16
                return torch.randn(1, h, w)
                
            def get_document_similarity(self, doc1_embeddings, doc2_embeddings):
                return 0.85  # Dummy-Ähnlichkeitswert
                
        return DummyColPaliProcessor()