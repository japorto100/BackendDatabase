from abc import ABC, abstractmethod
from typing import Dict, List

class BaseSearchProvider(ABC):
    """
    Basis-Klasse für alle Such-Provider
    """
    
    def __init__(self, api_key: str = None, base_url: str = None, 
                 custom_headers: Dict = None, config: Dict = None):
        self.api_key = api_key
        self.base_url = base_url
        self.custom_headers = custom_headers or {}
        self.config = config or {}

    @abstractmethod
    async def search(self, query: str, filters: Dict = None) -> List[Dict]:
        """
        Führt die eigentliche Suche durch
        """
        pass

    def prepare_request(self, query: str, filters: Dict = None) -> Dict:
        """
        Bereitet die API-Anfrage vor
        """
        return {
            'headers': {
                'Authorization': f'Bearer {self.api_key}',
                **self.custom_headers
            },
            'params': {
                'q': query,
                **(filters or {})
            }
        }

    def process_response(self, response: Dict) -> List[Dict]:
        """
        Verarbeitet die API-Antwort in ein einheitliches Format
        """
        return [] 