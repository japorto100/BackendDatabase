"""
External Knowledge Base connector for knowledge graph enrichment.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Union
import requests
from urllib.parse import quote

logger = logging.getLogger(__name__)

class ExternalKBConnector:
    """Base class for connecting to external knowledge bases"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the connector"""
        self.config = config or {}
        self.name = "Generic KB"
    
    def link_entity(self, entity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find matching entities in the external KB
        
        Args:
            entity: The entity to link
            
        Returns:
            List of potential matches with confidence scores
        """
        raise NotImplementedError("Subclasses must implement link_entity")
    
    def enrich_entity(self, entity: Dict[str, Any], external_id: str) -> Dict[str, Any]:
        """
        Retrieve additional information from the external KB
        
        Args:
            entity: The entity to enrich
            external_id: ID of the entity in the external KB
            
        Returns:
            Enriched entity
        """
        raise NotImplementedError("Subclasses must implement enrich_entity")
    
    def find_related_entities(self, external_id: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Find entities related to the given entity in the external KB
        
        Args:
            external_id: ID of the entity in the external KB
            max_results: Maximum number of results to return
            
        Returns:
            List of related entities
        """
        raise NotImplementedError("Subclasses must implement find_related_entities")


class WikidataConnector(ExternalKBConnector):
    """Connector for Wikidata knowledge base"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Wikidata connector"""
        super().__init__(config)
        self.name = "Wikidata"
        self.api_endpoint = self.config.get("api_endpoint", "https://www.wikidata.org/w/api.php")
        self.sparql_endpoint = self.config.get("sparql_endpoint", "https://query.wikidata.org/sparql")
        self.user_agent = self.config.get("user_agent", "LocalGPT-Vision Knowledge Graph")
        self.retry_count = self.config.get("retry_count", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)
    
    def link_entity(self, entity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find matching entities in Wikidata
        
        Args:
            entity: The entity to link
            
        Returns:
            List of potential matches with confidence scores
        """
        # Use entity label as search term
        entity_label = entity.get("label", "")
        if not entity_label:
            return []
        
        # Add entity type to search if available
        entity_type = entity.get("type", "")
        search_term = entity_label
        
        if entity_type:
            type_mapping = {
                "Person": "human",
                "Organization": "organization",
                "Location": "geographical object",
                "Event": "event",
                "Concept": "abstract object"
            }
            wikidata_type = type_mapping.get(entity_type)
            if wikidata_type:
                search_term = f"{entity_label} {wikidata_type}"
        
        # Perform search
        try:
            for attempt in range(self.retry_count):
                try:
                    params = {
                        "action": "wbsearchentities",
                        "format": "json",
                        "language": "en",
                        "search": search_term,
                        "type": "item",
                        "limit": 5
                    }
                    
                    headers = {
                        "User-Agent": self.user_agent
                    }
                    
                    response = requests.get(self.api_endpoint, params=params, headers=headers)
                    response.raise_for_status()
                    result = response.json()
                    
                    # Process search results
                    matches = []
                    for item in result.get("search", []):
                        # Calculate confidence score
                        # Simple string similarity for demonstration
                        confidence = self._calculate_similarity(entity_label.lower(), item.get("label", "").lower())
                        
                        matches.append({
                            "external_id": item.get("id"),
                            "external_label": item.get("label"),
                            "external_description": item.get("description"),
                            "external_url": item.get("concepturi"),
                            "confidence": confidence
                        })
                    
                    # Sort by confidence
                    matches.sort(key=lambda x: x["confidence"], reverse=True)
                    return matches
                
                except requests.RequestException as e:
                    if attempt < self.retry_count - 1:
                        logger.warning(f"Retrying Wikidata search after error: {str(e)}")
                        time.sleep(self.retry_delay)
                    else:
                        logger.error(f"Failed to search Wikidata: {str(e)}")
                        return []
        
        except Exception as e:
            logger.error(f"Error linking entity to Wikidata: {str(e)}")
            return []
    
    def enrich_entity(self, entity: Dict[str, Any], external_id: str) -> Dict[str, Any]:
        """
        Retrieve additional information from Wikidata
        
        Args:
            entity: The entity to enrich
            external_id: Wikidata ID (e.g., Q42 for Douglas Adams)
            
        Returns:
            Enriched entity
        """
        try:
            # Create a copy of the entity to avoid modifying the original
            enriched_entity = entity.copy()
            
            # Prepare external references if not present
            if "external_references" not in enriched_entity:
                enriched_entity["external_references"] = []
            
            # Add Wikidata reference if not already present
            wikidata_ref = next((ref for ref in enriched_entity.get("external_references", []) 
                              if ref.get("source") == "wikidata"), None)
            
            if not wikidata_ref:
                wikidata_ref = {
                    "source": "wikidata",
                    "id": external_id,
                    "url": f"https://www.wikidata.org/wiki/{external_id}"
                }
                enriched_entity["external_references"].append(wikidata_ref)
            
            # Query Wikidata for additional information
            sparql_query = f"""
                SELECT ?property ?propertyLabel ?value ?valueLabel WHERE {{
                  wd:{external_id} ?property ?value .
                  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
                  ?property wikibase:directClaim ?p .
                  ?propertyEntity wikibase:directClaim ?p .
                }}
                LIMIT 100
            """
            
            headers = {
                "Accept": "application/sparql-results+json",
                "User-Agent": self.user_agent
            }
            
            params = {
                "query": sparql_query,
                "format": "json"
            }
            
            response = requests.get(self.sparql_endpoint, params=params, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            # Process SPARQL results
            properties = {}
            for binding in result.get("results", {}).get("bindings", []):
                property_label = binding.get("propertyLabel", {}).get("value")
                value_label = binding.get("valueLabel", {}).get("value")
                
                if property_label and value_label and property_label != value_label:
                    # Skip technical properties
                    if property_label in ["instance of", "subclass of", "part of", "has part"]:
                        continue
                    
                    # Add property if not already present
                    if property_label not in properties:
                        properties[property_label] = []
                    
                    if value_label not in properties[property_label]:
                        properties[property_label].append(value_label)
            
            # Add properties to entity
            if "properties" not in enriched_entity:
                enriched_entity["properties"] = {}
            
            # Merge properties
            for prop, values in properties.items():
                if len(values) == 1:
                    enriched_entity["properties"][prop] = values[0]
                else:
                    enriched_entity["properties"][prop] = values
            
            return enriched_entity
            
        except Exception as e:
            logger.error(f"Error enriching entity from Wikidata: {str(e)}")
            return entity
    
    def find_related_entities(self, external_id: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Find entities related to the given entity in Wikidata
        
        Args:
            external_id: Wikidata ID
            max_results: Maximum number of results to return
            
        Returns:
            List of related entities
        """
        try:
            # Query for entities connected to the given entity
            sparql_query = f"""
                SELECT ?item ?itemLabel ?relation ?relationLabel WHERE {{
                  {{ wd:{external_id} ?relation ?item . }}
                  UNION
                  {{ ?item ?relation wd:{external_id} . }}
                  ?item wdt:P31 ?type .
                  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
                }}
                LIMIT {max_results}
            """
            
            headers = {
                "Accept": "application/sparql-results+json",
                "User-Agent": self.user_agent
            }
            
            params = {
                "query": sparql_query,
                "format": "json"
            }
            
            response = requests.get(self.sparql_endpoint, params=params, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            # Process SPARQL results
            related_entities = []
            for binding in result.get("results", {}).get("bindings", []):
                item_id = binding.get("item", {}).get("value")
                item_label = binding.get("itemLabel", {}).get("value")
                relation_label = binding.get("relationLabel", {}).get("value")
                
                if item_id and item_label and relation_label:
                    related_entities.append({
                        "external_id": item_id,
                        "external_label": item_label,
                        "relation": relation_label
                    })
            
            return related_entities
        
        except Exception as e:
            logger.error(f"Error finding related entities in Wikidata: {str(e)}")
            return []


class DBpediaGermanConnector(ExternalKBConnector):
    """Connector for German DBpedia knowledge base"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the DBpedia German connector"""
        super().__init__(config)
        self.name = "DBpedia German"
        self.sparql_endpoint = self.config.get("sparql_endpoint", "https://de.dbpedia.org/sparql")
        self.user_agent = self.config.get("user_agent", "LocalGPT-Vision Knowledge Graph")
        self.retry_count = self.config.get("retry_count", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)
    
    def link_entity(self, entity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find matching entities in German DBpedia"""
        entity_label = entity.get("label", "")
        if not entity_label:
            return []
        
        try:
            # Create a SPARQL query to find entities with similar labels
            sparql_query = f"""
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT ?entity ?label ?abstract WHERE {{
                  ?entity rdfs:label ?label .
                  OPTIONAL {{ ?entity <http://dbpedia.org/ontology/abstract> ?abstract }}
                  FILTER(LANG(?label) = "de" && CONTAINS(LCASE(?label), LCASE("{entity_label}")))
                }}
                LIMIT 5
            """
            
            headers = {
                "Accept": "application/sparql-results+json",
                "User-Agent": self.user_agent
            }
            
            params = {
                "query": sparql_query,
                "format": "json"
            }
            
            for attempt in range(self.retry_count):
                try:
                    response = requests.get(self.sparql_endpoint, params=params, headers=headers)
                    response.raise_for_status()
                    result = response.json()
                    
                    matches = []
                    for binding in result.get("results", {}).get("bindings", []):
                        entity_uri = binding.get("entity", {}).get("value")
                        label = binding.get("label", {}).get("value")
                        abstract = binding.get("abstract", {}).get("value", "")
                        
                        # Calculate confidence score
                        confidence = self._calculate_similarity(entity_label.lower(), label.lower())
                        
                        if entity_uri:
                            entity_id = entity_uri.split("/")[-1]
                            matches.append({
                                "external_id": entity_id,
                                "external_label": label,
                                "external_description": abstract[:200] + "..." if len(abstract) > 200 else abstract,
                                "external_url": entity_uri,
                                "source": "dbpedia_de",
                                "confidence": confidence
                            })
                    
                    # Sort by confidence
                    matches.sort(key=lambda x: x["confidence"], reverse=True)
                    return matches
                    
                except requests.RequestException as e:
                    if attempt < self.retry_count - 1:
                        logger.warning(f"Retrying DBpedia German search after error: {str(e)}")
                        time.sleep(self.retry_delay)
                    else:
                        logger.error(f"Failed to search DBpedia German: {str(e)}")
                        return []
        
        except Exception as e:
            logger.error(f"Error linking entity to DBpedia German: {str(e)}")
            return []
    
    def enrich_entity(self, entity: Dict[str, Any], external_id: str) -> Dict[str, Any]:
        """Retrieve additional information from DBpedia German"""
        try:
            # Create a copy of the entity
            enriched_entity = entity.copy()
            
            # Prepare external references
            if "external_references" not in enriched_entity:
                enriched_entity["external_references"] = []
            
            # Add DBpedia reference if not already present
            dbpedia_ref = next((ref for ref in enriched_entity.get("external_references", []) 
                             if ref.get("source") == "dbpedia_de"), None)
            
            if not dbpedia_ref:
                dbpedia_ref = {
                    "source": "dbpedia_de",
                    "id": external_id,
                    "url": f"https://de.dbpedia.org/resource/{external_id}"
                }
                enriched_entity["external_references"].append(dbpedia_ref)
            
            # Query DBpedia for properties
            sparql_query = f"""
                PREFIX dbr: <http://de.dbpedia.org/resource/>
                SELECT ?property ?propertyLabel ?value ?valueLabel WHERE {{
                  dbr:{external_id} ?property ?value .
                  OPTIONAL {{ ?property rdfs:label ?propertyLabel . FILTER(LANG(?propertyLabel) = "de") }}
                  OPTIONAL {{ ?value rdfs:label ?valueLabel . FILTER(LANG(?valueLabel) = "de") }}
                }}
                LIMIT 100
            """
            
            headers = {
                "Accept": "application/sparql-results+json",
                "User-Agent": self.user_agent
            }
            
            params = {
                "query": sparql_query,
                "format": "json"
            }
            
            response = requests.get(self.sparql_endpoint, params=params, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            # Process properties
            properties = {}
            for binding in result.get("results", {}).get("bindings", []):
                property_uri = binding.get("property", {}).get("value")
                property_label = binding.get("propertyLabel", {}).get("value")
                
                # Use last part of URI if label not available
                if not property_label:
                    property_label = property_uri.split("/")[-1]
                
                value = binding.get("value", {}).get("value")
                value_label = binding.get("valueLabel", {}).get("value")
                
                # Use value_label if available, otherwise use value
                final_value = value_label if value_label else value
                
                # Skip certain properties
                if property_label in ["type", "label", "Wikilink", "wikiPageWikiLink"]:
                    continue
                
                # Add property
                if property_label not in properties:
                    properties[property_label] = []
                
                if final_value not in properties[property_label]:
                    properties[property_label].append(final_value)
            
            # Add properties to entity
            if "properties" not in enriched_entity:
                enriched_entity["properties"] = {}
            
            # Merge properties
            for prop, values in properties.items():
                if len(values) == 1:
                    enriched_entity["properties"][prop] = values[0]
                else:
                    enriched_entity["properties"][prop] = values
            
            return enriched_entity
            
        except Exception as e:
            logger.error(f"Error enriching entity from DBpedia German: {str(e)}")
            return entity
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity score"""
        # Simple implementation - replace with better algorithm if needed
        from difflib import SequenceMatcher
        return SequenceMatcher(None, str1, str2).ratio()


class GNDConnector(ExternalKBConnector):
    """Connector for the German National Library's Integrated Authority File (GND)"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the GND connector"""
        super().__init__(config)
        self.name = "GND"
        self.api_endpoint = self.config.get("api_endpoint", "https://lobid.org/gnd/search")
        self.user_agent = self.config.get("user_agent", "LocalGPT-Vision Knowledge Graph")
        self.retry_count = self.config.get("retry_count", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)
    
    def link_entity(self, entity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find matching entities in GND"""
        entity_label = entity.get("label", "")
        if not entity_label:
            return []
        
        try:
            params = {
                "q": entity_label,
                "format": "json",
                "size": 5
            }
            
            headers = {
                "User-Agent": self.user_agent
            }
            
            for attempt in range(self.retry_count):
                try:
                    response = requests.get(self.api_endpoint, params=params, headers=headers)
                    response.raise_for_status()
                    result = response.json()
                    
                    matches = []
                    for item in result.get("member", []):
                        # Extract label
                        label = item.get("preferredName", "")
                        if not label and "preferredName" in item:
                            label = item["preferredName"].get("value", "")
                        
                        # Extract ID
                        gnd_id = item.get("gndIdentifier", "")
                        if not gnd_id and "id" in item:
                            gnd_id = item["id"].split("/")[-1]
                        
                        # Calculate confidence score
                        confidence = self._calculate_similarity(entity_label.lower(), label.lower())
                        
                        matches.append({
                            "external_id": gnd_id,
                            "external_label": label,
                            "external_description": self._extract_description(item),
                            "external_url": item.get("id", ""),
                            "source": "gnd",
                            "confidence": confidence
                        })
                    
                    # Sort by confidence
                    matches.sort(key=lambda x: x["confidence"], reverse=True)
                    return matches
                    
                except requests.RequestException as e:
                    if attempt < self.retry_count - 1:
                        logger.warning(f"Retrying GND search after error: {str(e)}")
                        time.sleep(self.retry_delay)
                    else:
                        logger.error(f"Failed to search GND: {str(e)}")
                        return []
        
        except Exception as e:
            logger.error(f"Error linking entity to GND: {str(e)}")
            return []
    
    def _extract_description(self, item: Dict[str, Any]) -> str:
        """Extract description from GND item"""
        # Try various fields that might contain descriptions
        for field in ["definition", "biographicalOrHistoricalInformation", "note"]:
            if field in item:
                if isinstance(item[field], str):
                    return item[field]
                elif isinstance(item[field], list) and len(item[field]) > 0:
                    return item[field][0]
                elif isinstance(item[field], dict) and "value" in item[field]:
                    return item[field]["value"]
        
        return ""
    
    def enrich_entity(self, entity: Dict[str, Any], external_id: str) -> Dict[str, Any]:
        """Retrieve additional information from GND"""
        try:
            # Get detailed information about the entity
            url = f"https://lobid.org/gnd/{external_id}.json"
            
            headers = {
                "User-Agent": self.user_agent
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            item = response.json()
            
            # Create a copy of the entity
            enriched_entity = entity.copy()
            
            # Prepare external references
            if "external_references" not in enriched_entity:
                enriched_entity["external_references"] = []
            
            # Add GND reference if not already present
            gnd_ref = next((ref for ref in enriched_entity.get("external_references", []) 
                         if ref.get("source") == "gnd"), None)
            
            if not gnd_ref:
                gnd_ref = {
                    "source": "gnd",
                    "id": external_id,
                    "url": f"https://d-nb.info/gnd/{external_id}"
                }
                enriched_entity["external_references"].append(gnd_ref)
            
            # Add additional properties
            if "properties" not in enriched_entity:
                enriched_entity["properties"] = {}
            
            # Add common properties
            property_mappings = {
                "dateOfBirth": "birth_date",
                "dateOfDeath": "death_date",
                "placeOfBirth": "birth_place",
                "placeOfDeath": "death_place",
                "professionOrOccupation": "occupation",
                "biographicalOrHistoricalInformation": "biography",
                "definition": "definition",
                "geographicAreaCode": "geographic_area"
            }
            
            for gnd_prop, entity_prop in property_mappings.items():
                if gnd_prop in item:
                    value = item[gnd_prop]
                    
                    # Handle different value formats
                    if isinstance(value, list):
                        if len(value) > 0:
                            if isinstance(value[0], dict) and "label" in value[0]:
                                enriched_entity["properties"][entity_prop] = [v["label"] for v in value]
                            else:
                                enriched_entity["properties"][entity_prop] = value
                    elif isinstance(value, dict) and "label" in value:
                        enriched_entity["properties"][entity_prop] = value["label"]
                    else:
                        enriched_entity["properties"][entity_prop] = value
            
            return enriched_entity
            
        except Exception as e:
            logger.error(f"Error enriching entity from GND: {str(e)}")
            return entity
    
    def find_related_entities(self, external_id: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Find entities related to the given entity in GND"""
        # GND API doesn't directly support relationships query
        # This would require custom implementation
        return []
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity score"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, str1, str2).ratio()


class SwissALConnector(ExternalKBConnector):
    """Connector for Swiss Administrative Linked Data (Swiss-AL)"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Swiss-AL connector"""
        super().__init__(config)
        self.name = "Swiss-AL"
        self.sparql_endpoint = self.config.get("sparql_endpoint", "https://ld.admin.ch/query")
        self.user_agent = self.config.get("user_agent", "LocalGPT-Vision Knowledge Graph")
        self.retry_count = self.config.get("retry_count", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)
        # Language preference, defaulting to German for Swiss context
        self.preferred_languages = self.config.get("preferred_languages", ["de", "en", "fr", "it"])
    
    def link_entity(self, entity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find matching entities in Swiss-AL"""
        entity_label = entity.get("label", "")
        if not entity_label:
            return []
        
        try:
            # Create a SPARQL query to find entities in Swiss-AL
            lang_filters = " || ".join([f"LANG(?label) = '{lang}'" for lang in self.preferred_languages])
            
            sparql_query = f"""
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX schema: <http://schema.org/>
                
                SELECT ?entity ?label ?description WHERE {{
                  ?entity rdfs:label ?label .
                  OPTIONAL {{ ?entity schema:description ?description }}
                  FILTER(({lang_filters}) && CONTAINS(LCASE(?label), LCASE("{entity_label}")))
                }}
                LIMIT 5
            """
            
            headers = {
                "Accept": "application/sparql-results+json",
                "User-Agent": self.user_agent
            }
            
            params = {
                "query": sparql_query,
                "format": "json"
            }
            
            for attempt in range(self.retry_count):
                try:
                    response = requests.get(self.sparql_endpoint, params=params, headers=headers)
                    response.raise_for_status()
                    result = response.json()
                    
                    matches = []
                    for binding in result.get("results", {}).get("bindings", []):
                        entity_uri = binding.get("entity", {}).get("value")
                        label = binding.get("label", {}).get("value")
                        description = binding.get("description", {}).get("value", "")
                        
                        # Calculate confidence score
                        confidence = self._calculate_similarity(entity_label.lower(), label.lower())
                        
                        # Extract ID from URI
                        entity_id = entity_uri.split("/")[-1] if entity_uri else ""
                        
                        matches.append({
                            "external_id": entity_id,
                            "external_label": label,
                            "external_description": description[:200] + "..." if len(description) > 200 else description,
                            "external_url": entity_uri,
                            "source": "swiss_al",
                            "confidence": confidence,
                            "language": binding.get("label", {}).get("xml:lang", "")
                        })
                    
                    # Sort by confidence
                    matches.sort(key=lambda x: x["confidence"], reverse=True)
                    return matches
                    
                except requests.RequestException as e:
                    if attempt < self.retry_count - 1:
                        logger.warning(f"Retrying Swiss-AL search after error: {str(e)}")
                        time.sleep(self.retry_delay)
                    else:
                        logger.error(f"Failed to search Swiss-AL: {str(e)}")
                        return []
        
        except Exception as e:
            logger.error(f"Error linking entity to Swiss-AL: {str(e)}")
            return []
    
    def enrich_entity(self, entity: Dict[str, Any], external_id: str) -> Dict[str, Any]:
        """Retrieve additional information from Swiss-AL"""
        try:
            # Create a copy of the entity
            enriched_entity = entity.copy()
            
            # Prepare external references
            if "external_references" not in enriched_entity:
                enriched_entity["external_references"] = []
            
            # Add Swiss-AL reference if not already present
            swiss_al_ref = next((ref for ref in enriched_entity.get("external_references", []) 
                              if ref.get("source") == "swiss_al"), None)
            
            if not swiss_al_ref:
                # Construct the full URI if needed
                if external_id.startswith("http"):
                    uri = external_id
                else:
                    uri = f"https://ld.admin.ch/resource/{external_id}"
                    
                swiss_al_ref = {
                    "source": "swiss_al",
                    "id": external_id,
                    "url": uri
                }
                enriched_entity["external_references"].append(swiss_al_ref)
            
            # Query Swiss-AL for additional properties
            uri = swiss_al_ref["url"]
            uri_encoded = f"<{uri}>"
            
            sparql_query = f"""
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX schema: <http://schema.org/>
                
                SELECT ?prop ?propLabel ?value ?valueLabel WHERE {{
                  {uri_encoded} ?prop ?value .
                  OPTIONAL {{ ?prop rdfs:label ?propLabel . FILTER(LANG(?propLabel) = 'de' || LANG(?propLabel) = 'en') }}
                  OPTIONAL {{ ?value rdfs:label ?valueLabel . FILTER(LANG(?valueLabel) = 'de' || LANG(?valueLabel) = 'en') }}
                }}
                LIMIT 100
            """
            
            headers = {
                "Accept": "application/sparql-results+json",
                "User-Agent": self.user_agent
            }
            
            params = {
                "query": sparql_query,
                "format": "json"
            }
            
            response = requests.get(self.sparql_endpoint, params=params, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            # Process properties
            properties = {}
            for binding in result.get("results", {}).get("bindings", []):
                prop_uri = binding.get("prop", {}).get("value")
                prop_label = binding.get("propLabel", {}).get("value")
                
                # Use last part of URI if label not available
                if not prop_label:
                    prop_label = prop_uri.split("/")[-1]
                
                value = binding.get("value", {}).get("value")
                value_label = binding.get("valueLabel", {}).get("value")
                
                # Use value_label if available, otherwise use value
                final_value = value_label if value_label else value
                
                # Skip certain properties
                if prop_label in ["type", "label", "sameAs"]:
                    continue
                
                # Add property
                if prop_label not in properties:
                    properties[prop_label] = []
                
                if final_value not in properties[prop_label]:
                    properties[prop_label].append(final_value)
            
            # Add properties to entity
            if "properties" not in enriched_entity:
                enriched_entity["properties"] = {}
            
            # Merge properties
            for prop, values in properties.items():
                if len(values) == 1:
                    enriched_entity["properties"][prop] = values[0]
                else:
                    enriched_entity["properties"][prop] = values
            
            return enriched_entity
            
        except Exception as e:
            logger.error(f"Error enriching entity from Swiss-AL: {str(e)}")
            return entity
    
    def find_related_entities(self, external_id: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Find entities related to the given entity in Swiss-AL"""
        try:
            # Construct the full URI if needed
            if external_id.startswith("http"):
                uri = external_id
            else:
                uri = f"https://ld.admin.ch/resource/{external_id}"
                
            uri_encoded = f"<{uri}>"
            
            # Query for entities connected to the given entity
            sparql_query = f"""
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                
                SELECT ?entity ?entityLabel ?relation ?relationLabel WHERE {{
                  {{ {uri_encoded} ?relation ?entity . }}
                  UNION
                  {{ ?entity ?relation {uri_encoded} . }}
                  ?entity rdfs:label ?entityLabel .
                  OPTIONAL {{ ?relation rdfs:label ?relationLabel }}
                  FILTER(ISURI(?entity))
                  FILTER(LANG(?entityLabel) = 'de' || LANG(?entityLabel) = 'en')
                }}
                LIMIT {max_results}
            """
            
            headers = {
                "Accept": "application/sparql-results+json",
                "User-Agent": self.user_agent
            }
            
            params = {
                "query": sparql_query,
                "format": "json"
            }
            
            response = requests.get(self.sparql_endpoint, params=params, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            # Process results
            related_entities = []
            for binding in result.get("results", {}).get("bindings", []):
                entity_uri = binding.get("entity", {}).get("value")
                entity_label = binding.get("entityLabel", {}).get("value")
                relation_uri = binding.get("relation", {}).get("value")
                relation_label = binding.get("relationLabel", {}).get("value", relation_uri.split("/")[-1])
                
                entity_id = entity_uri.split("/")[-1]
                
                related_entities.append({
                    "external_id": entity_id,
                    "external_label": entity_label,
                    "external_url": entity_uri,
                    "relation": relation_label,
                    "relation_uri": relation_uri,
                    "source": "swiss_al"
                })
            
            return related_entities
            
        except Exception as e:
            logger.error(f"Error finding related entities in Swiss-AL: {str(e)}")
            return []
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity score"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, str1, str2).ratio()


class CascadingKBConnector(ExternalKBConnector):
    """Connector that cascades through multiple knowledge bases in priority order"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the cascading connector"""
        super().__init__(config or {})
        self.name = "Cascading KB"
        
        # Create the connector instances
        primary_kb = self.config.get("primary_kb", "swiss_al")
        self.connectors = []
        
        if primary_kb == "swiss_al":
            self.connectors.append(SwissALConnector(self.config.get("swiss_al_config")))
            self.connectors.append(WikidataConnector(self.config.get("wikidata_config")))
            self.connectors.append(DBpediaGermanConnector(self.config.get("dbpedia_german_config")))
            self.connectors.append(GNDConnector(self.config.get("gnd_config")))
        else:
            # Alternative configurations if needed
            self.connectors.append(WikidataConnector(self.config.get("wikidata_config")))
            self.connectors.append(SwissALConnector(self.config.get("swiss_al_config")))
            self.connectors.append(DBpediaGermanConnector(self.config.get("dbpedia_german_config")))
            self.connectors.append(GNDConnector(self.config.get("gnd_config")))
    
    def link_entity(self, entity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find matching entities across knowledge bases with adaptive prioritization
        
        This implementation:
        1. Queries all configured KBs in parallel
        2. Applies domain-specific prioritization based on entity type
        3. Combines results with proper source attribution
        4. Ranks by confidence score and KB priority
        
        Args:
            entity: The entity to link
            
        Returns:
            List of matching external entities with confidence scores
        """
        all_matches = []
        
        # Track matches by connector
        connector_matches = {}
        
        # Query all connectors in parallel (for better performance)
        for connector in self.connectors:
            try:
                connector_name = connector.__class__.__name__.lower()
                if hasattr(connector, 'name'):
                    connector_name = connector.name.lower()
                    
                matches = connector.link_entity(entity)
                
                # Mark the source in each match
                for match in matches:
                    match["kb_source"] = connector_name
                
                connector_matches[connector_name] = matches
                all_matches.extend(matches)
            except Exception as e:
                logger.error(f"Error with connector {connector_name}: {str(e)}")
        
        # Get entity type for prioritization
        entity_type = entity.get("type", "unknown").lower()
        
        # Apply domain-specific prioritization
        if entity_type in ["government", "administration", "location"] and "swiss_al" in connector_matches:
            # For Swiss administrative entities, prioritize SwissAL
            swiss_matches = connector_matches.get("swiss_al", [])
            if swiss_matches and swiss_matches[0]["confidence"] > 0.7:
                # Start with SwissAL matches, then add others
                prioritized = swiss_matches + [m for m in all_matches if m["kb_source"] != "swiss_al"]
                return prioritized
            
        elif entity_type in ["person", "concept", "event"] and "wikidata" in connector_matches:
            # For general knowledge entities, prioritize Wikidata
            wikidata_matches = connector_matches.get("wikidata", [])
            if wikidata_matches and wikidata_matches[0]["confidence"] > 0.7:
                # Start with Wikidata matches, then add others
                prioritized = wikidata_matches + [m for m in all_matches if m["kb_source"] != "wikidata"]
                return prioritized
        
        # Apply combined ranking by confidence and source priority
        ranked_matches = []
        
        # Get configured source priority
        source_priority = self.config.get("source_priority", [
            "swiss_al", "wikidata", "gnd", "dbpedia_german"
        ])
        
        # Calculate weighted scores
        for match in all_matches:
            source = match["kb_source"]
            confidence = match["confidence"]
            
            # Get source priority index (lower is better)
            try:
                priority_index = source_priority.index(source)
            except ValueError:
                priority_index = len(source_priority)  # End of list if not found
            
            # Calculate priority weight (normalized to 0-1, higher is better)
            priority_weight = 1.0 - (priority_index / len(source_priority))
            
            # Calculate combined score (70% confidence, 30% priority)
            combined_score = (confidence * 0.7) + (priority_weight * 0.3)
            
            ranked_matches.append((match, combined_score))
        
        # Sort by combined score
        ranked_matches.sort(key=lambda x: x[1], reverse=True)
        
        # Return sorted matches
        return [match for match, _ in ranked_matches]
    
    def enrich_entity(self, entity: Dict[str, Any], external_id: str) -> Dict[str, Any]:
        """Enrich entity using multiple KB connectors when possible"""
        # Extract KB source from the external_id format if provided
        parts = external_id.split(":")
        if len(parts) > 1:
            kb_source = parts[0]
            real_id = parts[1]
        
        # Find the specified connector
        primary_connector = next((c for c in self.connectors 
                                if c.name.lower() == kb_source.lower()), None)
        
        if primary_connector:
            enriched_entity = primary_connector.enrich_entity(entity, real_id)
            
            # Try to supplement with information from other KBs
            # Only if we have a name/label to search with
            if "label" in enriched_entity:
                secondary_results = []
                
                for connector in self.connectors:
                    if connector != primary_connector:
                        try:
                            matches = connector.link_entity(enriched_entity)
                            if matches and matches[0]["confidence"] > 0.8:
                                # High confidence match, get additional info
                                sec_entity = connector.enrich_entity(
                                    enriched_entity, 
                                    matches[0]["external_id"]
                                )
                                secondary_results.append((connector.name, sec_entity))
                        except Exception as e:
                            logger.warning(f"Error enriching from {connector.name}: {e}")
                
                # Merge properties from secondary sources
                for source_name, sec_entity in secondary_results:
                    # Add source-specific external reference
                    if "external_references" not in enriched_entity:
                        enriched_entity["external_references"] = []
                    
                    # Check if this source reference already exists
                    if not any(ref.get("source") == source_name 
                              for ref in enriched_entity.get("external_references", [])):
                        # Add new external references
                        for ref in sec_entity.get("external_references", []):
                            if ref.get("source") == source_name:
                                enriched_entity["external_references"].append(ref)
                    
                    # Merge additional properties
                    for prop, value in sec_entity.get("properties", {}).items():
                        if prop not in enriched_entity.get("properties", {}):
                            if "properties" not in enriched_entity:
                                enriched_entity["properties"] = {}
                            enriched_entity["properties"][prop] = value
            
            return enriched_entity
    
    # If no specific source or not found, use primary connector
    return self.connectors[0].enrich_entity(entity, external_id)
    
    def find_related_entities(self, external_id: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Find related entities using the appropriate KB connector"""
        # Extract KB source from the external_id format
        parts = external_id.split(":")
        if len(parts) > 1:
            kb_source = parts[0]
            real_id = parts[1]
        else:
            # Default to primary connector if source not specified
            kb_source = self.connectors[0].name.lower()
            real_id = external_id
        
        # Find the right connector
        for connector in self.connectors:
            if connector.name.lower() == kb_source.lower():
                return connector.find_related_entities(real_id, max_results)
        
        # If no matching connector found, try the primary one
        return self.connectors[0].find_related_entities(external_id, max_results)

        
        """2. Using Embeddings for Entity Similarity (Modern Approach)
        According to recent research on knowledge graph entity matching arxiv:2411.11531, 
        using embeddings can significantly improve matching accuracy:"""
    def calculate_embedding_similarity(self, entity: Dict[str, Any], external_entity: Dict[str, Any]) -> float:
        """Calculate similarity using embeddings and contextual information"""
        # Get or generate embeddings
        entity_embedding = self._get_entity_embedding(entity)
        external_embedding = self._get_entity_embedding(external_entity)
        
        if entity_embedding is None or external_embedding is None:
            # Fall back to traditional similarity if embeddings unavailable
            return self.calculate_enhanced_similarity(entity, external_entity)
        
        # Calculate cosine similarity between embeddings
        embedding_similarity = self._cosine_similarity(entity_embedding, external_embedding)
        
        # Apply context-aware adjustments
        final_score = embedding_similarity
        
        # Adjust based on property matches
        property_adjustment = self._calculate_property_match_adjustment(entity, external_entity)
        final_score = min(1.0, final_score + property_adjustment)
        
        return final_score

    def _get_entity_embedding(self, entity: Dict[str, Any]) -> Optional[List[float]]:
        """Get or generate embedding for entity"""
        # Check if entity already has embedding
        if "embedding" in entity:
            return entity["embedding"]
        
        # Generate embedding using sentence transformer if available
        try:
            from sentence_transformers import SentenceTransformer
            
            # Create model if not already created
            if not hasattr(self, "_embedding_model"):
                self._embedding_model = SentenceTransformer('all-mpnet-base-v2')
            
            # Create text representation of entity
            text = self._entity_to_text(entity)
            
            # Generate embedding
            embedding = self._embedding_model.encode(text).tolist()
            return embedding
        except:
            # If sentence_transformers not available, return None
            return None

    def _entity_to_text(self, entity: Dict[str, Any]) -> str:
        """Convert entity to text representation for embedding"""
        parts = []
        
        # Add label/name
        label = entity.get("label", entity.get("external_label", ""))
        if label:
            parts.append(f"Name: {label}")
        
        # Add type
        entity_type = entity.get("type", "")
        if entity_type:
            parts.append(f"Type: {entity_type}")
        
        # Add key properties
        properties = entity.get("properties", {})
        for key, value in properties.items():
            if isinstance(value, (str, int, float, bool)):
                parts.append(f"{key}: {value}")
        
        # Add description if available
        description = entity.get("description", entity.get("external_description", ""))
        if description:
            parts.append(f"Description: {description}")
        
        return " ".join(parts)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        
        if not vec1 or not vec2:
            return 0.0
        
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    def _calculate_property_match_adjustment(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> float:
        """Calculate adjustment based on property matches"""
        props1 = entity1.get("properties", {})
        props2 = entity2.get("properties", {})
        
        if not props1 or not props2:
            return 0.0
        
        # Count exact matches on important properties
        matches = 0
        important_props = ["name", "date", "id", "location", "country", "title"]
        
        for prop in important_props:
            if prop in props1 and prop in props2 and props1[prop] == props2[prop]:
                matches += 1
        
        # Calculate adjustment (max 0.2)
        if len(important_props) > 0:
            return min(0.2, 0.2 * (matches / len(important_props)))
        
        return 0.0

class PerplexicaKGConnector:
    def __init__(self, kg_manager, vector_store):
        self.kg_manager = kg_manager
        self.vector_store = vector_store
        self.bidirectional_indexer = BidirectionalIndexer(
            kg_manager.graph_storage, 
            vector_store
        )
    
    def enhance_search_query(self, query):
        """Enhance search query with KG entity knowledge"""
        # Extract entities from query using KG
        from models_app.llm_providers.knowledge_graph_llm_interface import KnowledgeGraphLLMInterface
        kg_interface = KnowledgeGraphLLMInterface()
        query_analysis = kg_interface.analyze_query(query)
        
        # Find related entities in KG
        entity_mentions = query_analysis.get("entities", [])
        enhanced_terms = []
        
        for entity in entity_mentions:
            # Find similar entities in KG
            similar_entities = self.bidirectional_indexer.search_similar_entities(
                entity.get("text", ""), 
                top_k=3
            )
            
            # Add related terms
            for similar in similar_entities:
                enhanced_terms.append(similar.get("label", ""))
                # Add key properties that might help search
                for key, value in similar.get("properties", {}).items():
                    if key in ["category", "synonym", "alternative_name"]:
                        enhanced_terms.append(str(value))
        
        # Create enhanced query
        if enhanced_terms:
            enhanced_query = f"{query} {' '.join(enhanced_terms)}"
            return enhanced_query
        
        return query
    
    def store_search_results(self, query, results):
        """Store valuable search results in KG and vector DB"""
        # Process only high-quality results
        for result in results:
            if result.get("score", 0) > 0.7:  # Only store high-quality results
                # Create document record
                doc_metadata = {
                    "source": "web_search",
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "search_query": query
                }
                
                # Store in vector DB
                self.vector_store.add_texts(
                    texts=[result.get("snippet", "")],
                    metadata=doc_metadata
                )
                
                # Extract entities from result
                entities = self.kg_manager.entity_extractor.extract_entities(
                    result.get("snippet", "")
                )
                
                # Add entities to KG
                for entity in entities:
                    self.kg_manager.add_entity(entity)
                    
                    # Create bidirectional link
                    self.bidirectional_indexer.index_entity_sources(
                        entity,
                        {"document_path": result.get("url", ""), "type": "web_search"}
                    )
