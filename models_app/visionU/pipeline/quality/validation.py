import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class QualityValidator:
    """Validates the quality of processing results."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the quality validator."""
        self.config = config or {}
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.required_fields = self.config.get("required_fields", [])
        self.quality_metrics = []
    
    def validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate processing result quality.
        
        Args:
            result: Processing result to validate
            
        Returns:
            Dict containing validation results
        """
        validation_result = {
            "is_valid": True,
            "confidence_score": 0.0,
            "quality_score": 0.0,
            "issues": [],
            "warnings": []
        }
        
        try:
            # Check required fields
            missing_fields = self._check_required_fields(result)
            if missing_fields:
                validation_result["issues"].append({
                    "type": "missing_fields",
                    "fields": missing_fields
                })
                validation_result["is_valid"] = False
            
            # Check confidence scores
            confidence_issues = self._check_confidence_scores(result)
            if confidence_issues:
                validation_result["issues"].extend(confidence_issues)
                validation_result["is_valid"] = False
            
            # Calculate overall quality score
            quality_score = self._calculate_quality_score(result)
            validation_result["quality_score"] = quality_score
            
            # Add warnings for borderline cases
            warnings = self._generate_warnings(result, quality_score)
            if warnings:
                validation_result["warnings"].extend(warnings)
            
            # Store metrics
            self._store_quality_metrics(validation_result)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Quality validation failed: {str(e)}")
            return {
                "is_valid": False,
                "confidence_score": 0.0,
                "quality_score": 0.0,
                "issues": [{
                    "type": "validation_error",
                    "message": str(e)
                }],
                "warnings": []
            }
    
    def _check_required_fields(self, result: Dict[str, Any]) -> List[str]:
        """Check for required fields in the result."""
        missing_fields = []
        for field in self.required_fields:
            if field not in result:
                missing_fields.append(field)
        return missing_fields
    
    def _check_confidence_scores(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check confidence scores in the result."""
        issues = []
        
        # Check overall confidence if present
        if "confidence" in result:
            if result["confidence"] < self.confidence_threshold:
                issues.append({
                    "type": "low_confidence",
                    "score": result["confidence"],
                    "threshold": self.confidence_threshold
                })
        
        # Check individual element confidences
        if "elements" in result:
            for idx, element in enumerate(result["elements"]):
                if "confidence" in element and element["confidence"] < self.confidence_threshold:
                    issues.append({
                        "type": "low_element_confidence",
                        "element_index": idx,
                        "score": element["confidence"],
                        "threshold": self.confidence_threshold
                    })
        
        return issues
    
    def _calculate_quality_score(self, result: Dict[str, Any]) -> float:
        """Calculate overall quality score."""
        scores = []
        
        # Add overall confidence if present
        if "confidence" in result:
            scores.append(result["confidence"])
        
        # Add element confidences
        if "elements" in result:
            element_scores = [
                element.get("confidence", 0.0)
                for element in result["elements"]
                if "confidence" in element
            ]
            if element_scores:
                scores.append(sum(element_scores) / len(element_scores))
        
        # Calculate final score
        return sum(scores) / len(scores) if scores else 0.0
    
    def _generate_warnings(self, result: Dict[str, Any], quality_score: float) -> List[Dict[str, Any]]:
        """Generate warnings for potential issues."""
        warnings = []
        
        # Warning for borderline quality
        if 0.7 <= quality_score < 0.8:
            warnings.append({
                "type": "borderline_quality",
                "score": quality_score,
                "message": "Quality score is in borderline range"
            })
        
        # Warning for missing optional fields
        if "metadata" in result:
            expected_metadata = {"timestamp", "processor_version", "format"}
            missing_metadata = expected_metadata - set(result["metadata"].keys())
            if missing_metadata:
                warnings.append({
                    "type": "missing_metadata",
                    "fields": list(missing_metadata),
                    "message": "Some optional metadata fields are missing"
                })
        
        return warnings
    
    def _store_quality_metrics(self, validation_result: Dict[str, Any]) -> None:
        """Store quality metrics for trending analysis."""
        self.quality_metrics.append({
            "timestamp": validation_result.get("timestamp"),
            "quality_score": validation_result["quality_score"],
            "is_valid": validation_result["is_valid"],
            "issue_count": len(validation_result["issues"]),
            "warning_count": len(validation_result["warnings"])
        })
        
        # Keep only last 1000 metrics
        if len(self.quality_metrics) > 1000:
            self.quality_metrics = self.quality_metrics[-1000:]
    
    def get_quality_metrics(self) -> List[Dict[str, Any]]:
        """Get stored quality metrics."""
        return self.quality_metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset stored quality metrics."""
        self.quality_metrics = [] 