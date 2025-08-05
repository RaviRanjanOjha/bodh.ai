from typing import Dict, List, Optional
import re
from config import settings
from typing import Any
import json

class ComplianceChecker:
    def __init__(self, llm=None):  
        self.blocked_terms = [
            "insider trading", "inside information", "material nonpublic information",
            "manipulate the market", "front running", "ponzi scheme", "pyramid scheme",
            "guaranteed 100% returns", "absolutely risk-free", "guaranteed profit"
        ]
        self.blocked_patterns = [
            r"\bguarante[ed]\s+\d+%\s+return\b",  
            r"\babsolutely\s+risk[\-\s]*free\b",  
            r"\bponzi\s+scheme\b",
            r"\bpyramid\s+scheme\b"
        ]
        self.llm = llm 


    def check_message(self, message: str, client_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive compliance check with detailed results
        Returns dict with:
        - is_compliant: bool
        - reasons: list of strings if non-compliant
        - recommendations: list of strings for fixing issues
        """
        result = {
            "is_compliant": True,
            "reasons": [],
            "recommendations": []
        }
        
        # Phase 1: Rule-based checks
        failed_terms = [term for term in self.blocked_terms if term in message.lower()]
        failed_patterns = []
        
        for pattern in self.blocked_patterns:
            if re.search(pattern, message.lower()):
                failed_patterns.append(pattern)
                
        if failed_terms or failed_patterns:
            result["is_compliant"] = False
            result["reasons"].extend(
                f"Blocked term found: {term}" for term in failed_terms
            )
            result["reasons"].extend(
                f"Blocked pattern found: {pattern}" for pattern in failed_patterns
            )
            result["recommendations"].append(
                "Remove or rephrase prohibited terms/phrases"
            )
            
        # Phase 2: LLM check - DISABLED FOR NOW due to API issues
        # if result["is_compliant"]:  # Only do LLM check if passed phase 1
        #     llm_result = self._detailed_llm_check(message, client_id)
        #     if not llm_result.get("is_compliant", True):
        #         result["is_compliant"] = False
        #         result["reasons"].extend(llm_result.get("reasons", []))
        #         result["recommendations"].extend(llm_result.get("recommendations", []))
                
        return result

    def _detailed_llm_check(self, message: str, client_id: Optional[str]) -> Dict[str, Any]:
        """Detailed LLM compliance check with explanation"""
        if not self.llm:  
            return {"is_compliant": True, "reasons": [], "recommendations": []}
            
        prompt = f"""
        Perform comprehensive compliance check on this message:
        Message: {message}
        
        {f"Client ID: {client_id}" if client_id else ""}
        
        Analyze for:
        1. Regulatory compliance (SEC, FINRA)
        2. Fiduciary duty considerations
        3. Suitability for client risk profile
        4. Proper disclosures
        5. Conflicts of interest
        
        Return response as JSON with:
        - is_compliant (boolean)
        - reasons (list of strings if non-compliant)
        - recommendations (list of strings)
        this type of result is required bro: {"is_compliant": True, "reasons": [], "recommendations": []}
        JSON ONLY, no other text.
        """
        
        response = self.llm.generate_response(prompt) 
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"is_compliant": True, "reasons": [], "recommendations": []}