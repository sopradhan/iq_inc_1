"""
Guardrail System for validating corrective actions
Performs regex checks and basic validation
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class GuardrailResult:
    is_valid: bool
    confidence_score: float
    warnings: List[str]
    blocked_patterns: List[str]
    suggested_improvements: List[str]


class GuardrailSystem:
    """Validates corrective actions before approval"""
    
    def __init__(self):
        # Dangerous patterns that should be blocked
        self.blocked_patterns = [
            r"rm\s+-rf\s+/",  # Dangerous delete commands
            r"DROP\s+DATABASE",  # Database drops
            r"DELETE\s+FROM\s+\*",  # Wildcard deletes
            r"shutdown\s+-h\s+now",  # System shutdown
            r"format\s+[a-zA-Z]:",  # Disk format
            r"del\s+/[fFqQsS]",  # Force delete
        ]
        
        # Required keywords for valid corrective actions
        self.required_keywords = [
            r"(step|action|procedure|process)",
            r"(check|verify|validate|confirm|monitor)",
        ]
        
        # Azure-specific patterns to validate
        self.azure_patterns = {
            "resource_group": r"az\s+group",
            "vm_operations": r"az\s+vm",
            "network": r"az\s+network",
            "storage": r"az\s+storage",
            "monitoring": r"az\s+monitor",
        }
        
        # Severity-specific validation rules
        self.severity_rules = {
            "s1": {"min_steps": 5, "requires_rollback": True, "requires_notification": True},
            "s2": {"min_steps": 4, "requires_rollback": True, "requires_notification": True},
            "s3": {"min_steps": 3, "requires_rollback": False, "requires_notification": True},
            "s4": {"min_steps": 2, "requires_rollback": False, "requires_notification": False},
        }
    
    def check_blocked_patterns(self, text: str) -> List[str]:
        """Check for dangerous patterns"""
        found_patterns = []
        for pattern in self.blocked_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found_patterns.append(pattern)
        return found_patterns
    
    def check_required_keywords(self, text: str) -> Tuple[bool, List[str]]:
        """Check if text contains required keywords"""
        missing = []
        for keyword_pattern in self.required_keywords:
            if not re.search(keyword_pattern, text, re.IGNORECASE):
                missing.append(keyword_pattern)
        return len(missing) == 0, missing
    
    def validate_structure(self, text: str, severity: str) -> Tuple[bool, List[str]]:
        """Validate corrective action structure based on severity"""
        warnings = []
        
        if severity not in self.severity_rules:
            warnings.append(f"Unknown severity level: {severity}")
            return False, warnings
        
        rules = self.severity_rules[severity]
        
        # Count steps (numbered or bulleted)
        steps = re.findall(r"(?:^\d+\.|^[-*])\s+.+", text, re.MULTILINE)
        if len(steps) < rules["min_steps"]:
            warnings.append(
                f"Insufficient steps for {severity}: found {len(steps)}, "
                f"minimum required {rules['min_steps']}"
            )
        
        # Check for rollback plan
        if rules["requires_rollback"]:
            if not re.search(r"(rollback|revert|restore|backup)", text, re.IGNORECASE):
                warnings.append(f"{severity} incidents require rollback procedures")
        
        # Check for notification steps
        if rules["requires_notification"]:
            if not re.search(r"(notify|alert|inform|communicate|escalate)", text, re.IGNORECASE):
                warnings.append(f"{severity} incidents require notification steps")
        
        return len(warnings) == 0, warnings
    
    def check_azure_context(self, text: str, resource_type: str) -> List[str]:
        """Validate Azure-specific patterns"""
        suggestions = []
        
        # Check if Azure CLI commands are present
        has_azure_cmd = any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in self.azure_patterns.values()
        )
        
        if "azure" in resource_type.lower() and not has_azure_cmd:
            suggestions.append(
                "Consider adding Azure CLI commands for automation"
            )
        
        # Check for Azure resource-specific patterns
        if "vm" in resource_type.lower():
            if not re.search(self.azure_patterns["vm_operations"], text, re.IGNORECASE):
                suggestions.append("Consider adding VM-specific operations (az vm)")
        
        return suggestions
    
    def validate(
        self,
        corrective_action: str,
        severity: str,
        resource_type: str = ""
    ) -> GuardrailResult:
        """
        Main validation method
        Returns GuardrailResult with validation status
        """
        warnings = []
        blocked = []
        suggestions = []
        confidence = 1.0
        
        # 1. Check for blocked patterns
        blocked = self.check_blocked_patterns(corrective_action)
        if blocked:
            return GuardrailResult(
                is_valid=False,
                confidence_score=0.0,
                warnings=[f"Blocked dangerous pattern: {p}" for p in blocked],
                blocked_patterns=blocked,
                suggested_improvements=[]
            )
        
        # 2. Check required keywords
        has_keywords, missing = self.check_required_keywords(corrective_action)
        if not has_keywords:
            warnings.append("Missing required keywords for structured action")
            confidence -= 0.2
        
        # 3. Validate structure based on severity
        valid_structure, structure_warnings = self.validate_structure(
            corrective_action, severity
        )
        warnings.extend(structure_warnings)
        if not valid_structure:
            confidence -= 0.3
        
        # 4. Check Azure context
        azure_suggestions = self.check_azure_context(corrective_action, resource_type)
        suggestions.extend(azure_suggestions)
        
        # 5. Basic quality checks
        if len(corrective_action) < 100:
            warnings.append("Corrective action seems too brief")
            confidence -= 0.1
        
        if len(corrective_action) > 5000:
            warnings.append("Corrective action seems too verbose")
            confidence -= 0.1
        
        # Final validation
        is_valid = confidence >= 0.5 and len(blocked) == 0
        
        return GuardrailResult(
            is_valid=is_valid,
            confidence_score=max(0.0, min(1.0, confidence)),
            warnings=warnings,
            blocked_patterns=blocked,
            suggested_improvements=suggestions
        )


# Example Usage
if __name__ == "__main__":
    guardrail = GuardrailSystem()
    
    # Test case 1: Valid corrective action
    test_action_1 = """
    Step 1: Check the Azure VM status using az vm show
    Step 2: Verify the resource group configuration
    Step 3: Restart the VM using az vm restart
    Step 4: Monitor the VM health metrics
    Step 5: Notify the incident commander
    Step 6: Create backup before applying changes
    """
    
    # Test case 2: Invalid with dangerous command
    test_action_2 = """
    Step 1: Run rm -rf / to clean up
    Step 2: Restart the system
    """
    
    # Test case 3: Insufficient for S1 severity
    test_action_3 = """
    Step 1: Restart the service
    Step 2: Check logs
    """
    
    print("=== Guardrail Validation Tests ===\n")
    
    tests = [
        ("Valid S1 Action", test_action_1, "s1", "azure_vm"),
        ("Dangerous Command", test_action_2, "s2", "azure_vm"),
        ("Insufficient S1", test_action_3, "s1", "azure_vm"),
    ]
    
    for name, action, severity, resource in tests:
        print(f"Test: {name}")
        result = guardrail.validate(action, severity, resource)
        print(f"  Valid: {result.is_valid}")
        print(f"  Confidence: {result.confidence_score:.2f}")
        if result.warnings:
            print(f"  Warnings: {', '.join(result.warnings)}")
        if result.suggested_improvements:
            print(f"  Suggestions: {', '.join(result.suggested_improvements)}")
        print()
