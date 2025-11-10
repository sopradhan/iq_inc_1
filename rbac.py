"""
RBAC Manager for Incident Knowledge Base
Handles role-based access control for embedding operations
"""

from enum import Enum
from typing import List, Optional
from dataclasses import dataclass


class Permission(Enum):
    READ = "read"
    WRITE = "write"
    UPDATE = "update"
    DELETE = "delete"


class Role(Enum):
    ADMIN = "CompanyName.IT.OPS.ADMIN"
    SRE_MANAGER = "CompanyName.IT.OPS.SRE_MANAGER"
    L1_TEAM = "CompanyName.IT.OPS.L1TEAM"
    L2_TEAM = "CompanyName.IT.OPS.L2TEAM"
    L3_TEAM = "CompanyName.IT.OPS.L3TEAM"
    HR = "CompanyName.HR"


@dataclass
class RolePermissions:
    role: Role
    permissions: List[Permission]
    can_access_embeddings: bool
    can_modify_embeddings: bool


class RBACManager:
    """Manages role-based access control for knowledge base operations"""
    
    def __init__(self):
        self.role_permissions = {
            Role.ADMIN: RolePermissions(
                role=Role.ADMIN,
                permissions=[Permission.READ, Permission.WRITE, Permission.UPDATE, Permission.DELETE],
                can_access_embeddings=True,
                can_modify_embeddings=True
            ),
            Role.SRE_MANAGER: RolePermissions(
                role=Role.SRE_MANAGER,
                permissions=[Permission.READ, Permission.WRITE, Permission.UPDATE],
                can_access_embeddings=True,
                can_modify_embeddings=True
            ),
            Role.L1_TEAM: RolePermissions(
                role=Role.L1_TEAM,
                permissions=[Permission.READ],
                can_access_embeddings=True,
                can_modify_embeddings=False
            ),
            Role.L2_TEAM: RolePermissions(
                role=Role.L2_TEAM,
                permissions=[Permission.READ],
                can_access_embeddings=True,
                can_modify_embeddings=False
            ),
            Role.L3_TEAM: RolePermissions(
                role=Role.L3_TEAM,
                permissions=[Permission.READ, Permission.WRITE],
                can_access_embeddings=True,
                can_modify_embeddings=True
            ),
            Role.HR: RolePermissions(
                role=Role.HR,
                permissions=[],
                can_access_embeddings=False,
                can_modify_embeddings=False
            )
        }
    
    def get_role_from_string(self, role_string: str) -> Optional[Role]:
        """Parse role string to Role enum"""
        try:
            for role in Role:
                if role.value == role_string:
                    return role
            return None
        except Exception:
            return None
    
    def has_permission(self, role: Role, permission: Permission) -> bool:
        """Check if role has specific permission"""
        if role not in self.role_permissions:
            return False
        return permission in self.role_permissions[role].permissions
    
    def can_read_embeddings(self, role: Role) -> bool:
        """Check if role can read embeddings"""
        if role not in self.role_permissions:
            return False
        return self.role_permissions[role].can_access_embeddings
    
    def can_write_embeddings(self, role: Role) -> bool:
        """Check if role can write/update embeddings"""
        if role not in self.role_permissions:
            return False
        return self.role_permissions[role].can_modify_embeddings
    
    def validate_access(self, role_string: str, operation: str) -> tuple[bool, str]:
        """
        Validate if user has access for operation
        Returns: (is_allowed, message)
        """
        role = self.get_role_from_string(role_string)
        
        if not role:
            return False, f"❌ Invalid role: {role_string}"
        
        if role == Role.HR:
            return False, "❌ HR team has no access to incident knowledge base"
        
        operation_map = {
            "read": Permission.READ,
            "write": Permission.WRITE,
            "update": Permission.UPDATE,
            "delete": Permission.DELETE,
            "fetch": Permission.READ,
            "view": Permission.READ,
            "insert": Permission.WRITE,
            "edit": Permission.UPDATE
        }
        
        required_permission = operation_map.get(operation.lower())
        if not required_permission:
            return False, f"❌ Unknown operation: {operation}"
        
        if self.has_permission(role, required_permission):
            return True, f"✅ Access granted for {role.value}"
        else:
            return False, f"❌ Access denied: {role.value} cannot perform {operation}"


# Example Usage
if __name__ == "__main__":
    rbac = RBACManager()
    
    # Test cases
    test_cases = [
        ("CompanyName.IT.OPS.ADMIN", "update"),
        ("CompanyName.IT.OPS.L1TEAM", "read"),
        ("CompanyName.IT.OPS.L1TEAM", "write"),
        ("CompanyName.IT.OPS.SRE_MANAGER", "insert"),
        ("CompanyName.HR", "read"),
    ]
    
    print("=== RBAC Access Control Tests ===\n")
    for role_str, operation in test_cases:
        allowed, msg = rbac.validate_access(role_str, operation)
        print(f"{role_str} -> {operation}: {msg}")
