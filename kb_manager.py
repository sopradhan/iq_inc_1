"""
Knowledge Base Manager with RBAC and Namespace Support
Handles embedding operations with role-based access control
"""

import os
import sqlite3
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


@dataclass
class IncidentKnowledge:
    incident_id: str
    cloud_provider: str
    severity: str
    cause: str
    description: str
    impact: str
    remediation: str
    root_cause: str
    prevention: str
    affected_services: str
    business_impact: str
    detection_method: str
    incident_commander: str
    write_group: str
    read_group: str
    created_by: str
    
    # Namespace metadata
    company: str = "CompanyName"
    department: str = "IT"
    team: str = "OPS"


class KnowledgeBaseManager:
    """Manages incident knowledge base with RBAC"""
    
    def __init__(
        self,
        db_path: str,
        chroma_persist_dir: str,
        embedding_model: OpenAIEmbeddings,
        rbac_manager
    ):
        self.db_path = db_path
        self.chroma_persist_dir = chroma_persist_dir
        self.embedding_model = embedding_model
        self.rbac = rbac_manager
        
        # Initialize vector store
        self.vectordb = Chroma(
            persist_directory=chroma_persist_dir,
            embedding_function=embedding_model
        )
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        self._init_db()
    
    def _init_db(self):
        """Initialize incident knowledge base table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS incident_knowledge_base (
                incident_id TEXT PRIMARY KEY,
                cloud_provider TEXT NOT NULL,
                severity TEXT NOT NULL,
                cause TEXT NOT NULL,
                description TEXT NOT NULL,
                impact TEXT NOT NULL,
                remediation TEXT NOT NULL,
                root_cause TEXT NOT NULL,
                prevention TEXT NOT NULL,
                affected_services TEXT NOT NULL,
                business_impact TEXT NOT NULL,
                estimated_recovery_time TEXT NOT NULL,
                actual_recovery_time TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                detection_method TEXT NOT NULL,
                incident_commander TEXT NOT NULL,
                status TEXT NOT NULL,
                write_group TEXT NOT NULL,
                read_group TEXT NOT NULL,
                created_by TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                namespace_company TEXT DEFAULT 'CompanyName',
                namespace_department TEXT DEFAULT 'IT',
                namespace_team TEXT DEFAULT 'OPS'
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _build_namespace_metadata(
        self,
        incident: IncidentKnowledge
    ) -> Dict:
        """Build namespace metadata for filtering"""
        return {
            "company": incident.company,
            "department": incident.department,
            "team": incident.team,
            "severity": incident.severity,
            "cloud_provider": incident.cloud_provider,
            "write_group": incident.write_group,
            "read_group": incident.read_group
        }
    
    def add_knowledge(
        self,
        incident: IncidentKnowledge,
        user_role: str
    ) -> Dict:
        """
        Add incident knowledge to knowledge base
        Requires WRITE permission
        """
        # Validate RBAC
        can_write = self.rbac.validate_access(user_role, "write")
        if not can_write[0]:
            return {
                "success": False,
                "error": can_write[1]
            }
        
        try:
            # 1. Store in SQL database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO incident_knowledge_base (
                    incident_id, cloud_provider, severity, cause, description,
                    impact, remediation, root_cause, prevention, affected_services,
                    business_impact, estimated_recovery_time, actual_recovery_time,
                    start_time, end_time, detection_method, incident_commander,
                    status, write_group, read_group, created_by,
                    namespace_company, namespace_department, namespace_team
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                incident.incident_id,
                incident.cloud_provider,
                incident.severity,
                incident.cause,
                incident.description,
                incident.impact,
                incident.remediation,
                incident.root_cause,
                incident.prevention,
                incident.affected_services,
                incident.business_impact,
                "",  # estimated_recovery_time
                "",  # actual_recovery_time
                datetime.now().isoformat(),  # start_time
                "",  # end_time
                incident.detection_method,
                incident.incident_commander,
                "closed",  # status
                incident.write_group,
                incident.read_group,
                incident.created_by,
                incident.company,
                incident.department,
                incident.team
            ))
            
            conn.commit()
            conn.close()
            
            # 2. Create embedding document
            content = f"""
            Incident ID: {incident.incident_id}
            Cloud Provider: {incident.cloud_provider}
            Severity: {incident.severity}
            
            Description: {incident.description}
            
            Root Cause: {incident.root_cause}
            
            Remediation: {incident.remediation}
            
            Prevention: {incident.prevention}
            
            Impact: {incident.impact}
            Business Impact: {incident.business_impact}
            Affected Services: {incident.affected_services}
            """
            
            # 3. Create document with namespace metadata
            metadata = self._build_namespace_metadata(incident)
            metadata["incident_id"] = incident.incident_id
            metadata["source"] = "incident_knowledge_base"
            
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            
            # 4. Add to vector store
            self.vectordb.add_documents([doc])
            
            print(f"✅ Added incident {incident.incident_id} to knowledge base")
            
            return {
                "success": True,
                "incident_id": incident.incident_id,
                "namespace": f"{incident.company}.{incident.department}.{incident.team}.{incident.severity}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to add knowledge: {str(e)}"
            }
    
    def query_knowledge(
        self,
        query: str,
        user_role: str,
        namespace_filter: Optional[Dict] = None,
        k: int = 5
    ) -> Dict:
        """
        Query knowledge base with RBAC filtering
        Requires READ permission
        """
        # Validate RBAC
        can_read = self.rbac.validate_access(user_role, "read")
        if not can_read[0]:
            return {
                "success": False,
                "error": can_read[1]
            }
        
        try:
            # Build search kwargs
            search_kwargs = {"k": k}
            
            # Apply namespace filter if provided
            if namespace_filter:
                search_kwargs["filter"] = namespace_filter
            
            # Query vector store
            retriever = self.vectordb.as_retriever(search_kwargs=search_kwargs)
            docs = retriever.invoke(query)
            
            # Filter by read_group access
            filtered_docs = []
            for doc in docs:
                read_group = doc.metadata.get("read_group", "")
                # Check if user role matches read group
                if self._has_read_access(user_role, read_group):
                    filtered_docs.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })
            
            return {
                "success": True,
                "results": filtered_docs,
                "count": len(filtered_docs)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Query failed: {str(e)}"
            }
    
    def update_knowledge(
        self,
        incident_id: str,
        updates: Dict,
        user_role: str
    ) -> Dict:
        """
        Update incident knowledge
        Requires UPDATE permission
        """
        # Validate RBAC
        can_update = self.rbac.validate_access(user_role, "update")
        if not can_update[0]:
            return {
                "success": False,
                "error": can_update[1]
            }
        
        try:
            # Update SQL database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build update query dynamically
            update_fields = []
            values = []
            for key, value in updates.items():
                update_fields.append(f"{key} = ?")
                values.append(value)
            
            values.append(incident_id)
            
            query = f"""
                UPDATE incident_knowledge_base
                SET {', '.join(update_fields)}
                WHERE incident_id = ?
            """
            
            cursor.execute(query, values)
            conn.commit()
            conn.close()
            
            # Note: For vector store, you'd need to delete and re-add
            # This is simplified for demo purposes
            
            return {
                "success": True,
                "incident_id": incident_id,
                "updated_fields": list(updates.keys())
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Update failed: {str(e)}"
            }
    
    def _has_read_access(self, user_role: str, read_group: str) -> bool:
        """Check if user role has access to read group"""
        # Simplified: exact match or admin
        if "ADMIN" in user_role:
            return True
        return user_role in read_group or read_group == ""
    
    def get_namespace_stats(self, user_role: str) -> Dict:
        """Get statistics by namespace"""
        can_read = self.rbac.validate_access(user_role, "read")
        if not can_read[0]:
            return {"success": False, "error": can_read[1]}
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    namespace_company,
                    namespace_department,
                    namespace_team,
                    severity,
                    COUNT(*) as count
                FROM incident_knowledge_base
                GROUP BY namespace_company, namespace_department, namespace_team, severity
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            stats = []
            for row in rows:
                stats.append({
                    "namespace": f"{row[0]}.{row[1]}.{row[2]}.{row[3]}",
                    "count": row[4]
                })
            
            return {
                "success": True,
                "stats": stats
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# Example Usage
if __name__ == "__main__":
    from rbac_manager import RBACManager
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize
    rbac = RBACManager()
    
    import httpx
    client = httpx.Client(verify=False)
    
    embedding_model = OpenAIEmbeddings(
        base_url=os.getenv("api_endpoint"),
        api_key=os.getenv("api_key"),
        model="azure/genailab-maas-text-embedding-3-large",
        http_client=client
    )
    
    kb_manager = KnowledgeBaseManager(
        db_path="incident_knowledge.db",
        chroma_persist_dir="./chroma_kb",
        embedding_model=embedding_model,
        rbac_manager=rbac
    )
    
    # Test add knowledge as SRE Manager
    incident = IncidentKnowledge(
        incident_id="INC-2025-001",
        cloud_provider="Azure",
        severity="s1",
        cause="VM disk full",
        description="Production VM ran out of disk space",
        impact="Application downtime",
        remediation="Cleared logs and expanded disk",
        root_cause="Log rotation not configured",
        prevention="Implement log rotation and monitoring",
        affected_services="Web API",
        business_impact="Order processing halted",
        detection_method="Monitoring alert",
        incident_commander="John Doe",
        write_group="CompanyName.IT.OPS.SRE_MANAGER",
        read_group="CompanyName.IT.OPS",
        created_by="sre_manager_1"
    )
    
    result = kb_manager.add_knowledge(
        incident,
        user_role="CompanyName.IT.OPS.SRE_MANAGER"
    )
    print(result)
