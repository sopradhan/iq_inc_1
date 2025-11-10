"""
Human-In-The-Loop (HITL) Workflow for Incident Resolution
Integrates Agent, Guardrail, and Queue System
"""

import sqlite3
import json
from typing import Dict, Optional
from datetime import datetime
from enum import Enum


class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


class HITLWorkflow:
    """Manages human-in-the-loop workflow for incident resolution"""
    
    def __init__(
        self,
        db_path: str,
        agentic_rag,
        guardrail_system,
        queue_system
    ):
        self.db_path = db_path
        self.agent = agentic_rag
        self.guardrail = guardrail_system
        self.queue = queue_system
        
        self._init_approval_table()
    
    def _init_approval_table(self):
        """Initialize approval tracking table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hitl_approvals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id INTEGER NOT NULL,
                proposed_severity TEXT,
                proposed_corrective_action TEXT,
                resolution_source TEXT,
                confidence_score FLOAT,
                guardrail_result TEXT,
                approval_status TEXT DEFAULT 'pending',
                reviewer_id TEXT,
                reviewer_comments TEXT,
                approved_severity TEXT,
                approved_corrective_action TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reviewed_at TIMESTAMP,
                FOREIGN KEY (incident_id) REFERENCES enhanced_severity_mappings(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def process_incident(
        self,
        incident_id: int,
        user_role: str,
        namespace_filter: Optional[Dict] = None
    ) -> Dict:
        """
        Complete workflow: Agent resolution -> Guardrail -> Queue for approval
        """
        print(f"\n{'='*60}")
        print(f"🔄 Processing incident {incident_id}")
        print(f"{'='*60}")
        
        # Step 1: Get incident details
        incident = self.agent._get_incident_details(incident_id)
        if not incident:
            return {"success": False, "error": "Incident not found"}
        
        print(f"\n📋 Incident Details:")
        print(f"   Resource: {incident['resource_type']}")
        print(f"   Current Severity: {incident['severity_id']}")
        print(f"   BERT Score: {incident['bert_score']:.3f}")
        
        # Step 2: Agent generates resolution
        print(f"\n🤖 Agent resolving incident...")
        resolution_result = self.agent.resolve_incident(
            incident_id=incident_id,
            namespace_filter=namespace_filter
        )
        
        if not resolution_result['success']:
            return resolution_result
        
        corrective_action = resolution_result['corrective_action']
        source = resolution_result['source']
        confidence = resolution_result['confidence']
        
        print(f"\n✅ Resolution generated from: {source}")
        print(f"   Confidence: {confidence:.2f}")
        
        # Step 3: Run guardrail validation
        print(f"\n🛡️ Running guardrail validation...")
        guardrail_result = self.guardrail.validate(
            corrective_action=corrective_action,
            severity=incident['severity_id'],
            resource_type=incident['resource_type']
        )
        
        print(f"\n📊 Guardrail Results:")
        print(f"   Valid: {guardrail_result.is_valid}")
        print(f"   Confidence: {guardrail_result.confidence_score:.2f}")
        
        if guardrail_result.warnings:
            print(f"   ⚠️  Warnings:")
            for warning in guardrail_result.warnings:
                print(f"      - {warning}")
        
        if guardrail_result.blocked_patterns:
            print(f"   🚫 Blocked patterns detected:")
            for pattern in guardrail_result.blocked_patterns:
                print(f"      - {pattern}")
            
            # Reject immediately if dangerous patterns found
            return {
                "success": False,
                "error": "Dangerous patterns detected in corrective action",
                "blocked_patterns": guardrail_result.blocked_patterns
            }
        
        # Step 4: Determine if human approval needed
        requires_approval = (
            not guardrail_result.is_valid or
            confidence < 0.8 or
            source != "rag" or
            incident['severity_id'] in ['s1', 's2']
        )
        
        print(f"\n🔍 Approval Required: {requires_approval}")
        
        # Step 5: Create approval record
        approval_id = self._create_approval_record(
            incident_id=incident_id,
            proposed_severity=incident['severity_id'],
            corrective_action=corrective_action,
            source=source,
            confidence=confidence,
            guardrail_result=guardrail_result
        )
        
        # Step 6: Add to queue if approval needed
        if requires_approval:
            print(f"\n📮 Adding to approval queue...")
            queue_item = {
                "incident_id": incident_id,
                "approval_id": approval_id,
                "severity": incident['severity_id'],
                "resource_type": incident['resource_type'],
                "corrective_action": corrective_action[:500],  # Truncated
                "source": source,
                "confidence": confidence,
                "guardrail_valid": guardrail_result.is_valid,
                "warnings": guardrail_result.warnings
            }
            
            self.queue.enqueue(
                producer_id=f"agent_{user_role}",
                data=queue_item
            )
            
            return {
                "success": True,
                "incident_id": incident_id,
                "approval_id": approval_id,
                "status": "pending_approval",
                "requires_human_review": True,
                "corrective_action": corrective_action,
                "guardrail_score": guardrail_result.confidence_score
            }
        else:
            # Auto-approve if guardrail passed and high confidence
            print(f"\n✅ Auto-approving (high confidence + valid guardrail)")
            self._auto_approve(approval_id, corrective_action)
            
            # Update incident table
            self.agent.update_corrective_action(
                incident_id=incident_id,
                corrective_action=corrective_action,
                source=source,
                approved_by="auto_approved"
            )
            
            return {
                "success": True,
                "incident_id": incident_id,
                "approval_id": approval_id,
                "status": "auto_approved",
                "requires_human_review": False,
                "corrective_action": corrective_action
            }
    
    def _create_approval_record(
        self,
        incident_id: int,
        proposed_severity: str,
        corrective_action: str,
        source: str,
        confidence: float,
        guardrail_result
    ) -> int:
        """Create approval record in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO hitl_approvals (
                incident_id,
                proposed_severity,
                proposed_corrective_action,
                resolution_source,
                confidence_score,
                guardrail_result,
                approval_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            incident_id,
            proposed_severity,
            corrective_action,
            source,
            confidence,
            json.dumps({
                "is_valid": guardrail_result.is_valid,
                "confidence": guardrail_result.confidence_score,
                "warnings": guardrail_result.warnings,
                "suggestions": guardrail_result.suggested_improvements
            }),
            ApprovalStatus.PENDING.value
        ))
        
        approval_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return approval_id
    
    def _auto_approve(self, approval_id: int, corrective_action: str):
        """Auto-approve high-confidence resolutions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE hitl_approvals
            SET approval_status = ?,
                approved_corrective_action = ?,
                reviewer_id = ?,
                reviewed_at = ?
            WHERE id = ?
        """, (
            ApprovalStatus.APPROVED.value,
            corrective_action,
            "system_auto",
            datetime.now().isoformat(),
            approval_id
        ))
        
        conn.commit()
        conn.close()
    
    def review_and_approve(
        self,
        approval_id: int,
        reviewer_id: str,
        decision: str,
        override_severity: Optional[str] = None,
        override_corrective_action: Optional[str] = None,
        comments: Optional[str] = None
    ) -> Dict:
        """
        Human reviewer approves/rejects/modifies the resolution
        decision: 'approve', 'reject', 'modify'
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get approval record
            cursor.execute("""
                SELECT incident_id, proposed_severity, proposed_corrective_action
                FROM hitl_approvals
                WHERE id = ?
            """, (approval_id,))
            
            row = cursor.fetchone()
            if not row:
                return {"success": False, "error": "Approval record not found"}
            
            incident_id, proposed_severity, proposed_action = row
            
            # Determine final values
            if decision == "approve":
                final_severity = proposed_severity
                final_action = proposed_action
                status = ApprovalStatus.APPROVED
            elif decision == "modify":
                final_severity = override_severity or proposed_severity
                final_action = override_corrective_action or proposed_action
                status = ApprovalStatus.MODIFIED
            else:  # reject
                status = ApprovalStatus.REJECTED
                final_severity = None
                final_action = None
            
            # Update approval record
            cursor.execute("""
                UPDATE hitl_approvals
                SET approval_status = ?,
                    reviewer_id = ?,
                    reviewer_comments = ?,
                    approved_severity = ?,
                    approved_corrective_action = ?,
                    reviewed_at = ?
                WHERE id = ?
            """, (
                status.value,
                reviewer_id,
                comments,
                final_severity,
                final_action,
                datetime.now().isoformat(),
                approval_id
            ))
            
            # If approved or modified, update main incident table
            if status in [ApprovalStatus.APPROVED, ApprovalStatus.MODIFIED]:
                cursor.execute("""
                    UPDATE enhanced_severity_mappings
                    SET corrective_action = ?,
                        approved_corrective_action = ?,
                        is_llm_correction_approved = 1,
                        approved_by = ?,
                        approved_ts = ?,
                        updated_at = ?
                    WHERE id = ?
                """, (
                    final_action,
                    final_action,
                    reviewer_id,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    incident_id
                ))
            
            conn.commit()
            conn.close()
            
            print(f"\n✅ Incident {incident_id} {status.value} by {reviewer_id}")
            
            return {
                "success": True,
                "incident_id": incident_id,
                "approval_id": approval_id,
                "status": status.value,
                "final_severity": final_severity,
                "final_corrective_action": final_action
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_pending_approvals(self, reviewer_role: str) -> list:
        """Get all pending approvals for review"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                ha.id,
                ha.incident_id,
                esm.resource_type,
                esm.severity_id,
                ha.proposed_corrective_action,
                ha.resolution_source,
                ha.confidence_score,
                ha.created_at
            FROM hitl_approvals ha
            JOIN enhanced_severity_mappings esm ON ha.incident_id = esm.id
            WHERE ha.approval_status = 'pending'
            ORDER BY ha.created_at DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        approvals = []
        for row in rows:
            approvals.append({
                "approval_id": row[0],
                "incident_id": row[1],
                "resource_type": row[2],
                "severity": row[3],
                "corrective_action": row[4][:200],  # Truncated
                "source": row[5],
                "confidence": row[6],
                "created_at": row[7]
            })
        
        return approvals


# Example Usage
if __name__ == "__main__":
    from agentic_rag_system import AgenticRAG
    from guardrail_system import GuardrailSystem
    from jobs_queue import SQLiteQueue
    
    # Initialize components
    agent = AgenticRAG(
        db_path="incident_iq.db",
        chroma_persist_dir="./chroma_index",
        api_key="your_key",
        api_endpoint="your_endpoint",
        llm_model="azure/genailab-maas-gpt-4o",
        embedding_model="azure/genailab-maas-text-embedding-3-large"
    )
    
    guardrail = GuardrailSystem()
    queue = SQLiteQueue("incident_iq.db")
    
    # Create workflow
    hitl = HITLWorkflow(
        db_path="incident_iq.db",
        agentic_rag=agent,
        guardrail_system=guardrail,
        queue_system=queue
    )
    
    # Process incident
    result = hitl.process_incident(
        incident_id=1,
        user_role="CompanyName.IT.OPS.L1TEAM",
        namespace_filter={"severity": "s1"}
    )
    
    print(f"\n{'='*60}")
    print("WORKFLOW RESULT:")
    print(json.dumps(result, indent=2))
    
    # If pending approval, reviewer can approve
    if result.get("requires_human_review"):
        print(f"\n{'='*60}")
        print("PENDING APPROVALS:")
        pending = hitl.get_pending_approvals("CompanyName.IT.OPS.SRE_MANAGER")
        for item in pending:
            print(f"  Approval ID: {item['approval_id']}")
            print(f"  Incident: {item['incident_id']}")
            print(f"  Severity: {item['severity']}")
