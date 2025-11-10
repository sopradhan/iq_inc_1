"""
Main Integration Script for Agentic RAG Incident Management System
Combines all components: RBAC, Agent, Guardrail, HITL, Knowledge Base
"""

import os
import sys
import argparse
from dotenv import load_dotenv
import httpx

# Ensure all modules are importable
# from rbac_manager import RBACManager, Role
# from guardrail_system import GuardrailSystem
# from agentic_rag_system import AgenticRAG, ResolutionSource
# from hitl_workflow import HITLWorkflow, ApprovalStatus
# from knowledge_base_manager import KnowledgeBaseManager, IncidentKnowledge
# from jobs_queue import SQLiteQueue

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class IncidentManagementSystem:
    """
    Main system orchestrator
    Integrates all components for end-to-end incident management
    """
    
    def __init__(self, config_path: str = ".env"):
        """Initialize the complete system"""
        print("🚀 Initializing Incident Management System...")
        
        # Load environment
        load_dotenv(config_path)
        
        # Configuration
        self.db_path = os.getenv("DB_PATH", "incident_iq.db")
        self.chroma_dir = os.getenv("CHROMA_DIR", "./chroma_index")
        self.api_key = os.getenv("api_key")
        self.api_endpoint = os.getenv("api_endpoint")
        self.llm_model = os.getenv("LLM_MODEL", "azure/genailab-maas-gpt-4o")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "azure/genailab-maas-text-embedding-3-large")
        
        # Validate configuration
        if not all([self.api_key, self.api_endpoint]):
            raise ValueError("❌ Missing required environment variables")
        
        # Initialize HTTP client
        self.http_client = httpx.Client(verify=False, timeout=60.0)
        
        # Initialize components
        print("   📋 Initializing RBAC Manager...")
        # self.rbac = RBACManager()
        
        print("   🛡️  Initializing Guardrail System...")
        # self.guardrail = GuardrailSystem()
        
        print("   📮 Initializing Queue System...")
        # self.queue = SQLiteQueue(self.db_path)
        
        print("   🤖 Initializing LLM and Embeddings...")
        self.llm = ChatOpenAI(
            base_url=self.api_endpoint,
            api_key=self.api_key,
            model=self.llm_model,
            temperature=0.4,
            http_client=self.http_client
        )
        
        self.embedding_model = OpenAIEmbeddings(
            base_url=self.api_endpoint,
            api_key=self.api_key,
            model=self.embedding_model_name,
            http_client=self.http_client
        )
        
        print("   🔍 Initializing Agentic RAG...")
        # self.agent = AgenticRAG(
        #     db_path=self.db_path,
        #     chroma_persist_dir=self.chroma_dir,
        #     api_key=self.api_key,
        #     api_endpoint=self.api_endpoint,
        #     llm_model=self.llm_model,
        #     embedding_model=self.embedding_model_name
        # )
        
        print("   📚 Initializing Knowledge Base Manager...")
        # self.kb_manager = KnowledgeBaseManager(
        #     db_path=self.db_path,
        #     chroma_persist_dir=self.chroma_dir,
        #     embedding_model=self.embedding_model,
        #     rbac_manager=self.rbac
        # )
        
        print("   ✅ Initializing HITL Workflow...")
        # self.hitl = HITLWorkflow(
        #     db_path=self.db_path,
        #     agentic_rag=self.agent,
        #     guardrail_system=self.guardrail,
        #     queue_system=self.queue
        # )
        
        print("✅ System initialized successfully!\n")
    
    def process_batch_incidents(
        self,
        incident_ids: list,
        user_role: str,
        namespace_filter: dict = None
    ):
        """
        Process multiple incidents in batch
        """
        print(f"📦 Processing {len(incident_ids)} incidents in batch...")
        
        results = []
        for incident_id in incident_ids:
            print(f"\n{'='*70}")
            result = self.hitl.process_incident(
                incident_id=incident_id,
                user_role=user_role,
                namespace_filter=namespace_filter
            )
            results.append(result)
        
        # Summary
        print(f"\n{'='*70}")
        print("📊 BATCH PROCESSING SUMMARY")
        print(f"{'='*70}")
        
        auto_approved = sum(1 for r in results if r.get('status') == 'auto_approved')
        pending = sum(1 for r in results if r.get('requires_human_review'))
        failed = sum(1 for r in results if not r.get('success'))
        
        print(f"✅ Auto-approved: {auto_approved}")
        print(f"⏳ Pending Review: {pending}")
        print(f"❌ Failed: {failed}")
        print(f"📊 Total: {len(incident_ids)}")
        
        return results
    
    def add_knowledge_from_incident(
        self,
        incident_id: int,
        user_role: str
    ):
        """
        Convert an approved incident into knowledge base entry
        """
        print(f"📚 Converting incident {incident_id} to knowledge...")
        
        # Get incident details
        incident = self.agent._get_incident_details(incident_id)
        if not incident:
            print(f"❌ Incident {incident_id} not found")
            return False
        
        # Check if approved
        if not incident.get('is_llm_correction_approved'):
            print(f"⚠️  Incident {incident_id} not yet approved")
            return False
        
        # Create knowledge entry
        knowledge = IncidentKnowledge(
            incident_id=f"INC-{incident_id}",
            cloud_provider="Azure",  # Infer from resource_type
            severity=incident['severity_id'],
            cause=incident['matched_pattern'],
            description=incident['payload'][:500],
            impact="Service disruption",
            remediation=incident['approved_corrective_action'] or incident['corrective_action'],
            root_cause=incident['matched_pattern'],
            prevention="Implement monitoring and preventive measures",
            affected_services=incident['resource_type'],
            business_impact="User-facing service degradation",
            detection_method="Automated monitoring",
            incident_commander="SRE Team",
            write_group=user_role,
            read_group="CompanyName.IT.OPS",
            created_by=user_role
        )
        
        # Add to knowledge base
        result = self.kb_manager.add_knowledge(knowledge, user_role)
        
        if result['success']:
            print(f"✅ Incident {incident_id} added to knowledge base")
            print(f"   Namespace: {result['namespace']}")
        else:
            print(f"❌ Failed to add to knowledge base: {result['error']}")
        
        return result['success']
    
    def run_interactive_review(self, user_role: str):
        """
        Interactive CLI for reviewing pending approvals
        """
        print("\n" + "="*70)
        print("✅ INTERACTIVE APPROVAL REVIEW")
        print("="*70)
        
        # Get pending approvals
        pending = self.hitl.get_pending_approvals(user_role)
        
        if not pending:
            print("\n🎉 No pending approvals!")
            return
        
        print(f"\n📋 Found {len(pending)} pending approval(s):\n")
        
        for i, item in enumerate(pending, 1):
            print(f"{i}. Approval ID: {item['approval_id']}")
            print(f"   Incident: {item['incident_id']}")
            print(f"   Severity: {item['severity']}")
            print(f"   Resource: {item['resource_type']}")
            print(f"   Source: {item['source']}")
            print(f"   Confidence: {item['confidence']:.2f}")
            print()
        
        # Interactive selection
        while True:
            try:
                choice = input("\nEnter approval ID to review (or 'q' to quit): ")
                
                if choice.lower() == 'q':
                    break
                
                approval_id = int(choice)
                
                # Get details and review
                self._review_approval_interactive(approval_id, user_role)
                
            except ValueError:
                print("❌ Invalid input. Please enter a number or 'q'")
            except KeyboardInterrupt:
                print("\n\n👋 Exiting...")
                break
    
    def _review_approval_interactive(self, approval_id: int, reviewer_id: str):
        """Interactive review for a single approval"""
        print(f"\n{'='*70}")
        print(f"Reviewing Approval ID: {approval_id}")
        print(f"{'='*70}\n")
        
        # Show full details here
        print("1. Approve")
        print("2. Modify")
        print("3. Reject")
        print("4. Skip")
        
        choice = input("\nYour decision (1-4): ")
        
        decision_map = {
            "1": "approve",
            "2": "modify",
            "3": "reject"
        }
        
        if choice == "4":
            return
        
        decision = decision_map.get(choice)
        if not decision:
            print("❌ Invalid choice")
            return
        
        comments = input("Comments (optional): ")
        
        override_action = None
        if decision == "modify":
            print("\nEnter modified corrective action (press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if line:
                    lines.append(line)
                else:
                    break
            override_action = "\n".join(lines)
        
        # Submit decision
        result = self.hitl.review_and_approve(
            approval_id=approval_id,
            reviewer_id=reviewer_id,
            decision=decision,
            override_corrective_action=override_action,
            comments=comments
        )
        
        if result['success']:
            print(f"\n✅ Decision submitted: {decision.upper()}")
        else:
            print(f"\n❌ Error: {result['error']}")
    
    def generate_system_report(self):
        """Generate system statistics report"""
        print("\n" + "="*70)
        print("📊 INCIDENT MANAGEMENT SYSTEM REPORT")
        print("="*70 + "\n")
        
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total incidents
        cursor.execute("SELECT COUNT(*) FROM enhanced_severity_mappings")
        total = cursor.fetchone()[0]
        print(f"Total Incidents: {total}")
        
        # By severity
        cursor.execute("""
            SELECT severity_id, COUNT(*) 
            FROM enhanced_severity_mappings 
            GROUP BY severity_id
        """)
        print("\nBy Severity:")
        for row in cursor.fetchall():
            print(f"  {row[0]}: {row[1]}")
        
        # Approval status
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN is_llm_correction_approved = 1 THEN 1 ELSE 0 END) as approved,
                SUM(CASE WHEN is_llm_correction_approved = 0 THEN 1 ELSE 0 END) as pending
            FROM enhanced_severity_mappings
        """)
        approved, pending = cursor.fetchone()
        print(f"\nApproval Status:")
        print(f"  Approved: {approved}")
        print(f"  Pending: {pending}")
        
        # Knowledge base stats
        cursor.execute("SELECT COUNT(*) FROM incident_knowledge_base")
        kb_count = cursor.fetchone()[0]
        print(f"\nKnowledge Base Entries: {kb_count}")
        
        conn.close()
        
        print("\n" + "="*70)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Incident Management System with Agentic RAG"
    )
    
    parser.add_argument(
        '--mode',
        choices=['batch', 'interactive', 'report', 'add-knowledge'],
        default='interactive',
        help='Operation mode'
    )
    
    parser.add_argument(
        '--incidents',
        type=int,
        nargs='+',
        help='Incident IDs to process (for batch mode)'
    )
    
    parser.add_argument(
        '--role',
        type=str,
        default='CompanyName.IT.OPS.SRE_MANAGER',
        help='User role for RBAC'
    )
    
    parser.add_argument(
        '--incident-id',
        type=int,
        help='Single incident ID (for add-knowledge mode)'
    )
    
    args = parser.parse_args()
    
    # Initialize system
    try:
        system = IncidentManagementSystem()
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        sys.exit(1)
    
    # Execute based on mode
    if args.mode == 'batch':
        if not args.incidents:
            print("❌ --incidents required for batch mode")
            sys.exit(1)
        
        system.process_batch_incidents(
            incident_ids=args.incidents,
            user_role=args.role
        )
    
    elif args.mode == 'interactive':
        system.run_interactive_review(user_role=args.role)
    
    elif args.mode == 'report':
        system.generate_system_report()
    
    elif args.mode == 'add-knowledge':
        if not args.incident_id:
            print("❌ --incident-id required for add-knowledge mode")
            sys.exit(1)
        
        system.add_knowledge_from_incident(
            incident_id=args.incident_id,
            user_role=args.role
        )


if __name__ == "__main__":
    # Example usage in comments:
    # python main_integration.py --mode batch --incidents 1 2 3 --role CompanyName.IT.OPS.SRE_MANAGER
    # python main_integration.py --mode interactive --role CompanyName.IT.OPS.L3TEAM
    # python main_integration.py --mode report
    # python main_integration.py --mode add-knowledge --incident-id 5 --role CompanyName.IT.OPS.ADMIN
    
    main()
