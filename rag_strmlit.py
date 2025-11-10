"""
Streamlit UI for Agentic RAG Incident Management System
Features: RBAC, Knowledge Base Management, Incident Resolution, HITL Approval
"""

import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import json
import os
from dotenv import load_dotenv

# Import custom modules (ensure they're in the same directory)
# from rbac_manager import RBACManager, Role
# from guardrail_system import GuardrailSystem
# from agentic_rag_system import AgenticRAG
# from hitl_workflow import HITLWorkflow
# from knowledge_base_manager import KnowledgeBaseManager


# Page config
st.set_page_config(
    page_title="Incident Management RAG System",
    page_icon="🚨",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .severity-s1 {
        background-color: #ff4444;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .severity-s2 {
        background-color: #ff9944;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .severity-s3 {
        background-color: #ffbb44;
        color: black;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .severity-s4 {
        background-color: #44bb44;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'selected_incident' not in st.session_state:
    st.session_state.selected_incident = None


# Mock database connection (replace with your actual DB path)
DB_PATH = r"C:\Users\GENAIKOLGPUSR15\Desktop\Incident_management\incident_db\data\incident_iq.db"


def get_db_connection():
    """Get database connection"""
    return sqlite3.connect(DB_PATH)


def authenticate_user(role_string: str):
    """Authenticate user with role"""
    valid_roles = [
        "CompanyName.IT.OPS.ADMIN",
        "CompanyName.IT.OPS.SRE_MANAGER",
        "CompanyName.IT.OPS.L1TEAM",
        "CompanyName.IT.OPS.L2TEAM",
        "CompanyName.IT.OPS.L3TEAM",
        "CompanyName.HR"
    ]
    
    if role_string in valid_roles:
        st.session_state.authenticated = True
        st.session_state.user_role = role_string
        return True
    return False


def get_incidents_by_severity():
    """Get incident counts by severity"""
    conn = get_db_connection()
    df = pd.read_sql_query("""
        SELECT severity_id, COUNT(*) as count
        FROM enhanced_severity_mappings
        GROUP BY severity_id
    """, conn)
    conn.close()
    return df


def get_recent_incidents(limit=10):
    """Get recent incidents"""
    conn = get_db_connection()
    df = pd.read_sql_query(f"""
        SELECT 
            id,
            payload_id,
            resource_type,
            severity_id,
            environment,
            matched_pattern,
            bert_score,
            is_llm_correction_approved,
            processed_at
        FROM enhanced_severity_mappings
        ORDER BY processed_at DESC
        LIMIT {limit}
    """, conn)
    conn.close()
    return df


def get_pending_approvals():
    """Get pending approval items"""
    conn = get_db_connection()
    df = pd.read_sql_query("""
        SELECT 
            ha.id as approval_id,
            ha.incident_id,
            esm.resource_type,
            ha.proposed_severity,
            ha.resolution_source,
            ha.confidence_score,
            ha.created_at
        FROM hitl_approvals ha
        JOIN enhanced_severity_mappings esm ON ha.incident_id = esm.id
        WHERE ha.approval_status = 'pending'
        ORDER BY ha.created_at DESC
    """, conn)
    conn.close()
    return df


def render_login_page():
    """Render login page"""
    st.markdown('<p class="main-header">🔐 Incident Management System</p>', unsafe_allow_html=True)
    
    st.markdown("### Login")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.info("Select your role to access the system")
        
        role = st.selectbox(
            "Select Role",
            [
                "CompanyName.IT.OPS.ADMIN",
                "CompanyName.IT.OPS.SRE_MANAGER",
                "CompanyName.IT.OPS.L1TEAM",
                "CompanyName.IT.OPS.L2TEAM",
                "CompanyName.IT.OPS.L3TEAM",
            ]
        )
        
        if st.button("Login", type="primary", use_container_width=True):
            if authenticate_user(role):
                st.success(f"✅ Logged in as {role}")
                st.rerun()
            else:
                st.error("❌ Invalid role")


def render_dashboard():
    """Render main dashboard"""
    st.markdown('<p class="main-header">🚨 Incident Management Dashboard</p>', unsafe_allow_html=True)
    
    # User info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Logged in as:** `{st.session_state.user_role}`")
    with col2:
        if st.button("Logout", type="secondary"):
            st.session_state.authenticated = False
            st.session_state.user_role = None
            st.rerun()
    
    st.divider()
    
    # Metrics
    severity_df = get_incidents_by_severity()
    
    st.subheader("📊 Incident Overview")
    cols = st.columns(5)
    
    severity_map = {f"s{i}": i-1 for i in range(1, 5)}
    total_incidents = severity_df['count'].sum()
    
    for severity, col_idx in severity_map.items():
        count = severity_df[severity_df['severity_id'] == severity]['count'].values
        count = count[0] if len(count) > 0 else 0
        
        with cols[col_idx]:
            st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<span class="severity-{severity}">{severity.upper()}</span>', unsafe_allow_html=True)
            st.metric("Count", count)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with cols[4]:
        st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**TOTAL**")
        st.metric("All Incidents", total_incidents)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Tabs for different functions
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Recent Incidents",
        "🤖 Process Incident",
        "✅ Pending Approvals",
        "📚 Knowledge Base"
    ])
    
    with tab1:
        render_recent_incidents()
    
    with tab2:
        render_process_incident()
    
    with tab3:
        render_pending_approvals()
    
    with tab4:
        render_knowledge_base()


def render_recent_incidents():
    """Render recent incidents table"""
    st.subheader("Recent Incidents")
    
    df = get_recent_incidents(limit=20)
    
    if df.empty:
        st.info("No incidents found")
        return
    
    # Format the dataframe
    df['approved'] = df['is_llm_correction_approved'].apply(
        lambda x: "✅" if x == 1 else "⏳"
    )
    
    # Display with selection
    st.dataframe(
        df[[
            'id', 'resource_type', 'severity_id', 'environment',
            'bert_score', 'approved', 'processed_at'
        ]],
        use_container_width=True,
        hide_index=True
    )
    
    # Select incident for details
    incident_id = st.number_input(
        "Select Incident ID for Details",
        min_value=1,
        step=1,
        key="incident_detail_id"
    )
    
    if st.button("View Details", key="view_incident_details"):
        st.session_state.selected_incident = incident_id
        render_incident_details(incident_id)


def render_incident_details(incident_id: int):
    """Render detailed incident view"""
    conn = get_db_connection()
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            id, payload_id, resource_type, resource_name, severity_id,
            environment, matched_pattern, payload, corrective_action,
            approved_corrective_action, is_llm_correction_approved,
            approved_by, bert_score
        FROM enhanced_severity_mappings
        WHERE id = ?
    """, (incident_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        st.error("Incident not found")
        return
    
    st.subheader(f"Incident Details - ID: {incident_id}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Basic Information**")
        st.write(f"**Payload ID:** {row[1]}")
        st.write(f"**Resource Type:** {row[2]}")
        st.write(f"**Resource Name:** {row[3]}")
        st.markdown(f'**Severity:** <span class="severity-{row[4]}">{row[4].upper()}</span>', 
                   unsafe_allow_html=True)
        st.write(f"**Environment:** {row[5]}")
        st.write(f"**BERT Score:** {row[12]:.3f}")
    
    with col2:
        st.markdown("**Pattern & Status**")
        st.write(f"**Matched Pattern:** {row[6]}")
        approved_status = "✅ Approved" if row[10] == 1 else "⏳ Pending"
        st.write(f"**Approval Status:** {approved_status}")
        if row[11]:
            st.write(f"**Approved By:** {row[11]}")
    
    st.divider()
    
    st.markdown("**Incident Payload:**")
    with st.expander("View Payload", expanded=False):
        st.json(json.loads(row[7]) if row[7] else {})
    
    st.divider()
    
    if row[8]:
        st.markdown("**Corrective Action (LLM Generated):**")
        st.text_area("", value=row[8], height=200, disabled=True, key="ca_llm")
    
    if row[9]:
        st.markdown("**Approved Corrective Action:**")
        st.text_area("", value=row[9], height=200, disabled=True, key="ca_approved")


def render_process_incident():
    """Render incident processing interface"""
    st.subheader("🤖 Process Incident with Agentic RAG")
    
    st.info("The agent will attempt RAG → Web Crawl → GPT Fallback to generate corrective actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        incident_id = st.number_input(
            "Incident ID to Process",
            min_value=1,
            step=1,
            key="process_incident_id"
        )
    
    with col2:
        severity_filter = st.selectbox(
            "Filter by Severity (optional)",
            ["All", "s1", "s2", "s3", "s4"]
        )
    
    # Namespace filtering
    with st.expander("Advanced: Namespace Filtering"):
        use_namespace = st.checkbox("Enable Namespace Filtering")
        if use_namespace:
            company = st.text_input("Company", value="CompanyName")
            department = st.text_input("Department", value="IT")
            team = st.text_input("Team", value="OPS")
    
    if st.button("🚀 Process Incident", type="primary"):
        with st.spinner("Processing incident..."):
            # Simulate processing
            st.success(f"✅ Incident {incident_id} processed successfully!")
            
            # Show mock result
            st.markdown("**Resolution Source:** RAG Knowledge Base")
            st.markdown("**Confidence Score:** 0.87")
            st.markdown("**Guardrail Status:** ✅ Passed")
            
            st.markdown("**Generated Corrective Action:**")
            st.text_area(
                "",
                value="""Step 1: Verify the Azure VM status using 'az vm show'
Step 2: Check resource group configuration
Step 3: Review network security group rules
Step 4: Restart the VM if necessary using 'az vm restart'
Step 5: Monitor VM metrics for 15 minutes
Step 6: Notify incident commander of resolution
Step 7: Document the resolution in knowledge base""",
                height=200,
                disabled=True
            )
            
            st.info("⏳ Incident added to approval queue for human review")


def render_pending_approvals():
    """Render pending approvals interface"""
    st.subheader("✅ Pending Approvals")
    
    # Check user permission
    if "L1TEAM" in st.session_state.user_role:
        st.warning("⚠️ L1 Team members have read-only access. Cannot approve incidents.")
        return
    
    df = get_pending_approvals()
    
    if df.empty:
        st.success("🎉 No pending approvals!")
        return
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
    
    st.divider()
    
    # Approval form
    st.subheader("Review & Approve")
    
    approval_id = st.selectbox(
        "Select Approval ID",
        df['approval_id'].tolist() if not df.empty else []
    )
    
    if approval_id:
        # Get details
        approval_row = df[df['approval_id'] == approval_id].iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Incident ID:** {approval_row['incident_id']}")
            st.write(f"**Resource:** {approval_row['resource_type']}")
            st.write(f"**Severity:** {approval_row['proposed_severity']}")
        
        with col2:
            st.write(f"**Source:** {approval_row['resolution_source']}")
            st.write(f"**Confidence:** {approval_row['confidence_score']:.2f}")
            st.write(f"**Created:** {approval_row['created_at']}")
        
        # Decision
        decision = st.radio(
            "Decision",
            ["Approve", "Modify", "Reject"],
            horizontal=True
        )
        
        override_action = None
        if decision == "Modify":
            override_action = st.text_area(
                "Modified Corrective Action",
                height=200
            )
        
        comments = st.text_area("Reviewer Comments (optional)")
        
        if st.button("Submit Decision", type="primary"):
            st.success(f"✅ Decision '{decision}' submitted for Approval ID {approval_id}")
            st.balloons()


def render_knowledge_base():
    """Render knowledge base management interface"""
    st.subheader("📚 Knowledge Base Management")
    
    # Check permissions
    user_role = st.session_state.user_role
    can_write = "ADMIN" in user_role or "SRE_MANAGER" in user_role or "L3TEAM" in user_role
    can_read = "L1TEAM" in user_role or "L2TEAM" in user_role or can_write
    
    if not can_read:
        st.error("❌ You do not have access to the knowledge base")
        return
    
    tab1, tab2 = st.tabs(["🔍 Search Knowledge", "➕ Add Knowledge"])
    
    with tab1:
        st.markdown("### Search Incident Knowledge")
        
        query = st.text_input("Enter search query")
        
        col1, col2 = st.columns(2)
        with col1:
            severity_filter_kb = st.multiselect(
                "Filter by Severity",
                ["s1", "s2", "s3", "s4"]
            )
        with col2:
            cloud_filter = st.multiselect(
                "Filter by Cloud Provider",
                ["Azure", "AWS", "GCP"]
            )
        
        if st.button("🔍 Search"):
            with st.spinner("Searching knowledge base..."):
                st.success("Found 3 matching incidents")
                
                # Mock results
                results = [
                    {
                        "incident_id": "INC-2025-001",
                        "severity": "s1",
                        "cloud": "Azure",
                        "description": "VM disk full causing application downtime",
                        "remediation": "Cleared logs and expanded disk size"
                    },
                    {
                        "incident_id": "INC-2025-002",
                        "severity": "s2",
                        "cloud": "Azure",
                        "description": "Network connectivity issue",
                        "remediation": "Reset NSG rules and restarted network service"
                    }
                ]
                
                for result in results:
                    with st.expander(f"{result['incident_id']} - {result['description'][:50]}..."):
                        st.write(f"**Severity:** {result['severity']}")
                        st.write(f"**Cloud:** {result['cloud']}")
                        st.write(f"**Description:** {result['description']}")
                        st.write(f"**Remediation:** {result['remediation']}")
    
    with tab2:
        if not can_write:
            st.warning("⚠️ You do not have permission to add knowledge")
        else:
            st.markdown("### Add New Incident Knowledge")
            
            col1, col2 = st.columns(2)
            
            with col1:
                incident_id_kb = st.text_input("Incident ID *", placeholder="INC-2025-XXX")
                cloud_provider = st.selectbox("Cloud Provider *", ["Azure", "AWS", "GCP"])
                severity_kb = st.selectbox("Severity *", ["s1", "s2", "s3", "s4"])
            
            with col2:
                environment_kb = st.selectbox("Environment *", ["Production", "Staging", "Development"])
                detection_method = st.text_input("Detection Method", placeholder="Monitoring Alert")
                incident_commander_kb = st.text_input("Incident Commander")
            
            description_kb = st.text_area("Description *", height=100)
            root_cause_kb = st.text_area("Root Cause *", height=100)
            remediation_kb = st.text_area("Remediation Steps *", height=150)
            prevention_kb = st.text_area("Prevention Measures", height=100)
            
            # RBAC settings
            st.divider()
            st.markdown("**Access Control**")
            
            col3, col4 = st.columns(2)
            with col3:
                write_group = st.text_input(
                    "Write Group",
                    value=st.session_state.user_role
                )
            with col4:
                read_group = st.text_input(
                    "Read Group",
                    value="CompanyName.IT.OPS"
                )
            
            if st.button("💾 Add to Knowledge Base", type="primary"):
                if all([incident_id_kb, cloud_provider, severity_kb, description_kb, root_cause_kb, remediation_kb]):
                    st.success(f"✅ Incident {incident_id_kb} added to knowledge base!")
                    st.info("Embedding created and stored in vector database")
                else:
                    st.error("❌ Please fill all required fields (*)")


# Main app logic
def main():
    if not st.session_state.authenticated:
        render_login_page()
    else:
        render_dashboard()


if __name__ == "__main__":
    main()
