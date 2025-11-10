"""
Agentic RAG System with Web Crawling and GPT Fallback
Handles incident resolution with multiple strategies
"""

import os
import sqlite3
import httpx
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from bs4 import BeautifulSoup


class ResolutionSource(Enum):
    RAG = "rag"
    WEB_CRAWL = "web_crawl"
    GPT_FALLBACK = "gpt_fallback"
    HUMAN = "human"


class AgenticRAG:
    """Intelligent RAG agent with web crawling and fallback strategies"""
    
    def __init__(
        self,
        db_path: str,
        chroma_persist_dir: str,
        api_key: str,
        api_endpoint: str,
        llm_model: str,
        embedding_model: str
    ):
        self.db_path = db_path
        self.http_client = httpx.Client(verify=False, timeout=30.0)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            base_url=api_endpoint,
            api_key=api_key,
            model=llm_model,
            temperature=0.4,
            http_client=self.http_client
        )
        
        # Initialize embeddings
        self.embedding_model = OpenAIEmbeddings(
            base_url=api_endpoint,
            api_key=api_key,
            model=embedding_model,
            http_client=self.http_client
        )
        
        # Initialize vector store
        self.vectordb = Chroma(
            persist_directory=chroma_persist_dir,
            embedding_function=self.embedding_model
        )
        
        # Azure documentation URLs for web crawling
        self.azure_docs_urls = [
            "https://learn.microsoft.com/en-us/troubleshoot/azure/",
            "https://learn.microsoft.com/en-us/azure/",
        ]
    
    def _get_incident_details(self, incident_id: int) -> Optional[Dict]:
        """Fetch incident details from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    id, payload_id, subscription_id, resource_group,
                    resource_type, resource_name, environment, severity_id,
                    bert_score, matched_pattern, payload, source_type
                FROM enhanced_severity_mappings
                WHERE id = ?
            """, (incident_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            return {
                "id": row[0],
                "payload_id": row[1],
                "subscription_id": row[2],
                "resource_group": row[3],
                "resource_type": row[4],
                "resource_name": row[5],
                "environment": row[6],
                "severity_id": row[7],
                "bert_score": row[8],
                "matched_pattern": row[9],
                "payload": row[10],
                "source_type": row[11]
            }
        except Exception as e:
            print(f"❌ Error fetching incident: {e}")
            return None
    
    def _search_rag_knowledge(
        self,
        incident: Dict,
        namespace_filter: Optional[Dict] = None
    ) -> Tuple[bool, Optional[str], List[Dict]]:
        """
        Search RAG knowledge base with namespace filtering
        Returns: (found, corrective_action, source_docs)
        """
        try:
            # Build search query
            query = f"""
            Resource Type: {incident['resource_type']}
            Severity: {incident['severity_id']}
            Pattern: {incident['matched_pattern']}
            Environment: {incident['environment']}
            Issue: {incident['payload'][:500]}
            """
            
            # Search with metadata filter if provided
            search_kwargs = {"k": 5}
            if namespace_filter:
                search_kwargs["filter"] = namespace_filter
            
            retriever = self.vectordb.as_retriever(search_kwargs=search_kwargs)
            docs = retriever.invoke(query)
            
            if not docs or len(docs) == 0:
                return False, None, []
            
            # Check relevance score (if available)
            # For now, assume first result is most relevant
            context = "\n\n".join([doc.page_content for doc in docs[:3]])
            
            # Generate corrective action from RAG context
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an incident resolution expert. 
                Based on the provided context from knowledge base, generate detailed 
                corrective actions for the incident. Be specific and actionable."""),
                ("human", """Context from Knowledge Base:
                {context}
                
                Incident Details:
                {incident_details}
                
                Provide step-by-step corrective actions:""")
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            corrective_action = chain.invoke({
                "context": context,
                "incident_details": query
            })
            
            # Return sources for citation
            sources = [
                {
                    "content": doc.page_content[:200],
                    "metadata": doc.metadata
                }
                for doc in docs[:3]
            ]
            
            return True, corrective_action, sources
            
        except Exception as e:
            print(f"❌ RAG search error: {e}")
            return False, None, []
    
    def _web_crawl_resolution(self, incident: Dict) -> Tuple[bool, Optional[str]]:
        """
        Crawl Azure documentation for resolution
        Returns: (found, corrective_action)
        """
        try:
            # Build search query for Azure
            search_query = f"{incident['resource_type']} {incident['matched_pattern']} Azure troubleshooting"
            
            # Use Bing/Google search or directly crawl known URLs
            # For demo, we'll simulate with a targeted approach
            
            search_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
            
            # Note: In production, use proper web scraping with rate limiting
            # This is a simplified example
            
            response = self.http_client.get(search_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract relevant links (simplified)
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'learn.microsoft.com' in href or 'azure' in href:
                    links.append(href)
            
            if not links:
                return False, None
            
            # Fetch content from first relevant link
            target_url = links[0] if links else None
            if target_url:
                content_response = self.http_client.get(target_url)
                content_soup = BeautifulSoup(content_response.text, 'html.parser')
                
                # Extract main content
                main_content = content_soup.get_text()[:2000]
                
                # Use LLM to extract corrective action
                prompt = f"""Based on this Azure documentation:
                {main_content}
                
                For incident: {incident['matched_pattern']} on {incident['resource_type']}
                
                Extract and format corrective actions:"""
                
                response = self.llm.invoke([HumanMessage(content=prompt)])
                return True, response.content
            
            return False, None
            
        except Exception as e:
            print(f"❌ Web crawl error: {e}")
            return False, None
    
    def _gpt_fallback_resolution(self, incident: Dict) -> str:
        """Generate corrective action using GPT as fallback"""
        prompt = f"""You are an expert Azure incident responder. Generate detailed 
        corrective actions for this incident:
        
        Resource Type: {incident['resource_type']}
        Severity: {incident['severity_id']}
        Environment: {incident['environment']}
        Pattern: {incident['matched_pattern']}
        Details: {incident['payload'][:1000]}
        
        Provide comprehensive step-by-step corrective actions with:
        1. Immediate response steps
        2. Investigation procedures
        3. Resolution steps
        4. Verification steps
        5. Notification steps
        6. Rollback plan (if applicable)
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def resolve_incident(
        self,
        incident_id: int,
        namespace_filter: Optional[Dict] = None,
        require_human_approval: bool = True
    ) -> Dict:
        """
        Main agent method to resolve incident
        Tries RAG -> Web Crawl -> GPT Fallback
        """
        print(f"\n🤖 Agent resolving incident {incident_id}...")
        
        # Step 1: Fetch incident
        incident = self._get_incident_details(incident_id)
        if not incident:
            return {
                "success": False,
                "error": "Incident not found"
            }
        
        print(f"📋 Incident: {incident['resource_type']} | Severity: {incident['severity_id']}")
        
        # Step 2: Try RAG knowledge base
        print("🔍 Searching RAG knowledge base...")
        rag_found, rag_action, rag_sources = self._search_rag_knowledge(
            incident, namespace_filter
        )
        
        if rag_found and rag_action:
            print("✅ Found resolution in RAG knowledge base")
            return {
                "success": True,
                "incident_id": incident_id,
                "source": ResolutionSource.RAG.value,
                "corrective_action": rag_action,
                "confidence": incident['bert_score'],
                "sources": rag_sources,
                "requires_human_approval": require_human_approval
            }
        
        # Step 3: Try web crawling
        print("🌐 Attempting web crawl for Azure documentation...")
        web_found, web_action = self._web_crawl_resolution(incident)
        
        if web_found and web_action:
            print("✅ Found resolution via web crawl")
            return {
                "success": True,
                "incident_id": incident_id,
                "source": ResolutionSource.WEB_CRAWL.value,
                "corrective_action": web_action,
                "confidence": 0.7,
                "requires_human_approval": True  # Always require approval for web-sourced
            }
        
        # Step 4: GPT Fallback
        print("🔄 Using GPT fallback for resolution...")
        gpt_action = self._gpt_fallback_resolution(incident)
        
        return {
            "success": True,
            "incident_id": incident_id,
            "source": ResolutionSource.GPT_FALLBACK.value,
            "corrective_action": gpt_action,
            "confidence": 0.5,
            "requires_human_approval": True  # Always require approval for GPT fallback
        }
    
    def update_corrective_action(
        self,
        incident_id: int,
        corrective_action: str,
        source: str,
        approved_by: Optional[str] = None
    ) -> bool:
        """Update incident with corrective action"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            update_query = """
                UPDATE enhanced_severity_mappings
                SET corrective_action = ?,
                    is_llm_correction_approved = ?,
                    approved_by = ?,
                    approved_ts = ?,
                    updated_at = ?
                WHERE id = ?
            """
            
            is_approved = 1 if approved_by else 0
            timestamp = datetime.now().isoformat()
            
            cursor.execute(update_query, (
                corrective_action,
                is_approved,
                approved_by,
                timestamp if approved_by else None,
                timestamp,
                incident_id
            ))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Updated incident {incident_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error updating incident: {e}")
            return False


# Example Usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    agent = AgenticRAG(
        db_path="path/to/incident_iq.db",
        chroma_persist_dir="./chroma_index",
        api_key=os.getenv("api_key"),
        api_endpoint=os.getenv("api_endpoint"),
        llm_model="azure/genailab-maas-gpt-4o",
        embedding_model="azure/genailab-maas-text-embedding-3-large"
    )
    
    # Resolve incident with namespace filtering
    result = agent.resolve_incident(
        incident_id=1,
        namespace_filter={
            "company": "CompanyName",
            "department": "IT",
            "team": "Ops",
            "severity": "s1"
        }
    )
    
    print(f"\n📊 Resolution Result:")
    print(f"Source: {result.get('source')}")
    print(f"Confidence: {result.get('confidence')}")
    print(f"\nCorrective Action:\n{result.get('corrective_action')}")
