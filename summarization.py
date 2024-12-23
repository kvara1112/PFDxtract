# summarization.py
from dataclasses import dataclass
from typing import Dict, List
import re
import streamlit as st
import pandas as pd
import logging
from collections import defaultdict

@dataclass
class DocumentSummary:
    title: str
    extractive: str
    abstractive: str
    metadata: Dict
    facts: List[Dict]
    confidence: float

# Define key patterns
SECTION_PATTERNS = {
    'circumstances': r'CIRCUMSTANCES OF (?:THE )?DEATH\s*(.*?)(?=CORONER|$)',
    'concerns': r'CORONER'?S CONCERNS\s*(.*?)(?=MATTERS|$)', 
    'actions': r'(?:MATTERS|ACTION) OF CONCERN\s*(.*?)(?=\n\n|$)',
    'response': r'(?:In response to|Following)\s*(.*?)(?=\n\n|$)'
}

METADATA_PATTERNS = {
    'ref': r'Ref(?:erence)?:\s*([-\d]+)',
    'date': r'Date of report:\s*(\d{1,2}(?:/|-)\d{1,2}(?:/|-)\d{4})',
    'deceased': r'Deceased name:\s*([^\n]+)',
    'coroner': r'Coroner(?:\'?s)? name:\s*([^\n]+)',
    'area': r'Coroner(?:\'?s)? [Aa]rea:\s*([^\n]+)',
    'category': r'Category:\s*([^\n]+)'
}

def extract_key_sections(text: str) -> Dict:
    """Extract key sections with exact text matches"""
    sections = {
        'circumstances': None,
        'concerns': None,
        'actions': None,
        'response': None,
        'metadata': {}
    }
    
    # Extract sections with source tracking
    for section, pattern in SECTION_PATTERNS.items():
        match = re.search(pattern, text, re.I | re.S)
        if match:
            sections[section] = {
                'content': match.group(1).strip(),
                'source': match.group(0),
                'start': match.start(),
                'end': match.end()
            }
            
    # Extract metadata
    for key, pattern in METADATA_PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            sections['metadata'][key] = match.group(1).strip()
            
    return sections

def generate_summary(doc: Dict) -> DocumentSummary:
    """Generate fact-based summary with verification"""
    text = str(doc.get('Content', ''))
    sections = extract_key_sections(text)
    
    # Build extractive summary
    extractive_parts = []
    facts = []
    
    for section in ['circumstances', 'concerns', 'actions']:
        if sections[section]:
            content = sections[section]['content']
            extractive_parts.append(f"{section.upper()}:\n{content[:300]}...")
            facts.append({
                'type': section,
                'content': content,
                'source': sections[section]['source']
            })
    
    extractive = '\n\n'.join(extractive_parts)
    
    # Build abstractive summary from verified facts only
    abstract_parts = []
    meta = sections['metadata']
    
    if meta.get('ref') and meta.get('date'):
        abstract_parts.append(
            f"Prevention of Future Deaths report {meta['ref']} dated {meta['date']}"
        )
    
    if meta.get('deceased'):
        abstract_parts.append(f"regarding the death of {meta['deceased']}")
        
    for section in ['circumstances', 'concerns', 'actions']:
        if sections[section]:
            title = section.title()
            abstract_parts.append(
                f"\n{title}: {sections[section]['content'][:200]}..."
            )
    
    abstractive = ' '.join(abstract_parts)
    
    return DocumentSummary(
        title=doc.get('Title', ''),
        extractive=extractive,
        abstractive=abstractive,
        metadata=sections['metadata'],
        facts=facts,
        confidence=len(facts) / 3
    )

    def display_cluster_summaries(cluster_docs: List[dict]) -> None:
    """Display document summaries for cluster"""
    # Generate summaries
    summaries = []
    responses = []
    
    for doc in cluster_docs:
        # Check if response
        content = str(doc.get('Content', '')).lower()
        is_response = any(phrase in content for phrase in [
            'in response to',
            'responding to',
            'following the regulation 28'
        ])
        
        summary = generate_summary(doc)
        if is_response:
            responses.append(summary) 
        else:
            summaries.append(summary)
    
    # Display reports
    st.markdown("### Reports")
    for summary in summaries:
        with st.expander(f"{summary.title} (Confidence: {summary.confidence:.2%})"):
            tab1, tab2 = st.tabs(["Extractive Summary", "Abstractive Summary"])
            
            with tab1:
                st.markdown(summary.extractive)
                
            with tab2:
                st.markdown(summary.abstractive)
                
            if st.checkbox("Show Source Facts", key=f"facts_{summary.title}"):
                for fact in summary.facts:
                    st.markdown(f"**{fact['type'].title()}**")
                    st.markdown(f"Content: {fact['content']}")
                    st.markdown(f"Source: `{fact['source']}`")

    # Display responses
    if responses:
        st.markdown("### Responses")
        for response in responses:
            with st.expander(f"{response.title}"):
                st.markdown(response.extractive)
