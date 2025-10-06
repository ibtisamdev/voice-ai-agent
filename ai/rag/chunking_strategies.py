"""
Text chunking strategies for legal documents.
Provides various methods to split documents while preserving context and legal structure.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import tiktoken
from app.core.config import settings

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    LEGAL_SECTION = "legal_section"
    HYBRID = "hybrid"


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    chunk_id: str
    chunk_index: int
    total_chunks: int
    start_char: int
    end_char: int
    token_count: int
    strategy: str
    section_type: Optional[str] = None
    legal_citations: Optional[List[str]] = None
    source_document: Optional[str] = None
    overlap_with_previous: bool = False


@dataclass
class TextChunk:
    """A chunk of text with metadata."""
    text: str
    metadata: ChunkMetadata


class LegalDocumentChunker:
    """Chunker specialized for legal documents."""
    
    def __init__(self):
        self.max_chunk_size = settings.MAX_CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        
        # Legal section headers (regex patterns)
        self.section_patterns = [
            r'^\s*(?:ARTICLE|Article)\s+[IVX\d]+[:\.]',
            r'^\s*(?:SECTION|Section)\s+\d+(?:\.\d+)*[:\.]',
            r'^\s*(?:CLAUSE|Clause)\s+\d+[:\.]',
            r'^\s*\d+\.\s+[A-Z]',  # Numbered sections
            r'^\s*\([a-z]\)\s+',   # Lettered subsections
            r'^\s*\([ivx]+\)\s+',  # Roman numeral subsections
            r'^\s*(?:WHEREAS|THEREFORE|NOW THEREFORE)[,:]',
            r'^\s*(?:RECITALS|DEFINITIONS|TERMS AND CONDITIONS)',
            r'^\s*(?:REPRESENTATIONS AND WARRANTIES)',
            r'^\s*(?:INDEMNIFICATION|GOVERNING LAW|JURISDICTION)',
            r'^\s*(?:ENTIRE AGREEMENT|SEVERABILITY|AMENDMENT)'
        ]
        
        # Sentence boundary patterns
        self.sentence_endings = r'[.!?]+(?:\s+|$)'
        
        # Citation patterns
        self.citation_patterns = [
            r'\b\d+\s+[A-Z][a-z]+\.?\s+\d+\b',
            r'\b\d+\s+[A-Z]\.?\s*\d+d?\s+\d+\b',
            r'\b[A-Z]+\s+§\s*\d+(?:\.\d+)*\b'
        ]
    
    def chunk_document(
        self,
        text: str,
        strategy: ChunkingStrategy = ChunkingStrategy.HYBRID,
        max_chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        preserve_structure: bool = True,
        source_document: Optional[str] = None
    ) -> List[TextChunk]:
        """Chunk a document using the specified strategy."""
        max_chunk_size = max_chunk_size or self.max_chunk_size
        overlap = overlap or self.chunk_overlap
        
        try:
            if strategy == ChunkingStrategy.FIXED_SIZE:
                return self._chunk_fixed_size(text, max_chunk_size, overlap, source_document)
            elif strategy == ChunkingStrategy.SEMANTIC:
                return self._chunk_semantic(text, max_chunk_size, overlap, source_document)
            elif strategy == ChunkingStrategy.SENTENCE:
                return self._chunk_by_sentences(text, max_chunk_size, overlap, source_document)
            elif strategy == ChunkingStrategy.PARAGRAPH:
                return self._chunk_by_paragraphs(text, max_chunk_size, overlap, source_document)
            elif strategy == ChunkingStrategy.LEGAL_SECTION:
                return self._chunk_by_legal_sections(text, max_chunk_size, overlap, source_document)
            elif strategy == ChunkingStrategy.HYBRID:
                return self._chunk_hybrid(text, max_chunk_size, overlap, preserve_structure, source_document)
            else:
                raise ValueError(f"Unknown chunking strategy: {strategy}")
                
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            # Fallback to fixed size chunking
            return self._chunk_fixed_size(text, max_chunk_size, overlap, source_document)
    
    def _chunk_fixed_size(
        self,
        text: str,
        max_chunk_size: int,
        overlap: int,
        source_document: Optional[str]
    ) -> List[TextChunk]:
        """Chunk text into fixed-size pieces with overlap."""
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + max_chunk_size, len(text))
            
            # Try to end at word boundary
            if end < len(text):
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                metadata = ChunkMetadata(
                    chunk_id=f"chunk_{chunk_index}",
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will be updated later
                    start_char=start,
                    end_char=end,
                    token_count=self._estimate_token_count(chunk_text),
                    strategy="fixed_size",
                    source_document=source_document,
                    overlap_with_previous=start > 0 and overlap > 0
                )
                
                chunks.append(TextChunk(text=chunk_text, metadata=metadata))
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + 1, end - overlap)
        
        # Update total chunks count
        for chunk in chunks:
            chunk.metadata.total_chunks = len(chunks)
        
        return chunks
    
    def _chunk_by_sentences(
        self,
        text: str,
        max_chunk_size: int,
        overlap: int,
        source_document: Optional[str]
    ) -> List[TextChunk]:
        """Chunk text by sentences, keeping related sentences together."""
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        start_char = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed max size, finalize current chunk
            if current_chunk and current_size + sentence_size > max_chunk_size:
                chunk_text = ' '.join(current_chunk).strip()
                
                if chunk_text:
                    end_char = start_char + len(chunk_text)
                    metadata = ChunkMetadata(
                        chunk_id=f"sentence_chunk_{chunk_index}",
                        chunk_index=chunk_index,
                        total_chunks=0,
                        start_char=start_char,
                        end_char=end_char,
                        token_count=self._estimate_token_count(chunk_text),
                        strategy="sentence",
                        source_document=source_document
                    )
                    
                    chunks.append(TextChunk(text=chunk_text, metadata=metadata))
                    chunk_index += 1
                    
                    # Handle overlap
                    if overlap > 0:
                        overlap_sentences = self._get_overlap_sentences(current_chunk, overlap)
                        current_chunk = overlap_sentences
                        current_size = sum(len(s) for s in overlap_sentences)
                        start_char = end_char - current_size
                    else:
                        current_chunk = []
                        current_size = 0
                        start_char = end_char
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Handle remaining sentences
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if chunk_text:
                metadata = ChunkMetadata(
                    chunk_id=f"sentence_chunk_{chunk_index}",
                    chunk_index=chunk_index,
                    total_chunks=0,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    token_count=self._estimate_token_count(chunk_text),
                    strategy="sentence",
                    source_document=source_document
                )
                
                chunks.append(TextChunk(text=chunk_text, metadata=metadata))
        
        # Update total chunks count
        for chunk in chunks:
            chunk.metadata.total_chunks = len(chunks)
        
        return chunks
    
    def _chunk_by_paragraphs(
        self,
        text: str,
        max_chunk_size: int,
        overlap: int,
        source_document: Optional[str]
    ) -> List[TextChunk]:
        """Chunk text by paragraphs, combining small paragraphs."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        start_char = 0
        
        for paragraph in paragraphs:
            paragraph_size = len(paragraph)
            
            # If this paragraph alone exceeds max size, split it further
            if paragraph_size > max_chunk_size:
                # Finalize current chunk if it has content
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    metadata = ChunkMetadata(
                        chunk_id=f"para_chunk_{chunk_index}",
                        chunk_index=chunk_index,
                        total_chunks=0,
                        start_char=start_char,
                        end_char=start_char + len(chunk_text),
                        token_count=self._estimate_token_count(chunk_text),
                        strategy="paragraph",
                        source_document=source_document
                    )
                    
                    chunks.append(TextChunk(text=chunk_text, metadata=metadata))
                    chunk_index += 1
                    start_char += len(chunk_text) + 2  # +2 for \n\n
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph by sentences
                large_para_chunks = self._chunk_by_sentences(
                    paragraph, max_chunk_size, overlap, source_document
                )
                
                for large_chunk in large_para_chunks:
                    large_chunk.metadata.chunk_id = f"para_chunk_{chunk_index}"
                    large_chunk.metadata.chunk_index = chunk_index
                    large_chunk.metadata.strategy = "paragraph"
                    chunks.append(large_chunk)
                    chunk_index += 1
                
                continue
            
            # If adding this paragraph would exceed max size, finalize current chunk
            if current_chunk and current_size + paragraph_size + 2 > max_chunk_size:  # +2 for \n\n
                chunk_text = '\n\n'.join(current_chunk)
                end_char = start_char + len(chunk_text)
                
                metadata = ChunkMetadata(
                    chunk_id=f"para_chunk_{chunk_index}",
                    chunk_index=chunk_index,
                    total_chunks=0,
                    start_char=start_char,
                    end_char=end_char,
                    token_count=self._estimate_token_count(chunk_text),
                    strategy="paragraph",
                    source_document=source_document
                )
                
                chunks.append(TextChunk(text=chunk_text, metadata=metadata))
                chunk_index += 1
                
                # Handle overlap
                if overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-1]  # Keep last paragraph for overlap
                    current_chunk = [overlap_text] if len(overlap_text) < overlap else []
                    current_size = len(overlap_text) if current_chunk else 0
                    start_char = end_char - current_size - 2 if current_chunk else end_char
                else:
                    current_chunk = []
                    current_size = 0
                    start_char = end_char
            
            current_chunk.append(paragraph)
            current_size += paragraph_size + (2 if current_chunk else 0)  # +2 for \n\n separator
        
        # Handle remaining paragraphs
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            metadata = ChunkMetadata(
                chunk_id=f"para_chunk_{chunk_index}",
                chunk_index=chunk_index,
                total_chunks=0,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                token_count=self._estimate_token_count(chunk_text),
                strategy="paragraph",
                source_document=source_document
            )
            
            chunks.append(TextChunk(text=chunk_text, metadata=metadata))
        
        # Update total chunks count
        for chunk in chunks:
            chunk.metadata.total_chunks = len(chunks)
        
        return chunks
    
    def _chunk_by_legal_sections(
        self,
        text: str,
        max_chunk_size: int,
        overlap: int,
        source_document: Optional[str]
    ) -> List[TextChunk]:
        """Chunk text by legal sections and subsections."""
        sections = self._identify_legal_sections(text)
        
        if not sections:
            # Fallback to paragraph chunking if no legal sections found
            return self._chunk_by_paragraphs(text, max_chunk_size, overlap, source_document)
        
        chunks = []
        chunk_index = 0
        
        for section in sections:
            section_text = section['text']
            section_type = section['type']
            start_char = section['start']
            
            # If section is small enough, keep as one chunk
            if len(section_text) <= max_chunk_size:
                metadata = ChunkMetadata(
                    chunk_id=f"legal_section_{chunk_index}",
                    chunk_index=chunk_index,
                    total_chunks=0,
                    start_char=start_char,
                    end_char=start_char + len(section_text),
                    token_count=self._estimate_token_count(section_text),
                    strategy="legal_section",
                    section_type=section_type,
                    source_document=source_document
                )
                
                chunks.append(TextChunk(text=section_text, metadata=metadata))
                chunk_index += 1
            else:
                # Split large section further
                section_chunks = self._chunk_by_paragraphs(
                    section_text, max_chunk_size, overlap, source_document
                )
                
                for i, section_chunk in enumerate(section_chunks):
                    section_chunk.metadata.chunk_id = f"legal_section_{chunk_index}_{i}"
                    section_chunk.metadata.chunk_index = chunk_index
                    section_chunk.metadata.strategy = "legal_section"
                    section_chunk.metadata.section_type = section_type
                    chunks.append(section_chunk)
                    chunk_index += 1
        
        # Update total chunks count
        for chunk in chunks:
            chunk.metadata.total_chunks = len(chunks)
        
        return chunks
    
    def _chunk_hybrid(
        self,
        text: str,
        max_chunk_size: int,
        overlap: int,
        preserve_structure: bool,
        source_document: Optional[str]
    ) -> List[TextChunk]:
        """Hybrid chunking that combines multiple strategies."""
        # First, try to identify legal sections
        sections = self._identify_legal_sections(text)
        
        if preserve_structure and sections:
            # Use legal section chunking as primary strategy
            chunks = self._chunk_by_legal_sections(text, max_chunk_size, overlap, source_document)
        else:
            # Fall back to sentence-based chunking with paragraph awareness
            chunks = self._chunk_by_sentences(text, max_chunk_size, overlap, source_document)
        
        # Post-process to add citations and improve metadata
        for chunk in chunks:
            citations = self._extract_citations(chunk.text)
            if citations:
                chunk.metadata.legal_citations = citations
        
        # Update strategy to hybrid
        for chunk in chunks:
            chunk.metadata.strategy = "hybrid"
        
        return chunks
    
    def _chunk_semantic(
        self,
        text: str,
        max_chunk_size: int,
        overlap: int,
        source_document: Optional[str]
    ) -> List[TextChunk]:
        """Semantic chunking based on topic coherence (simplified version)."""
        # For now, use sentence-based chunking with topic-aware boundaries
        # In a more sophisticated version, this would use embeddings to detect topic shifts
        
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        start_char = 0
        
        for i, sentence in enumerate(sentences):
            sentence_size = len(sentence)
            
            # Simple topic boundary detection (look for transition words/phrases)
            is_topic_boundary = self._is_topic_boundary(sentence, sentences[i-1] if i > 0 else "")
            
            # If at topic boundary or size limit, finalize chunk
            if (current_chunk and 
                (current_size + sentence_size > max_chunk_size or 
                 (is_topic_boundary and current_size > max_chunk_size * 0.5))):
                
                chunk_text = ' '.join(current_chunk).strip()
                if chunk_text:
                    end_char = start_char + len(chunk_text)
                    metadata = ChunkMetadata(
                        chunk_id=f"semantic_chunk_{chunk_index}",
                        chunk_index=chunk_index,
                        total_chunks=0,
                        start_char=start_char,
                        end_char=end_char,
                        token_count=self._estimate_token_count(chunk_text),
                        strategy="semantic",
                        source_document=source_document
                    )
                    
                    chunks.append(TextChunk(text=chunk_text, metadata=metadata))
                    chunk_index += 1
                    
                    # Handle overlap
                    if overlap > 0:
                        overlap_sentences = self._get_overlap_sentences(current_chunk, overlap)
                        current_chunk = overlap_sentences
                        current_size = sum(len(s) for s in overlap_sentences)
                        start_char = end_char - current_size
                    else:
                        current_chunk = []
                        current_size = 0
                        start_char = end_char
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Handle remaining sentences
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if chunk_text:
                metadata = ChunkMetadata(
                    chunk_id=f"semantic_chunk_{chunk_index}",
                    chunk_index=chunk_index,
                    total_chunks=0,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    token_count=self._estimate_token_count(chunk_text),
                    strategy="semantic",
                    source_document=source_document
                )
                
                chunks.append(TextChunk(text=chunk_text, metadata=metadata))
        
        # Update total chunks count
        for chunk in chunks:
            chunk.metadata.total_chunks = len(chunks)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling legal document peculiarities."""
        # Handle abbreviations that shouldn't be sentence boundaries
        abbreviations = [
            'Inc.', 'Corp.', 'LLC', 'Ltd.', 'Co.', 'U.S.', 'U.S.C.',
            'C.F.R.', 'Fed.', 'Supp.', 'F.2d', 'F.3d', 'S.Ct.',
            'Cal.', 'N.Y.', 'Tex.', 'Fla.', 'Ill.', 'Pa.'
        ]
        
        # Temporarily replace abbreviations
        temp_text = text
        for i, abbr in enumerate(abbreviations):
            temp_text = temp_text.replace(abbr, f"__ABBR_{i}__")
        
        # Split on sentence boundaries
        sentences = re.split(self.sentence_endings, temp_text)
        
        # Restore abbreviations and clean up
        result = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Restore abbreviations
                for i, abbr in enumerate(abbreviations):
                    sentence = sentence.replace(f"__ABBR_{i}__", abbr)
                result.append(sentence)
        
        return result
    
    def _identify_legal_sections(self, text: str) -> List[Dict[str, Any]]:
        """Identify legal sections in the text."""
        sections = []
        lines = text.split('\n')
        current_section = None
        current_text = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                if current_text:
                    current_text.append('')
                continue
            
            # Check if line matches any section pattern
            section_type = None
            for pattern in self.section_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    section_type = self._classify_section_type(line)
                    break
            
            if section_type:
                # Save previous section
                if current_section:
                    current_section['text'] = '\n'.join(current_text).strip()
                    if current_section['text']:
                        sections.append(current_section)
                
                # Start new section
                start_char = text.find(line, sum(len(l) + 1 for l in lines[:i]))
                current_section = {
                    'type': section_type,
                    'header': line,
                    'start': start_char,
                    'line_number': i
                }
                current_text = [line]
            else:
                if current_text:
                    current_text.append(line)
        
        # Save last section
        if current_section and current_text:
            current_section['text'] = '\n'.join(current_text).strip()
            if current_section['text']:
                sections.append(current_section)
        
        return sections
    
    def _classify_section_type(self, header: str) -> str:
        """Classify the type of legal section."""
        header_lower = header.lower()
        
        if 'whereas' in header_lower:
            return 'recital'
        elif 'therefore' in header_lower:
            return 'operative'
        elif 'definition' in header_lower:
            return 'definitions'
        elif 'representation' in header_lower or 'warrant' in header_lower:
            return 'representations'
        elif 'indemnif' in header_lower:
            return 'indemnification'
        elif 'govern' in header_lower:
            return 'governing_law'
        elif re.match(r'^\s*(?:article|section)\s+', header_lower):
            return 'article_section'
        elif re.match(r'^\s*\d+\.', header):
            return 'numbered_section'
        else:
            return 'general'
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract legal citations from text."""
        citations = []
        for pattern in self.citation_patterns:
            citations.extend(re.findall(pattern, text))
        return list(set(citations))  # Remove duplicates
    
    def _is_topic_boundary(self, current_sentence: str, previous_sentence: str) -> bool:
        """Simple heuristic to detect topic boundaries."""
        transition_phrases = [
            'however', 'nevertheless', 'furthermore', 'moreover',
            'in addition', 'on the other hand', 'in contrast',
            'meanwhile', 'subsequently', 'in conclusion'
        ]
        
        current_lower = current_sentence.lower()
        return any(phrase in current_lower for phrase in transition_phrases)
    
    def _get_overlap_sentences(self, sentences: List[str], overlap_chars: int) -> List[str]:
        """Get sentences that fit within the overlap character limit."""
        overlap_sentences = []
        char_count = 0
        
        for sentence in reversed(sentences):
            if char_count + len(sentence) <= overlap_chars:
                overlap_sentences.insert(0, sentence)
                char_count += len(sentence)
            else:
                break
        
        return overlap_sentences
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for text."""
        try:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            return len(encoding.encode(text))
        except Exception:
            # Fallback to word count approximation
            return int(len(text.split()) * 1.3)


# Global chunker instance
document_chunker = LegalDocumentChunker()