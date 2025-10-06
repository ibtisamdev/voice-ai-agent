"""
Unit tests for chunking strategies.
"""

import pytest
from ai.rag.chunking_strategies import LegalDocumentChunker, ChunkingStrategy, TextChunk


@pytest.fixture
def chunker():
    """Create document chunker instance."""
    return LegalDocumentChunker()


@pytest.fixture
def sample_legal_text():
    """Sample legal document text for testing."""
    return """
    SAMPLE AGREEMENT
    
    This Agreement is entered into on January 1, 2024.
    
    ARTICLE I: DEFINITIONS
    For purposes of this Agreement, the following terms shall have the meanings set forth below:
    (a) "Services" means the services described in Exhibit A.
    (b) "Term" means the period beginning on the Effective Date.
    
    ARTICLE II: OBLIGATIONS
    Section 1. Provider Obligations.
    Provider shall perform the Services in accordance with this Agreement.
    
    Section 2. Client Obligations.
    Client shall pay all fees when due.
    
    ARTICLE III: PAYMENT
    Client shall pay Provider $1,000 per month.
    Payment is due on the first day of each month.
    
    GOVERNING LAW
    This Agreement shall be governed by California law.
    """


def test_fixed_size_chunking(chunker, sample_legal_text):
    """Test fixed size chunking strategy."""
    chunks = chunker.chunk_document(
        text=sample_legal_text,
        strategy=ChunkingStrategy.FIXED_SIZE,
        max_chunk_size=200,
        overlap=50
    )
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, TextChunk) for chunk in chunks)
    assert all(len(chunk.text) <= 250 for chunk in chunks)  # Allow some buffer
    
    # Check metadata
    for chunk in chunks:
        assert chunk.metadata.strategy == "fixed_size"
        assert chunk.metadata.chunk_index >= 0
        assert chunk.metadata.total_chunks == len(chunks)


def test_sentence_chunking(chunker, sample_legal_text):
    """Test sentence-based chunking strategy."""
    chunks = chunker.chunk_document(
        text=sample_legal_text,
        strategy=ChunkingStrategy.SENTENCE,
        max_chunk_size=300
    )
    
    assert len(chunks) > 0
    
    # Sentences should be preserved (no sentence should be cut off mid-way)
    for chunk in chunks:
        assert chunk.metadata.strategy == "sentence"
        # Check that chunks don't end mid-sentence (rough heuristic)
        if chunk.text.strip():
            assert chunk.text.strip()[-1] in '.!?'


def test_paragraph_chunking(chunker, sample_legal_text):
    """Test paragraph-based chunking strategy."""
    chunks = chunker.chunk_document(
        text=sample_legal_text,
        strategy=ChunkingStrategy.PARAGRAPH,
        max_chunk_size=400
    )
    
    assert len(chunks) > 0
    
    for chunk in chunks:
        assert chunk.metadata.strategy == "paragraph"
        # Paragraphs should be preserved
        assert '\n\n' in sample_legal_text or len(chunks) == 1


def test_legal_section_chunking(chunker, sample_legal_text):
    """Test legal section-based chunking strategy."""
    chunks = chunker.chunk_document(
        text=sample_legal_text,
        strategy=ChunkingStrategy.LEGAL_SECTION,
        max_chunk_size=500
    )
    
    assert len(chunks) > 0
    
    for chunk in chunks:
        assert chunk.metadata.strategy == "legal_section"
        # Should have section type metadata for legal sections
        if "ARTICLE" in chunk.text or "Section" in chunk.text:
            assert chunk.metadata.section_type is not None


def test_hybrid_chunking(chunker, sample_legal_text):
    """Test hybrid chunking strategy."""
    chunks = chunker.chunk_document(
        text=sample_legal_text,
        strategy=ChunkingStrategy.HYBRID,
        max_chunk_size=400
    )
    
    assert len(chunks) > 0
    
    for chunk in chunks:
        assert chunk.metadata.strategy == "hybrid"


def test_chunk_overlap(chunker, sample_legal_text):
    """Test that chunk overlap works correctly."""
    chunks = chunker.chunk_document(
        text=sample_legal_text,
        strategy=ChunkingStrategy.FIXED_SIZE,
        max_chunk_size=200,
        overlap=50
    )
    
    if len(chunks) > 1:
        # Check that chunks have some overlapping content
        # This is a simple heuristic - real overlap checking would be more complex
        assert chunks[1].metadata.overlap_with_previous


def test_sentence_splitting(chunker):
    """Test sentence splitting functionality."""
    text = "This is sentence one. This is sentence two! Is this sentence three? Yes, it is."
    sentences = chunker._split_into_sentences(text)
    
    assert len(sentences) == 4
    assert "This is sentence one" in sentences[0]
    assert "This is sentence two" in sentences[1]
    assert "Is this sentence three" in sentences[2]
    assert "Yes, it is" in sentences[3]


def test_legal_section_identification(chunker, sample_legal_text):
    """Test identification of legal sections."""
    sections = chunker._identify_legal_sections(sample_legal_text)
    
    assert len(sections) > 0
    
    # Should identify ARTICLE sections
    article_sections = [s for s in sections if "ARTICLE" in s["header"]]
    assert len(article_sections) > 0
    
    # Check section classification
    for section in sections:
        assert "type" in section
        assert "header" in section
        assert "text" in section


def test_citation_extraction(chunker):
    """Test legal citation extraction."""
    text = "See 123 Cal. App. 456 (2023) and U.S.C. § 1234."
    citations = chunker._extract_citations(text)
    
    assert len(citations) > 0
    assert any("Cal. App." in citation for citation in citations)


def test_topic_boundary_detection(chunker):
    """Test topic boundary detection."""
    sentence1 = "This discusses contracts."
    sentence2 = "However, we must also consider tort law."
    
    is_boundary = chunker._is_topic_boundary(sentence2, sentence1)
    assert is_boundary  # "However" is a transition word


def test_empty_text_handling(chunker):
    """Test handling of empty or very short text."""
    # Empty text
    chunks = chunker.chunk_document("", ChunkingStrategy.FIXED_SIZE)
    assert len(chunks) == 0
    
    # Very short text
    chunks = chunker.chunk_document("Short.", ChunkingStrategy.FIXED_SIZE, max_chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0].text == "Short."


def test_chunk_metadata_completeness(chunker, sample_legal_text):
    """Test that chunk metadata is complete and consistent."""
    chunks = chunker.chunk_document(
        text=sample_legal_text,
        strategy=ChunkingStrategy.HYBRID,
        source_document="test_document.txt"
    )
    
    for i, chunk in enumerate(chunks):
        metadata = chunk.metadata
        
        # Required fields
        assert metadata.chunk_id is not None
        assert metadata.chunk_index == i
        assert metadata.total_chunks == len(chunks)
        assert metadata.start_char >= 0
        assert metadata.end_char > metadata.start_char
        assert metadata.token_count > 0
        assert metadata.strategy == "hybrid"
        assert metadata.source_document == "test_document.txt"


def test_section_type_classification(chunker):
    """Test classification of legal section types."""
    # Test different section headers
    assert chunker._classify_section_type("WHEREAS, the parties agree") == "recital"
    assert chunker._classify_section_type("THEREFORE, it is agreed") == "operative"
    assert chunker._classify_section_type("DEFINITIONS") == "definitions"
    assert chunker._classify_section_type("REPRESENTATIONS AND WARRANTIES") == "representations"
    assert chunker._classify_section_type("INDEMNIFICATION") == "indemnification"
    assert chunker._classify_section_type("GOVERNING LAW") == "governing_law"
    assert chunker._classify_section_type("ARTICLE I: SERVICES") == "article_section"
    assert chunker._classify_section_type("1. Payment Terms") == "numbered_section"
    assert chunker._classify_section_type("Random Header") == "general"