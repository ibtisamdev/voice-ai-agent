"""
Unit tests for document processor.
"""

import pytest
import tempfile
import os
from pathlib import Path
import asyncio

from ai.rag.document_processor import LegalDocumentProcessor, DocumentProcessingError


@pytest.fixture
def document_processor():
    """Create document processor instance."""
    return LegalDocumentProcessor()


@pytest.fixture
def sample_text_file():
    """Create a sample text file for testing."""
    content = """
    SAMPLE LEGAL CONTRACT
    
    This Agreement is entered into on January 1, 2024, between Party A and Party B.
    
    WHEREAS, Party A desires to provide services;
    WHEREAS, Party B desires to receive services;
    
    NOW THEREFORE, the parties agree as follows:
    
    1. Services. Party A shall provide the following services...
    2. Payment. Party B shall pay $1,000 per month.
    3. Term. This agreement shall commence on January 1, 2024.
    
    GOVERNING LAW
    This agreement shall be governed by the laws of California.
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.mark.asyncio
async def test_process_text_file(document_processor, sample_text_file):
    """Test processing a text file."""
    result = await document_processor.process_file(sample_text_file)
    
    assert result["text"]
    assert result["metadata"]
    assert result["document_type"]
    assert result["source"] == sample_text_file
    
    # Check metadata
    metadata = result["metadata"]
    assert "character_count" in metadata
    assert "word_count" in metadata
    assert "document_hash" in metadata
    assert metadata["source"] == sample_text_file


@pytest.mark.asyncio 
async def test_document_type_detection(document_processor):
    """Test automatic document type detection."""
    # Contract text
    contract_text = "This Agreement is entered into whereas the parties agree"
    doc_type = document_processor._detect_document_type(contract_text)
    assert doc_type == "contract"
    
    # Brief text
    brief_text = "Plaintiff respectfully submits this motion to the court"
    doc_type = document_processor._detect_document_type(brief_text)
    assert doc_type == "brief"
    
    # Estate text
    estate_text = "This is the last will and testament of the testator"
    doc_type = document_processor._detect_document_type(estate_text)
    assert doc_type == "estate"


def test_text_cleaning(document_processor):
    """Test text cleaning functionality."""
    dirty_text = "This   has    extra     spaces\n\n\n\nand multiple newlines"
    clean_text = document_processor._clean_text(dirty_text)
    
    assert "   " not in clean_text  # No multiple spaces
    assert "\n\n\n" not in clean_text  # No triple newlines


def test_metadata_extraction(document_processor):
    """Test metadata extraction from text."""
    text = """
    January 1, 2024 contract between ABC Corp and XYZ Inc.
    Case citation: 123 Cal. App. 456 (2023)
    U.S.C. § 1234
    """
    
    metadata = document_processor._extract_text_metadata(text)
    
    assert "dates_found" in metadata
    assert "legal_citations" in metadata
    assert len(metadata["dates_found"]) > 0
    assert len(metadata["legal_citations"]) > 0


@pytest.mark.asyncio
async def test_invalid_file_path(document_processor):
    """Test processing non-existent file."""
    with pytest.raises(DocumentProcessingError):
        await document_processor.process_file("/nonexistent/file.txt")


@pytest.mark.asyncio
async def test_unsupported_file_type(document_processor):
    """Test processing unsupported file type."""
    with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
        temp_path = f.name
    
    try:
        with pytest.raises(DocumentProcessingError):
            await document_processor.process_file(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_token_count_estimation(document_processor):
    """Test token count estimation."""
    text = "This is a sample text for token counting."
    token_count = document_processor.get_token_count(text)
    
    assert isinstance(token_count, (int, float))
    assert token_count > 0


def test_processing_time_estimation(document_processor):
    """Test processing time estimation."""
    # Create a small test file
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(b"test content")
        temp_path = f.name
    
    try:
        estimated_time = document_processor.estimate_processing_time(temp_path)
        assert isinstance(estimated_time, float)
        assert estimated_time >= 0
    finally:
        os.unlink(temp_path)