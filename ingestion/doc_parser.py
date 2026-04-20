"""
Document Parser Module for Industrial-Grade Document Ingestion

This module implements a robust document parsing pipeline capable of handling:
1. Complex PDFs with nested tables, multi-column layouts, and mixed content
2. Web pages with dynamic content and JavaScript rendering
3. Fallback mechanisms for parsing failures

Design Philosophy:
- Docling for complex PDFs (IBM's state-of-the-art document understanding)
- Crawl4AI for web scraping (handles dynamic content better than BeautifulSoup)
- Graceful degradation with fallback strategies
- Unified output format (Markdown) for downstream processing

Why Docling?
- Preserves document structure (tables, headers, lists)
- Handles multi-column layouts correctly
- Extracts text with positional information
- Better OCR integration than PyPDF
- Open-source and actively maintained by IBM
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from pypdf import PdfReader
import pytesseract
from pdf2image import convert_from_path
from crawl4ai import AsyncWebCrawler
import asyncio

from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentParser:
    """
    Industrial-grade document parser with fallback strategies.
    
    Design Decisions:
    - Primary: Docling for complex PDFs (tables, multi-column)
    - Fallback 1: PyPDF for simple text-only PDFs
    - Fallback 2: Tesseract OCR for image-based PDFs
    - Web: Crawl4AI for dynamic content rendering
    """
    
    def __init__(
        self,
        enable_ocr: bool = True,
        ocr_lang: str = "chi_sim+eng",
    ):
        """
        Initialize the document parser.
        
        Args:
            enable_ocr: Whether to enable OCR fallback for image-based PDFs
            ocr_lang: Tesseract OCR language (default: Chinese + English)
        """
        self.enable_ocr = enable_ocr
        self.ocr_lang = ocr_lang
        
        # Initialize Docling converter
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = enable_ocr
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        
        self.docling_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: pipeline_options,
            }
        )
    
    def parse_pdf(
        self,
        file_path: str,
        use_fallback: bool = True,
    ) -> str:
        """
        Parse PDF document to Markdown format.
        
        Parsing Strategy:
        1. Try Docling (best for complex layouts, tables, multi-column)
        2. Fallback to PyPDF (fast for simple text-only PDFs)
        3. Fallback to Tesseract OCR (for image-based/scanned PDFs)
        
        Args:
            file_path: Path to PDF file
            use_fallback: Whether to use fallback strategies on failure
        
        Returns:
            Markdown formatted text
        
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If all parsing methods fail
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected PDF file, got: {file_path.suffix}")
        
        logger.info(f"Parsing PDF: {file_path}")
        
        # Strategy 1: Try Docling (primary)
        try:
            result = self._parse_with_docling(file_path)
            logger.info("Successfully parsed with Docling")
            return result
        except Exception as e:
            logger.warning(f"Docling parsing failed: {e}")
            
            if not use_fallback:
                raise
        
        # Strategy 2: Fallback to PyPDF (for simple text-only PDFs)
        try:
            result = self._parse_with_pypdf(file_path)
            logger.info("Successfully parsed with PyPDF fallback")
            return result
        except Exception as e:
            logger.warning(f"PyPDF parsing failed: {e}")
        
        # Strategy 3: Fallback to Tesseract OCR (for image-based PDFs)
        if self.enable_ocr:
            try:
                result = self._parse_with_ocr(file_path)
                logger.info("Successfully parsed with OCR fallback")
                return result
            except Exception as e:
                logger.warning(f"OCR parsing failed: {e}")
        
        raise ValueError(f"All parsing methods failed for: {file_path}")
    
    def _parse_with_docling(self, file_path: Path) -> str:
        """
        Parse PDF using Docling (IBM's document understanding library).
        
        Advantages of Docling:
        - Preserves document structure (tables, headers, lists)
        - Handles multi-column layouts correctly
        - Extracts text with positional information
        - Better table extraction than PyPDF
        - Supports natively embedded OCR
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            Markdown formatted text
        """
        result = self.docling_converter.convert(str(file_path))
        return result.document.export_to_markdown()
    
    def _parse_with_pypdf(self, file_path: Path) -> str:
        """
        Parse PDF using PyPDF (fallback for simple text-only PDFs).
        
        Use Cases:
        - Fast extraction for simple, text-only PDFs
        - When Docling fails due to compatibility issues
        - Low-resource environments
        
        Limitations:
        - Cannot handle multi-column layouts
        - Poor table extraction
        - No OCR support
        - Loses formatting information
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            Plain text (no Markdown structure)
        """
        reader = PdfReader(str(file_path))
        text = ""
        
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
        
        return text.strip()
    
    def _parse_with_ocr(self, file_path: Path) -> str:
        """
        Parse image-based PDF using Tesseract OCR.
        
        Use Cases:
        - Scanned documents
        - Image-only PDFs
        - PDFs with embedded images containing text
        
        Limitations:
        - Slow (requires image conversion)
        - Accuracy depends on image quality
        - Requires Tesseract binary installation
        - Loses layout structure
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            OCR extracted text
        """
        # Convert PDF to images
        images = convert_from_path(str(file_path), dpi=300)
        
        text = ""
        for i, image in enumerate(images):
            # Perform OCR on each image
            page_text = pytesseract.image_to_string(
                image,
                lang=self.ocr_lang,
                config='--psm 6'  # Assume uniform block of text
            )
            text += f"--- Page {i + 1} ---\n{page_text}\n\n"
        
        return text.strip()
    
    async def parse_webpage(
        self,
        url: str,
        wait_for_selector: Optional[str] = None,
    ) -> str:
        """
        Parse webpage to Markdown format using Crawl4AI.
        
        Why Crawl4AI instead of BeautifulSoup/requests?
        - Handles dynamic JavaScript content (SPA, React, Vue)
        - Executes JavaScript before extraction
        - Better handling of lazy-loaded content
        - Extracts structured data (meta, links, images)
        - Built-in content cleaning and deduplication
        
        Args:
            url: Webpage URL to crawl
            wait_for_selector: CSS selector to wait for before extraction
        
        Returns:
            Markdown formatted content
        """
        logger.info(f"Crawling webpage: {url}")
        
        async with AsyncWebCrawler(verbose=True) as crawler:
            result = await crawler.arun(
                url=url,
                wait_for=wait_for_selector,
                bypass_cache=True,
                process_iframes=True,
            )
            
            if not result.success:
                raise ValueError(f"Failed to crawl {url}: {result.error_message}")
            
            # Crawl4AI returns cleaned markdown by default
            markdown_content = result.markdown
            
            return markdown_content
    
    def parse_file(self, file_path: str) -> Document:
        """
        Parse any supported file format to a Document object.
        
        Supported formats:
        - PDF (with fallback strategies)
        - Future: DOCX, TXT, MD (extensible)
        
        Args:
            file_path: Path to file
        
        Returns:
            Document object with parsed content and metadata
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == ".pdf":
            content = self.parse_pdf(str(file_path), use_fallback=True)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Create Document with metadata
        doc = Document(
            page_content=content,
            metadata={
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix[1:].lower(),
            }
        )
        
        return doc


# Singleton instance
_parser_instance: Optional[DocumentParser] = None


def get_parser() -> DocumentParser:
    """
    Get or create a singleton document parser instance.
    
    Returns:
        Shared DocumentParser instance
    """
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = DocumentParser()
    return _parser_instance
