# src/jd_extractor.py
"""
Job Description Text Extraction Module
Supports: .txt, .pdf, .docx, .doc
"""
import os
import docx
import PyPDF2


def extract_jd_from_pdf(file_path):
    """Extract text from PDF job description"""
    try:
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        print(f"❌ PDF extraction error {file_path}: {e}")
        return ""


def extract_jd_from_docx(file_path):
    """Extract text from DOCX/DOC job description"""
    try:
        doc = docx.Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        return text.strip()
    except Exception as e:
        print(f"❌ DOCX extraction error {file_path}: {e}")
        return ""


def extract_jd_from_txt(file_path):
    """Extract text from TXT job description"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read().strip()
        except Exception as e:
            print(f"❌ TXT extraction error {file_path}: {e}")
            return ""
    except Exception as e:
        print(f"❌ TXT extraction error {file_path}: {e}")
        return ""


def extract_jd_text(file_path):
    """
    Main function to extract job description text from any supported format
    
    Args:
        file_path: Path to job description file (.txt, .pdf, .docx, .doc)
    
    Returns:
        str: Extracted text content
    """
    if not file_path or not os.path.exists(file_path):
        return ""
    
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        return extract_jd_from_pdf(file_path)
    elif ext in [".docx", ".doc"]:
        return extract_jd_from_docx(file_path)
    elif ext == ".txt":
        return extract_jd_from_txt(file_path)
    else:
        print(f"⚠️ Unsupported file format: {ext}")
        return ""


def extract_jd_from_bytes(file_bytes, filename):
    """
    Extract job description text from file bytes (for web uploads)
    
    Args:
        file_bytes: Binary file content
        filename: Original filename to determine format
    
    Returns:
        str: Extracted text content
    """
    import tempfile
    
    # Create temporary file
    ext = os.path.splitext(filename)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name
    
    try:
        text = extract_jd_text(tmp_path)
        return text
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)