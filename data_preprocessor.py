import pdfplumber
import streamlit as st
from pathlib import Path
import re
from typing import Optional, List

class DataPreprocessor:
    """Class to handle PDF processing and text cleaning for financial documents"""
    
    def __init__(self):
        self.supported_formats = ['.pdf']
        
    def clean_financial_text(self, text: str) -> str:
        """Clean financial document text by removing headers, footers, page numbers"""
        if not text:
            return ""
        
        # Remove specific TCS document headers and footers
        text = re.sub(r'This data can be easily copy pasted into a Microsoft Excel sheet PRINT', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Tata Consultancy Services Previous Years.*?�', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Source : Dion Global Solutions Limited\.?', '', text, flags=re.IGNORECASE)
        
        # Remove page numbers and markers
        text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
        text = re.sub(r'--- Page \d+ ---', '', text)
        
        # Preserve line structure for financial data
        # Keep existing line breaks and don't collapse all whitespace
        text = re.sub(r'[ \t]+', ' ', text)  # Only collapse spaces/tabs, keep newlines
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
        
        # Clean up lines but preserve structure
        lines = []
        for line in text.split('\n'):
            cleaned_line = line.strip()
            if cleaned_line:  # Keep non-empty lines
                lines.append(cleaned_line)
            elif lines and lines[-1]:  # Add empty line only if previous line wasn't empty
                lines.append('')
        
        return '\n'.join(lines).strip()
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text and tables from PDF with proper formatting"""
        try:
            text_content = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text_content.append(f"\n=== PAGE {page_num} ===\n")
                    
                    # Check if page has tables
                    tables = page.extract_tables()
                    
                    if tables:
                        # If tables exist, extract them with formatting
                        text_content.append(f"\n--- TABLES FROM PAGE {page_num} ---\n")
                        for table_num, table in enumerate(tables, 1):
                            text_content.append(f"\nTable {table_num}:\n")
                            text_content.append("-" * 50 + "\n")
                            
                            for row_num, row in enumerate(table):
                                if row:
                                    # Clean and format row
                                    cleaned_row = []
                                    for cell in row:
                                        cell_text = str(cell).strip() if cell else ""
                                        cleaned_row.append(cell_text)
                                    
                                    # Join cells with proper spacing for readability
                                    row_text = " | ".join(cleaned_row)
                                    text_content.append(row_text + "\n")
                                    
                                    # Add separator after header row
                                    if row_num == 0:
                                        text_content.append("-" * len(row_text) + "\n")
                            
                            text_content.append("\n")
                        
                        # Extract non-table text (filter out table content)
                        page_text = page.extract_text()
                        if page_text:
                            # Remove table-like content from regular text
                            non_table_text = self.filter_table_content(page_text)
                            if non_table_text.strip():
                                text_content.append(f"\n--- OTHER TEXT FROM PAGE {page_num} ---\n")
                                text_content.append(non_table_text)
                                text_content.append("\n")
                    else:
                        # No tables, extract regular text
                        page_text = page.extract_text()
                        if page_text:
                            text_content.append(page_text)
                            text_content.append("\n")
            
            return "".join(text_content)
        
        except Exception as e:
            st.error(f"Error extracting PDF: {str(e)}")
            return ""
    
    def filter_table_content(self, text: str) -> str:
        """Remove table-like content from regular text to avoid duplication"""
        if not text:
            return ""
        
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip lines that look like table data (multiple numbers/values separated by spaces)
            if line and not re.match(r'^[\d\s\.\,\|\-]+$', line):
                # Skip lines with too many numeric patterns (likely table rows)
                numeric_parts = re.findall(r'\d+', line)
                if len(numeric_parts) < 4:  # Keep lines with fewer than 4 numbers
                    filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def process_document(self, file_path: str, output_path: Optional[str] = None) -> str:
        """Process document and return clean text"""
        
        if not Path(file_path).exists():
            st.error(f"File not found: {file_path}")
            return ""
        
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension not in self.supported_formats:
            st.error(f"Unsupported file format: {file_extension}")
            return ""
        
        st.info(f"Processing {Path(file_path).name}...")
        
        # Extract raw text
        if file_extension == '.pdf':
            raw_text = self.extract_pdf_text(file_path)
        else:
            return ""
        
        if not raw_text:
            st.error("Failed to extract text")
            return ""
        
        # Clean the text
        cleaned_text = self.clean_financial_text(raw_text)
        
        # Save if output path provided
        if output_path:
            self.save_text(cleaned_text, output_path)
        
        st.success(f"Document processed successfully! Text length: {len(cleaned_text)} characters")
        return cleaned_text
    
    def save_text(self, text: str, output_path: str) -> bool:
        """Save text to file"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            st.success(f"Text saved to: {output_path}")
            return True
        except Exception as e:
            st.error(f"Failed to save: {str(e)}")
            return False
    
    def get_document_info(self, file_path: str) -> dict:
        """Get basic information about the document"""
        try:
            path = Path(file_path)
            info = {
                "filename": path.name,
                "file_size": path.stat().st_size,
                "format": path.suffix.lower()
            }
            
            if path.suffix.lower() == '.pdf':
                with pdfplumber.open(file_path) as pdf:
                    info["total_pages"] = len(pdf.pages)
            
            return info
        except Exception as e:
            return {"error": str(e)}