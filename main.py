from data_preprocessor import DataPreprocessor

class Main:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
    
    def convert_pdf_to_text(self):
        """Convert TCS financial statement PDF to clean text"""
        pdf_file = "data/financial-statements/tcs financial statement.pdf"
        output_file = "data/financial-statements/tcs_clean.txt"
        
        print("Starting PDF conversion...")
        
        # Get document info
        doc_info = self.preprocessor.get_document_info(pdf_file)
        print(f"Document info: {doc_info}")
        
        # Convert PDF to clean text
        text = self.preprocessor.process_document(pdf_file, output_file)
        
        if text:
            print("Conversion completed successfully!")
            print(f"Original file: {pdf_file}")
            print(f"Output file: {output_file}")
            print(f"Text length: {len(text)} characters")
            print(f"Word count: {len(text.split())} words")
            
            print("\n--- First 300 characters ---")
            print(text[:300])
            print("...")
            return text
        else:
            print("Conversion failed!")
            return None
    
    def run(self):
        """Main execution method"""
        print("Financial QA System - Data Preprocessing")
        print("=" * 50)
        
        # Convert PDF
        result = self.convert_pdf_to_text()
        
        if result:
            print("\nData preprocessing completed successfully!")
        else:
            print("\nData preprocessing failed!")

if __name__ == "__main__":
    main = Main()
    main.run()