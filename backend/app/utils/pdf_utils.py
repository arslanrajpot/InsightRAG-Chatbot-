from pypdf import PdfReader

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "".join(page.extract_text() for page in reader.pages if page.extract_text())