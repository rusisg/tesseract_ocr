import fitz  # PyMuPDF
from PIL import Image
import io

def convert_pdf_to_images(file_bytes: bytes) -> list[Image.Image]:
    images = []
    pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(dpi=200)
        
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        images.append(image)
        
    pdf_document.close()
    return images