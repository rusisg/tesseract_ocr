from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import cv2

from paddleocr import PaddleOCR

def load_paddle_model():
    ocr_ru = PaddleOCR(use_angle_cls=True, lang="ru")  # кириллица (рус + казахский)
    ocr_en = PaddleOCR(use_angle_cls=True, lang="en")  # латиница (английский)
    return {"ru": ocr_ru, "en": ocr_en}

def extract_full_text(image, ocr_models):
    all_texts = []

    for lang, ocr in ocr_models.items():
        results = ocr.ocr(image)
        for res in results:
            for line in res:
                text = line[1][0]
                all_texts.append(text)

    # убираем дубликаты и пустые строки
    unique_texts = list({t.strip() for t in all_texts if t.strip()})
    return unique_texts
