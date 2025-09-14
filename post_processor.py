import cv2
import numpy as np
from PIL import Image
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import json
import os


# --- Stage 1: Image Pre-processing ---
def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    image_np = np.array(image)

    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return Image.fromarray(thresh)


# --- Stage 3: LLM Fusion ---
def load_llm_pipeline():
    model_id = "microsoft/Phi-3-mini-4k-instruct"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            load_in_4bit=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return pipeline("text-generation", model=model, tokenizer=tokenizer)

    except Exception as e:
        print(f"[WARN] Could not load {model_id}: {e}")
        print("[INFO] Falling back to a lightweight CPU model (facebook/bart-base).")

        fallback_id = "facebook/bart-base"
        model = AutoModelForCausalLM.from_pretrained(fallback_id)
        tokenizer = AutoTokenizer.from_pretrained(fallback_id)
        return pipeline("text-generation", model=model, tokenizer=tokenizer)


def fuse_and_clean_with_llm(llm_pipeline, donut_json: dict, paddle_text: list):
    if llm_pipeline is None:
        return {"error": "LLM Pipeline not loaded.", "donut_output": donut_json}

    donut_str = json.dumps(donut_json, indent=2, ensure_ascii=False)

    paddle_str = "\n".join(paddle_text)
    if len(paddle_str) > 1000:
        paddle_str = paddle_str[:1000] + "\n... [truncated] ..."

    prompt = (
        "You are an expert data validation analyst for a bank. "
        "Your task is to perfect a JSON object extracted from a document.\n\n"
        "Rules:\n"
        "1. Use the 'raw text' to correct typos and fill in missing values.\n"
        "2. Reformat all dates to YYYY-MM-DD.\n"
        "3. Clean monetary amounts (float/int only).\n"
        "4. Validate IBAN/bank account numbers.\n"
        "5. Output only valid JSON, no extra text.\n\n"
        f"<structured_json>\n{donut_str}\n</structured_json>\n\n"
        f"<raw_text>\n{paddle_str}\n</raw_text>\n\n"
        "Final JSON:\n"
    )

    outputs = llm_pipeline(
        prompt,
        max_new_tokens=512,
        do_sample=False,
        temperature=0.0,
    )
    cleaned_output = outputs[0]["generated_text"]

    try:
        json_start = cleaned_output.find('{')
        json_end = cleaned_output.rfind('}') + 1
        json_str = cleaned_output[json_start:json_end]
        return json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        return {"error": "Failed to parse LLM output", "raw_output": cleaned_output}
