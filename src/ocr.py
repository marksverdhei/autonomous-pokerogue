
ocr_model = PaddleOCR(
    ocr_version='PP-OCRv5',   # ensure v5 models
    lang='en',                # pick your language pack
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

def process_frame():
    pass
