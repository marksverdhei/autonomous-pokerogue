import imagehash

# --- Example usage ---
# examples = {
#     "main menu": "main_menu.png",
#     "battle": "battle.png",
#     "party menu": "party_menu.png",
#     "computer": "computer.png",
# }


def classify_screen(img_path, examples):
    """
    Classify img_path as one of the example labels.
    
    examples: dict[label -> path_to_example_image]
    Returns: best_label, best_distance
    """
    img = Image.open(img_path).convert("RGB")
    h = imagehash.phash(img)  # 64-bit perceptual hash
    
    best_label, best_dist = None, 999
    for label, ex_path in examples.items():
        ex_hash = imagehash.phash(Image.open(ex_path).convert("RGB"))
        dist = h - ex_hash  # Hamming distance
        if dist < best_dist:
            best_label, best_dist = label, dist
    
    return best_label, best_dist

def identify_cursor():
    """
    Used for textual scaffolding
    """
    pass


def identify_state():
    """
    which menu
    """
    pass


def screenshot_ocr():
    pass
