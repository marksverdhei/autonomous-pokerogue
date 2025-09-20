import os
from transformers import pipeline

from transformers.image_utils import load_image

ckpt = "google/siglip2-base-patch16-224"
image_classifier = pipeline(model=ckpt, task="zero-shot-image-classification")

files = os.listdir("/home/me/Pictures/pokerogue/")
for file in files:
    image = load_image("/home/me/Pictures/pokerogue/" + file)
    candidate_labels = [
        "In a pokemon battle",
        "the pokemon summary screen",
        "your pokeon has fainted, select a new pokemon",
        "In the pokemon PC, assemble your team."
    ]
    outputs = image_classifier(image, candidate_labels)
    print(outputs)

