from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from textwrap import wrap
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class Captioner:
    def __init__(self) -> None:
        self.checkpoint = "microsoft/git-base"
        self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint)
        self.processor = AutoProcessor.from_pretrained(self.checkpoint)

    def caption(self, path : str) -> str:
        try:
            image = Image.open("test3.jpg")

            device = "cuda" if torch.cuda.is_available() else "cpu"

            inputs = self.processor(images=image, return_tensors="pt").to(device)
            pixel_values = inputs.pixel_values

            generated_ids = self.model.generate(pixel_values=pixel_values, max_length=50)
            generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(generated_caption)
            return generated_caption
        except:
            print("Error has occured!")

class GUI:
    pass

if __name__ == "__main__":
    GUI()