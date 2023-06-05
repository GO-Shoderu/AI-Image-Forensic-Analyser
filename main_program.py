from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from textwrap import wrap
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import PySimpleGUI as sg

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
    # This function will take in a folder path, and return a list of strings, each string being a filename of that extension type
    # The returned list will be an exhaustive list of filenames of a certain filetype, to be iterated through when we're batch processing
    # a certain folder.
    def search_folder_for_extension(self, folder_path : str) -> list[str]:
        pass

    # This function will take the list of files generated by "search_folder_for_extension" and iterate through all files, 
    # performing a captioning process on each and returning a "dict" (dictionary) object with filename/caption pairs
    def batch_process(self, filenames : list[str]) -> dict:
        pass
    
    # This function will search self.calculated_captions for a specific search term, and returna  list of filenames 
    # where that search term has founda  match.
    def search(self, search_term : str) -> list[str]:
        if self.calculated_captions == None:
            print("Dictionary empty!")
        pass

    def __init__(self) -> None:
        self.calculated_captions : dict = None

        layout = [
            [sg.Multiline("Placeholder", expand_x=True, expand_y=True, disabled=True, key="placeholder_multiline")],
            [sg.FolderBrowse(key='folder_browser'), sg.Button("Search", key="open_search")]
        ]
        window = sg.Window("image-captioner", layout=layout, size=(1280, 800), resizable=True)

        while True:
            event, values = window.read()

            if event == "whatever":
                folder_path = values["folder_browser"]
                # process_folder()
                pass

            if event == sg.WIN_CLOSED:
                exit()

if __name__ == "__main__":
    GUI()