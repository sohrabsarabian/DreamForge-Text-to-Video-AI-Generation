import torch
from CLIP import clip

class CLIPModel:
    def __init__(self, device):
        self.device = device
        self.model = self.load_model()

    def load_model(self):
        model, _ = clip.load('ViT-B/32', jit=False)
        model.eval()
        return model.to(self.device)

    def encode_text(self, text):
        return self.model.encode_text(clip.tokenize(text).to(self.device))

    def encode_image(self, image):
        return self.model.encode_image(image)