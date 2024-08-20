import torch
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel

class TamingModel:
    def __init__(self, config_path, checkpoint_path, device):
        self.device = device
        self.config = self.load_config(config_path)
        self.model = self.load_model(checkpoint_path)

    @staticmethod
    def load_config(config_path):
        return OmegaConf.load(config_path)

    def load_model(self, checkpoint_path):
        model = VQModel(**self.config.model.params)
        state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        return model.eval().to(self.device)

    def generate(self, z):
        return self.model.decode(z)