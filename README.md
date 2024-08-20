# DreamForge-Text-to-Video-AI-Generation

![DreamForge Demo](Movie.mp4)

## 🌟 Overview

DreamForge is a cutting-edge AI-powered tool that transforms text prompts into captivating video sequences. By leveraging the power of CLIP and Taming Transformer models, DreamForge generates visually stunning imagery that evolves based on your textual descriptions.

## 🚀 Features

- **Text-to-Image Generation**: Convert textual descriptions into vivid images.
- **Video Creation**: Seamlessly interpolate between generated images to create fluid video sequences.
- **Multi-Prompt Support**: Combine multiple text prompts to create complex, layered imagery.
- **Exclusion Prompts**: Specify elements to exclude from the generation process.
- **Fine-tuned Control**: Adjust various parameters to customize the generation process.

## 🛠 Installation

1. Clone the DreamForge repository:
   ```
   git clone https://github.com/sohrabsarabian/DreamForge-Text-to-Video-AI-Generation.git
   cd DreamForge
   ```

2. Clone the required model repositories:
   ```
   git clone https://github.com/openai/CLIP.git
   git clone https://github.com/CompVis/taming-transformers.git
   ```

3. Install the necessary Python libraries:
   ```
   pip install --no-deps ftfy regex tqdm
   pip install omegaconf==2.0.0 pytorch-lightning==1.0.8
   pip uninstall torchtext --yes
   pip install einops
   ```

4. Add the CLIP and Taming Transformer directories to your PYTHONPATH:
   ```
   export PYTHONPATH=$PYTHONPATH:$PWD/CLIP:$PWD/taming-transformers
   ```

Note: Ensure you have PyTorch installed. If not, install it according to the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/), selecting the appropriate version for your system and CUDA setup.

## 💡 Usage

Run the main script with your desired parameters:

```bash
python main.py \
  --learning_rate 0.5 \
  --batch_size 1 \
  --wd 0.1 \
  --noise_factor 0.22 \
  --total_iter 400 \
  --size1 450 \
  --size2 450 \
  --taming_config_path path/to/taming_config.yaml \
  --taming_checkpoint_path path/to/taming_checkpoint.ckpt \
  --include "a serene lake at sunset" "a bustling cityscape at night" "a lush forest in spring" \
  --exclude "people, vehicles" \
  --extras "vibrant colors" \
  --output_video dreamforge_output.mp4 \
  --w1 1.0 \
  --w2 1.0 \
  --alpha 1.0 \
  --beta 0.5 \
  --show_step 10 \
  --num_crops 32
```

Adjust the parameters as needed to fine-tune your generation process.

## 🎛 Parameters

- `learning_rate`: Learning rate for optimization (default: 0.5)
- `batch_size`: Batch size (default: 1)
- `wd`: Weight decay (default: 0.1)
- `noise_factor`: Noise factor for image augmentation (default: 0.22)
- `total_iter`: Total number of iterations (default: 400)
- `size1`, `size2`: Image dimensions (default: 450x450)
- `taming_config_path`: Path to the Taming Transformer configuration file
- `taming_checkpoint_path`: Path to the Taming Transformer checkpoint file
- `include`: List of text prompts to include in the generation
- `exclude`: Text prompt specifying elements to exclude
- `extras`: Additional context for the generation
- `output_video`: Name of the output video file
- `w1`, `w2`: Weights for include and extras prompts
- `alpha`, `beta`: Weights for main and penalize losses
- `show_step`: Interval for displaying generation progress
- `num_crops`: Number of crops for CLIP processing

## 🤝 Contributing

Contributions to DreamForge are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License

## 🙏 Acknowledgements

- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [Taming Transformers](https://github.com/CompVis/taming-transformers) by CompVis

## 📧 Contact

For any queries or suggestions, please open an issue in this repository.

Happy Forging! 🎨🎬✨
