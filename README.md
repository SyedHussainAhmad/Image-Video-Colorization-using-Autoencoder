# Image and Video Colorization using Autoencoder.

A comprehensive PyTorch implementation for automatic colorization of grayscale images and videos using deep learning. The project uses a U-Net architecture with ResNet-50 encoder and perceptual loss for high-quality colorization results.

## Features

- **Advanced U-Net Architecture**: ResNet-50 encoder with skip connections for better feature extraction
- **Perceptual Loss**: VGG19-based perceptual loss combined with pixel loss for realistic colors
- **Video Colorization**: Frame-by-frame video processing with batch optimization
- **Real-time Processing**: Live webcam colorization capability
- **Batch Processing**: Bulk image colorization for large datasets
- **Comprehensive Evaluation**: PSNR and SSIM metrics for quality assessment
- **Data Augmentation**: Advanced augmentation techniques for robust training
- **GPU Acceleration**: CUDA support for faster training and inference

## Model Architecture

### U-Net with ResNet-50 Encoder
- **Encoder**: Pre-trained ResNet-50 for feature extraction
- **Decoder**: Transposed convolutions with skip connections
- **Input**: Grayscale L channel (256×256)
- **Output**: AB color channels (256×256)
- **Color Space**: LAB color space for better color representation

### Loss Function
- **Perceptual Loss**: VGG19 feature-based loss
- **Pixel Loss**: MSE between predicted and ground truth
- **Combined**: `pixel_loss + 0.1 × perceptual_loss`

## Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended, 8GB+ VRAM)
- Google Colab environment (as configured) or local environment

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd colorization-project
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download COCO dataset** (for training):
```bash
mkdir -p data/coco/images
wget -O data/coco/val2017.zip http://images.cocodataset.org/zips/val2017.zip
unzip data/coco/val2017.zip -d data/coco/images/
```

## Usage

### 1. Training

Train the colorization model on COCO dataset:

```python
# Train the model
model, train_losses, val_losses = train_model(data_dir='data/coco/images/val2017')

# Plot training progress
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.show()
```

### 2. Single Image Colorization

```python
# Quick test on a single image
quick_test_colorization('best_colorization_model.pth', 'path/to/grayscale/image.jpg')
```

### 3. Video Colorization

```python
# Initialize video colorizer
colorizer = VideoColorizer('best_colorization_model.pth')

# Colorize video
colorizer.colorize_video('input_video.mp4', 'colorized_output.mp4', batch_size=4)
```

### 4. Batch Image Processing

```python
# Process entire folder of images
batch_colorize_images('input_folder/', 'output_folder/', 'best_colorization_model.pth')
```

### 5. Real-time Colorization

```python
# Real-time webcam colorization
real_time_colorization('best_colorization_model.pth', camera_id=0)
```

## Dataset

### COCO Dataset
- **Training Data**: COCO 2017 validation set (~5,000 images)
- **Format**: RGB images automatically converted to LAB
- **Preprocessing**: Resize, crop, normalize, augment
- **Split**: 80% training, 20% validation

### Custom Dataset
For custom datasets, organize images as:
```
data/
├── images/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Training Configuration

### Hyperparameters
- **Batch Size**: 8 (adjust based on GPU memory)
- **Learning Rate**: 
  - Encoder: 1e-5 (pre-trained layers)
  - Decoder: 1e-4 (new layers)
- **Epochs**: 100
- **Optimizer**: Adam with different LR for encoder/decoder
- **Scheduler**: ReduceLROnPlateau
- **Image Size**: 256×256

### Data Augmentation
```python
transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.RandomCrop((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor()
])
```

## Model Performance

### Evaluation Metrics
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index

### Expected Performance
- **PSNR**: ~25-30 dB (higher is better)
- **SSIM**: ~0.85-0.95 (higher is better)
- **Training Time**: ~10-15 hours on RTX 3080

## Key Components

### ColorizationDataset Class
- Custom PyTorch dataset for LAB color space
- Automatic RGB→LAB conversion
- Built-in augmentation support
- Train/validation splitting

### ColorizationModel Class
- U-Net architecture with ResNet-50 backbone
- Skip connections for detail preservation
- Batch normalization and ReLU activations
- Tanh output activation for AB channels

### PerceptualLoss Class
- VGG19-based perceptual loss
- LAB→RGB conversion for VGG input
- Combined pixel and perceptual loss
- Error handling for robust training

### VideoColorizer Class
- Frame-by-frame video processing
- Batch processing optimization
- Progress tracking with tqdm
- OpenCV integration

## Advanced Features

### 1. Model Saving/Loading
```python
# Save model
torch.save(model.state_dict(), 'colorization_model.pth')

# Load model
model.load_state_dict(torch.load('colorization_model.pth', map_location=device))
```

### 2. Google Drive Integration
```python
# Mount and save to Google Drive
from google.colab import drive
drive.mount('/content/drive')
torch.save(model.state_dict(), '/content/drive/MyDrive/colorization_model.pth')
```

### 3. Error Handling
- Robust image loading with fallbacks
- Exception handling in training loops
- Graceful degradation for corrupted data

## Performance Optimization

### GPU Optimization
- Use `pin_memory=True` in DataLoader
- Batch processing for videos
- CUDA memory management
- Mixed precision training (optional)

### Speed Improvements
- Resize images to 128×128 for real-time processing
- Batch multiple frames together
- Use smaller models for deployment

## Troubleshooting

### Common Issues

**CUDA Out of Memory**:
```python
# Reduce batch size
batch_size = 4  # Instead of 8

# Use gradient accumulation
accumulation_steps = 2
```

**Poor Colorization Quality**:
- Increase training epochs
- Adjust perceptual loss weight
- Use better data augmentation
- Check dataset quality

**Slow Video Processing**:
- Reduce video resolution
- Use smaller batch sizes
- Process on GPU if available

### Memory Management
```python
# Clear GPU cache
torch.cuda.empty_cache()

# Use smaller image sizes
dataset = ColorizationDataset(data_dir, size=128)  # Instead of 256
```

## Applications

- **Historical Photo Restoration**: Colorize old black and white photographs
- **Film Restoration**: Colorize classic movies and documentaries
- **Medical Imaging**: Enhance grayscale medical scans
- **Security Footage**: Improve visibility of surveillance videos
- **Art and Creative Projects**: Artistic colorization experiments

## Model Variants

### Lightweight Version
- Use ResNet-18 instead of ResNet-50
- Reduce decoder complexity
- Lower image resolution (128×128)

### High-Quality Version
- Use ResNet-101 or EfficientNet backbone
- Increase image resolution (512×512)
- Add attention mechanisms
- Use progressive training

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -am 'Add enhancement'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Create Pull Request

## Future Improvements

- [ ] Attention mechanisms for better detail preservation
- [ ] GAN-based adversarial training
- [ ] Progressive training for higher resolutions
- [ ] Mobile deployment optimization
- [ ] Web interface for easy usage
- [ ] Support for other color spaces (HSV, YUV)

## License

This project is for educational and research purposes. Ensure compliance with dataset licensing terms.

## Acknowledgments

- **PyTorch Team**: For the deep learning framework
- **COCO Dataset**: For training data
- **ResNet**: He et al. for the backbone architecture
- **U-Net**: Ronneberger et al. for the segmentation architecture
- **Perceptual Loss**: Johnson et al. for the loss function concept

## Citation

If you use this code in your research, please cite:
```bibtex
@misc{colorization2024,
  title={Deep Learning Image and Video Colorization},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo}}
}
```
---
## 👤 Author

**Syed Hussain Ahmad**
- GitHub: [@SyedHussainAhmad](https://github.com/SyedHussainAhmad)
- LinkedIn: [Syed Hussain Ahmad](https://www.linkedin.com/in/syedhussainahmad/)

---
⭐ Star this repository if you found it helpful!

💡 Have suggestions or found a bug? Please open an issue or submit a pull request.
