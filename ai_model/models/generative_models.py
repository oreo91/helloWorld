#!/usr/bin/env python3
"""
Generative Models for Image Generation

This module contains various generative models that can create images:
- Generative Adversarial Network (GAN)
- Variational Autoencoder (VAE)
- Simple Pattern Generator
- Noise-to-Image Generator
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import random
import os
from typing import Tuple, List, Optional
import cv2

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Some advanced features will be limited.")

class SimpleImageGenerator:
    """
    A simple image generator that creates basic patterns and shapes.
    No deep learning required - perfect for getting started!
    """
    
    def __init__(self, image_size: Tuple[int, int] = (64, 64)):
        self.image_size = image_size
        self.width, self.height = image_size
    
    def generate_random_pattern(self, pattern_type: str = "circles") -> np.ndarray:
        """Generate a random pattern image."""
        # Create a blank image
        img = Image.new('RGB', self.image_size, color='black')
        draw = ImageDraw.Draw(img)
        
        if pattern_type == "circles":
            return self._generate_circles(draw, img)
        elif pattern_type == "rectangles":
            return self._generate_rectangles(draw, img)
        elif pattern_type == "lines":
            return self._generate_lines(draw, img)
        elif pattern_type == "noise":
            return self._generate_noise()
        elif pattern_type == "gradient":
            return self._generate_gradient()
        else:
            return self._generate_abstract(draw, img)
    
    def _generate_circles(self, draw, img) -> np.ndarray:
        """Generate random circles."""
        num_circles = random.randint(3, 8)
        for _ in range(num_circles):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            radius = random.randint(5, min(self.width, self.height) // 4)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
        return np.array(img)
    
    def _generate_rectangles(self, draw, img) -> np.ndarray:
        """Generate random rectangles."""
        num_rects = random.randint(2, 6)
        for _ in range(num_rects):
            x1, y1 = random.randint(0, self.width//2), random.randint(0, self.height//2)
            x2, y2 = random.randint(x1, self.width), random.randint(y1, self.height)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.rectangle([x1, y1, x2, y2], fill=color)
        return np.array(img)
    
    def _generate_lines(self, draw, img) -> np.ndarray:
        """Generate random lines."""
        num_lines = random.randint(5, 15)
        for _ in range(num_lines):
            x1, y1 = random.randint(0, self.width), random.randint(0, self.height)
            x2, y2 = random.randint(0, self.width), random.randint(0, self.height)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            width = random.randint(1, 5)
            draw.line([x1, y1, x2, y2], fill=color, width=width)
        return np.array(img)
    
    def _generate_noise(self) -> np.ndarray:
        """Generate random noise image."""
        return np.random.randint(0, 256, (self.height, self.width, 3), dtype=np.uint8)
    
    def _generate_gradient(self) -> np.ndarray:
        """Generate gradient image."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        start_color = [random.randint(0, 255) for _ in range(3)]
        end_color = [random.randint(0, 255) for _ in range(3)]
        
        for i in range(self.height):
            ratio = i / self.height
            color = [int(start_color[j] * (1 - ratio) + end_color[j] * ratio) for j in range(3)]
            img[i, :] = color
        return img
    
    def _generate_abstract(self, draw, img) -> np.ndarray:
        """Generate abstract art."""
        # Combine multiple elements
        self._generate_circles(draw, img)
        self._generate_lines(draw, img)
        return np.array(img)
    
    def generate_face_emoji(self) -> np.ndarray:
        """Generate a simple smiley face."""
        img = Image.new('RGB', self.image_size, color='yellow')
        draw = ImageDraw.Draw(img)
        
        # Face outline
        center_x, center_y = self.width // 2, self.height // 2
        face_radius = min(self.width, self.height) // 2 - 5
        draw.ellipse([center_x - face_radius, center_y - face_radius, 
                     center_x + face_radius, center_y + face_radius], 
                     fill='yellow', outline='black', width=2)
        
        # Eyes
        eye_radius = face_radius // 8
        left_eye_x = center_x - face_radius // 3
        right_eye_x = center_x + face_radius // 3
        eye_y = center_y - face_radius // 3
        
        draw.ellipse([left_eye_x - eye_radius, eye_y - eye_radius,
                     left_eye_x + eye_radius, eye_y + eye_radius], fill='black')
        draw.ellipse([right_eye_x - eye_radius, eye_y - eye_radius,
                     right_eye_x + eye_radius, eye_y + eye_radius], fill='black')
        
        # Smile
        mouth_y = center_y + face_radius // 4
        mouth_width = face_radius
        draw.arc([center_x - mouth_width//2, mouth_y - mouth_width//4,
                 center_x + mouth_width//2, mouth_y + mouth_width//4], 
                 start=0, end=180, fill='black', width=3)
        
        return np.array(img)


class NoiseToImageGenerator:
    """
    Generates images from random noise using mathematical transformations.
    This demonstrates how neural networks might transform noise into images.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (64, 64)):
        self.image_size = image_size
        self.width, self.height = image_size
    
    def generate_from_noise(self, noise_vector: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate an image from a noise vector."""
        if noise_vector is None:
            noise_vector = np.random.randn(100)  # 100-dimensional noise
        
        # Transform noise into image using mathematical functions
        img = np.zeros((self.height, self.width, 3))
        
        for i in range(self.height):
            for j in range(self.width):
                # Use noise vector to determine pixel values
                idx = ((i * self.width + j) * len(noise_vector)) // (self.height * self.width)
                idx = min(idx, len(noise_vector) - 1)
                
                # Apply transformations
                r = int(255 * (np.sin(noise_vector[idx] * 10) + 1) / 2)
                g = int(255 * (np.cos(noise_vector[idx] * 8) + 1) / 2)
                b = int(255 * (np.sin(noise_vector[idx] * 6 + np.pi/4) + 1) / 2)
                
                img[i, j] = [r, g, b]
        
        return img.astype(np.uint8)
    
    def generate_texture(self, texture_type: str = "wood") -> np.ndarray:
        """Generate texture-like images."""
        img = np.zeros((self.height, self.width, 3))
        
        if texture_type == "wood":
            return self._generate_wood_texture()
        elif texture_type == "marble":
            return self._generate_marble_texture()
        elif texture_type == "cloud":
            return self._generate_cloud_texture()
        else:
            return self._generate_fabric_texture()
    
    def _generate_wood_texture(self) -> np.ndarray:
        """Generate wood-like texture."""
        img = np.zeros((self.height, self.width, 3))
        base_color = [139, 69, 19]  # Brown
        
        for i in range(self.height):
            for j in range(self.width):
                # Create wood grain pattern
                grain = np.sin(j * 0.1) * 20 + np.random.normal(0, 10)
                color = [max(0, min(255, base_color[k] + grain)) for k in range(3)]
                img[i, j] = color
        
        return img.astype(np.uint8)
    
    def _generate_marble_texture(self) -> np.ndarray:
        """Generate marble-like texture."""
        img = np.zeros((self.height, self.width, 3))
        
        for i in range(self.height):
            for j in range(self.width):
                # Create marble veining
                value = np.sin(i * 0.1 + j * 0.05) * 100 + np.random.normal(200, 30)
                color = [max(0, min(255, value + np.random.normal(0, 20))) for _ in range(3)]
                img[i, j] = color
        
        return img.astype(np.uint8)
    
    def _generate_cloud_texture(self) -> np.ndarray:
        """Generate cloud-like texture."""
        img = np.zeros((self.height, self.width, 3))
        
        for i in range(self.height):
            for j in range(self.width):
                # Create cloud pattern
                value = 255 * (np.sin(i * 0.02) * np.cos(j * 0.02) + 1) / 2
                value += np.random.normal(0, 30)
                cloud_color = max(180, min(255, value))
                img[i, j] = [cloud_color, cloud_color + 10, 255]
        
        return img.astype(np.uint8)
    
    def _generate_fabric_texture(self) -> np.ndarray:
        """Generate fabric-like texture."""
        img = np.zeros((self.height, self.width, 3))
        base_color = [100, 100, 150]
        
        for i in range(self.height):
            for j in range(self.width):
                # Create weave pattern
                weave = np.sin(i * 0.5) * np.cos(j * 0.5) * 30
                color = [max(0, min(255, base_color[k] + weave + np.random.normal(0, 15))) 
                        for k in range(3)]
                img[i, j] = color
        
        return img.astype(np.uint8)


if TORCH_AVAILABLE:
    class SimpleGAN:
        """
        A simple Generative Adversarial Network for image generation.
        This is a basic implementation for educational purposes.
        """
        
        def __init__(self, latent_dim: int = 100, image_size: int = 28):
            self.latent_dim = latent_dim
            self.image_size = image_size
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Generator network
            self.generator = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, image_size * image_size),
                nn.Tanh()
            ).to(self.device)
            
            # Discriminator network
            self.discriminator = nn.Sequential(
                nn.Linear(image_size * image_size, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 1),
                nn.Sigmoid()
            ).to(self.device)
            
            # Optimizers
            self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002)
            self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002)
            self.criterion = nn.BCELoss()
        
        def generate_images(self, num_images: int = 1) -> np.ndarray:
            """Generate images using the trained generator."""
            self.generator.eval()
            with torch.no_grad():
                noise = torch.randn(num_images, self.latent_dim).to(self.device)
                generated = self.generator(noise)
                generated = generated.view(num_images, self.image_size, self.image_size)
                # Convert from [-1, 1] to [0, 255]
                generated = ((generated + 1) * 127.5).cpu().numpy().astype(np.uint8)
            return generated
        
        def train_step(self, real_images: torch.Tensor):
            """Perform one training step."""
            batch_size = real_images.size(0)
            real_images = real_images.view(batch_size, -1).to(self.device)
            
            # Train discriminator
            self.d_optimizer.zero_grad()
            
            # Real images
            real_labels = torch.ones(batch_size, 1).to(self.device)
            real_output = self.discriminator(real_images)
            d_loss_real = self.criterion(real_output, real_labels)
            
            # Fake images
            noise = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_images = self.generator(noise)
            fake_labels = torch.zeros(batch_size, 1).to(self.device)
            fake_output = self.discriminator(fake_images.detach())
            d_loss_fake = self.criterion(fake_output, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.d_optimizer.step()
            
            # Train generator
            self.g_optimizer.zero_grad()
            fake_output = self.discriminator(fake_images)
            g_loss = self.criterion(fake_output, real_labels)
            g_loss.backward()
            self.g_optimizer.step()
            
            return d_loss.item(), g_loss.item()


class ImageGenerationUtils:
    """Utility functions for image generation and visualization."""
    
    @staticmethod
    def save_image(image: np.ndarray, filename: str):
        """Save a numpy array as an image."""
        if len(image.shape) == 3:
            img = Image.fromarray(image.astype(np.uint8))
        else:
            img = Image.fromarray(image.astype(np.uint8), mode='L')
        img.save(filename)
        print(f"💾 Saved image: {filename}")
    
    @staticmethod
    def create_image_grid(images: List[np.ndarray], grid_size: Tuple[int, int] = None) -> np.ndarray:
        """Create a grid of images for display."""
        num_images = len(images)
        
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(num_images)))
            rows = int(np.ceil(num_images / cols))
        else:
            rows, cols = grid_size
        
        if len(images[0].shape) == 3:
            h, w, c = images[0].shape
            grid = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
        else:
            h, w = images[0].shape
            grid = np.zeros((rows * h, cols * w), dtype=np.uint8)
        
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            if len(img.shape) == 3:
                grid[row*h:(row+1)*h, col*w:(col+1)*w, :] = img
            else:
                grid[row*h:(row+1)*h, col*w:(col+1)*w] = img
        
        return grid
    
    @staticmethod
    def display_images(images: List[np.ndarray], titles: List[str] = None, figsize: Tuple[int, int] = (12, 8)):
        """Display multiple images in a grid."""
        num_images = len(images)
        cols = min(4, num_images)
        rows = int(np.ceil(num_images / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, img in enumerate(images):
            if len(img.shape) == 3:
                axes[i].imshow(img)
            else:
                axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
            if titles and i < len(titles):
                axes[i].set_title(titles[i])
        
        # Hide unused subplots
        for i in range(num_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig


def create_sample_images():
    """Create sample images using different generators."""
    print("🎨 Creating sample images...")
    
    # Simple pattern generator
    simple_gen = SimpleImageGenerator((128, 128))
    
    # Generate different types of images
    images = []
    titles = []
    
    # Generate patterns
    for pattern in ["circles", "rectangles", "lines", "gradient"]:
        img = simple_gen.generate_random_pattern(pattern)
        images.append(img)
        titles.append(f"Pattern: {pattern}")
    
    # Generate emoji
    emoji = simple_gen.generate_face_emoji()
    images.append(emoji)
    titles.append("Emoji Face")
    
    # Noise to image generator
    noise_gen = NoiseToImageGenerator((128, 128))
    
    # Generate from noise
    noise_img = noise_gen.generate_from_noise()
    images.append(noise_img)
    titles.append("From Noise")
    
    # Generate textures
    for texture in ["wood", "marble", "cloud"]:
        texture_img = noise_gen.generate_texture(texture)
        images.append(texture_img)
        titles.append(f"Texture: {texture}")
    
    return images, titles


if __name__ == "__main__":
    # Demo the image generation capabilities
    print("🧠 AI Image Generation Demo")
    print("=" * 40)
    
    # Create sample images
    images, titles = create_sample_images()
    
    # Display images
    utils = ImageGenerationUtils()
    fig = utils.display_images(images, titles, figsize=(15, 10))
    plt.savefig("generated_images_demo.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save individual images
    os.makedirs("generated_images", exist_ok=True)
    for i, (img, title) in enumerate(zip(images, titles)):
        filename = f"generated_images/{i+1:02d}_{title.replace(' ', '_').replace(':', '')}.png"
        utils.save_image(img, filename)
    
    print(f"\n✅ Generated {len(images)} images!")
    print("📁 Check the 'generated_images' folder for individual images")
    print("🖼️  Check 'generated_images_demo.png' for the complete grid")