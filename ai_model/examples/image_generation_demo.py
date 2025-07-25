#!/usr/bin/env python3
"""
Image Generation Demo

This script demonstrates the image generation capabilities of your AI model.
It shows how to create various types of images using different techniques.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.generative_models import (
    SimpleImageGenerator, 
    NoiseToImageGenerator, 
    ImageGenerationUtils
)

def demonstrate_simple_generation():
    """Demonstrate simple image generation techniques."""
    print("🎨 Simple Image Generation Demo")
    print("=" * 50)
    
    # Create generator
    generator = SimpleImageGenerator(image_size=(128, 128))
    utils = ImageGenerationUtils()
    
    # Generate different patterns
    patterns = ["circles", "rectangles", "lines", "gradient", "noise"]
    images = []
    titles = []
    
    print("Generating patterns...")
    for pattern in patterns:
        img = generator.generate_random_pattern(pattern)
        images.append(img)
        titles.append(f"{pattern.title()} Pattern")
        print(f"  ✓ {pattern}")
    
    # Generate emoji
    print("Generating emoji...")
    emoji = generator.generate_face_emoji()
    images.append(emoji)
    titles.append("Smiley Face")
    
    # Display all images
    fig = utils.display_images(images, titles, figsize=(15, 8))
    plt.savefig("simple_generation_demo.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return images, titles

def demonstrate_noise_to_image():
    """Demonstrate noise-to-image generation."""
    print("\n🌊 Noise-to-Image Generation Demo")
    print("=" * 50)
    
    generator = NoiseToImageGenerator(image_size=(128, 128))
    utils = ImageGenerationUtils()
    
    images = []
    titles = []
    
    # Generate from different noise vectors
    print("Generating from noise vectors...")
    for i in range(3):
        # Create different noise characteristics
        if i == 0:
            noise = np.random.randn(100)  # Standard Gaussian
        elif i == 1:
            noise = np.random.uniform(-2, 2, 100)  # Uniform distribution
        else:
            noise = np.random.exponential(1, 100)  # Exponential distribution
        
        img = generator.generate_from_noise(noise)
        images.append(img)
        titles.append(f"Noise Type {i+1}")
        print(f"  ✓ Noise variation {i+1}")
    
    # Generate textures
    print("Generating textures...")
    textures = ["wood", "marble", "cloud", "fabric"]
    for texture in textures:
        img = generator.generate_texture(texture)
        images.append(img)
        titles.append(f"{texture.title()} Texture")
        print(f"  ✓ {texture}")
    
    # Display results
    fig = utils.display_images(images, titles, figsize=(15, 10))
    plt.savefig("noise_to_image_demo.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return images, titles

def demonstrate_artistic_generation():
    """Demonstrate artistic image generation."""
    print("\n🎭 Artistic Image Generation Demo")
    print("=" * 50)
    
    generator = SimpleImageGenerator(image_size=(256, 256))
    utils = ImageGenerationUtils()
    
    images = []
    titles = []
    
    print("Creating artistic images...")
    
    # Generate abstract art
    for i in range(4):
        # Set random seed for reproducible "art"
        np.random.seed(42 + i)
        img = generator.generate_random_pattern("abstract")
        images.append(img)
        titles.append(f"Abstract Art {i+1}")
        print(f"  ✓ Abstract artwork {i+1}")
    
    # Generate variations of emoji
    for mood in ["happy", "sad", "surprised"]:
        # For this demo, we'll just generate the same emoji
        # In a real implementation, you'd modify the generation parameters
        img = generator.generate_face_emoji()
        images.append(img)
        titles.append(f"Emoji ({mood})")
        print(f"  ✓ {mood} emoji")
    
    # Display artistic results
    fig = utils.display_images(images, titles, figsize=(15, 10))
    plt.savefig("artistic_generation_demo.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return images, titles

def demonstrate_image_variations():
    """Demonstrate creating variations of the same image."""
    print("\n🔄 Image Variations Demo")
    print("=" * 50)
    
    generator = SimpleImageGenerator(image_size=(128, 128))
    noise_gen = NoiseToImageGenerator(image_size=(128, 128))
    utils = ImageGenerationUtils()
    
    images = []
    titles = []
    
    print("Creating variations...")
    
    # Create multiple variations of circles
    for i in range(6):
        np.random.seed(i * 10)  # Different seeds for variations
        img = generator.generate_random_pattern("circles")
        images.append(img)
        titles.append(f"Circles Var {i+1}")
    
    # Create variations from similar noise
    base_noise = np.random.randn(100)
    for i in range(3):
        # Add small variations to the base noise
        varied_noise = base_noise + np.random.randn(100) * 0.1
        img = noise_gen.generate_from_noise(varied_noise)
        images.append(img)
        titles.append(f"Noise Var {i+1}")
    
    print(f"  ✓ Created {len(images)} variations")
    
    # Display variations
    fig = utils.display_images(images, titles, figsize=(15, 8))
    plt.savefig("variations_demo.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return images, titles

def save_all_generated_images(all_images, all_titles):
    """Save all generated images to files."""
    print("\n💾 Saving Generated Images")
    print("=" * 50)
    
    # Create output directory
    output_dir = "generated_images"
    os.makedirs(output_dir, exist_ok=True)
    
    utils = ImageGenerationUtils()
    
    # Save all images
    for category, (images, titles) in all_images.items():
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        for i, (img, title) in enumerate(zip(images, titles)):
            filename = f"{category_dir}/{i+1:02d}_{title.replace(' ', '_').replace('(', '').replace(')', '').replace(':', '')}.png"
            utils.save_image(img, filename)
    
    print(f"📁 All images saved to '{output_dir}' directory")

def create_summary_grid():
    """Create a summary grid showing all generation capabilities."""
    print("\n📊 Creating Summary Grid")
    print("=" * 50)
    
    # Generate one example of each type
    simple_gen = SimpleImageGenerator((64, 64))
    noise_gen = NoiseToImageGenerator((64, 64))
    utils = ImageGenerationUtils()
    
    examples = []
    labels = []
    
    # One of each type
    types = [
        ("circles", "Circles"),
        ("rectangles", "Rectangles"), 
        ("lines", "Lines"),
        ("gradient", "Gradient"),
        ("noise", "Random Noise")
    ]
    
    for pattern_type, label in types:
        img = simple_gen.generate_random_pattern(pattern_type)
        examples.append(img)
        labels.append(label)
    
    # Add emoji
    emoji = simple_gen.generate_face_emoji()
    examples.append(emoji)
    labels.append("Emoji")
    
    # Add noise-generated image
    noise_img = noise_gen.generate_from_noise()
    examples.append(noise_img)
    labels.append("From Noise")
    
    # Add texture
    texture = noise_gen.generate_texture("wood")
    examples.append(texture)
    labels.append("Wood Texture")
    
    # Create and save grid
    fig = utils.display_images(examples, labels, figsize=(12, 8))
    plt.suptitle("AI Image Generation Capabilities Summary", fontsize=16, y=0.98)
    plt.savefig("ai_image_generation_summary.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Summary grid saved as 'ai_image_generation_summary.png'")

def main():
    """Main demo function."""
    print("🧠 AI Image Generation Comprehensive Demo")
    print("=" * 60)
    print("This demo showcases all image generation capabilities of your AI model!")
    print()
    
    all_images = {}
    
    # Run all demonstrations
    try:
        # Simple generation
        images1, titles1 = demonstrate_simple_generation()
        all_images["simple"] = (images1, titles1)
        
        # Noise to image
        images2, titles2 = demonstrate_noise_to_image()
        all_images["noise_to_image"] = (images2, titles2)
        
        # Artistic generation
        images3, titles3 = demonstrate_artistic_generation()
        all_images["artistic"] = (images3, titles3)
        
        # Variations
        images4, titles4 = demonstrate_image_variations()
        all_images["variations"] = (images4, titles4)
        
        # Save all images
        save_all_generated_images(all_images, None)
        
        # Create summary
        create_summary_grid()
        
        # Final summary
        total_images = sum(len(images) for images, _ in all_images.values())
        print(f"\n🎉 Demo Complete!")
        print(f"✅ Generated {total_images} unique images")
        print(f"📁 Check the 'generated_images' folder for all outputs")
        print(f"🖼️  Check the summary files for quick overviews")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install numpy matplotlib pillow")

if __name__ == "__main__":
    main()