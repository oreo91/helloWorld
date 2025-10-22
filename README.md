# 🏗️ 3D Model Builder from Images

A comprehensive Python framework for creating 3D models from images using multiple computer vision and machine learning approaches.

## ✨ Features

### 🔧 Multiple Reconstruction Methods
- **Traditional Photogrammetry**: SIFT feature detection, matching, and triangulation
- **Neural Depth Estimation**: AI-powered monocular depth estimation using transformers
- **Multi-View Stereo**: Advanced stereo vision techniques
- **Single Image to 3D**: Convert individual images to 3D point clouds

### 🎯 Mesh Generation Algorithms
- **Poisson Surface Reconstruction**: Smooth, watertight surfaces
- **Alpha Shapes**: Detailed surface reconstruction preserving fine features
- **Ball Pivoting**: Robust reconstruction for noisy data

### 💻 Multiple Interfaces
- **Web Interface**: User-friendly Gradio-based web app
- **Command Line**: Powerful CLI for batch processing
- **Python API**: Full programmatic access

### 📁 Output Formats
- **PLY**: Point clouds and meshes
- **OBJ**: Wavefront 3D model format
- **STL**: Ready for 3D printing
- **JSON**: Metadata and parameters

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd 3d-model-builder

# Install dependencies
pip install -r requirements.txt
```

### Web Interface

Launch the web interface for easy interaction:

```bash
python web_interface.py
```

Then open your browser to `http://localhost:7860`

### Command Line Usage

```bash
# Basic usage with image directory
python cli_interface.py --input-dir ./images --output-dir ./output

# Specify individual images
python cli_interface.py --images img1.jpg img2.jpg img3.jpg --output-dir ./output

# Use different mesh reconstruction method
python cli_interface.py --input-dir ./images --mesh-method alpha_shape --visualize

# Generate only point cloud
python cli_interface.py --input-dir ./images --no-mesh --visualize
```

### Python API

```python
from model_builder_3d import Model3DBuilder

# Initialize builder
builder = Model3DBuilder(output_dir="output_3d")

# Reconstruct from images
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
success = builder.reconstruct_from_images(image_paths, mesh_method="poisson")

if success:
    # Save results
    output_files = builder.save_results("my_model")
    
    # Visualize
    builder.visualize()
```

## 📖 Detailed Usage

### Traditional Photogrammetry

Best for: Objects with rich texture and multiple overlapping views

```python
from model_builder_3d import Model3DBuilder

builder = Model3DBuilder()

# Load and process images
image_paths = ["view1.jpg", "view2.jpg", "view3.jpg"]
builder.reconstruct_from_images(image_paths)

# Access results
point_cloud = builder.point_cloud  # Open3D point cloud
mesh = builder.mesh                # Open3D triangle mesh
```

### Neural Depth Estimation

Best for: Single images or when traditional methods fail

```python
from neural_reconstruction import NeuralReconstruction

neural_recon = NeuralReconstruction()

# Single image to 3D
image = cv2.imread("photo.jpg")
point_cloud, depth_map = neural_recon.single_image_to_3d(image)

# Multi-view neural reconstruction
images = [cv2.imread(f"img{i}.jpg") for i in range(1, 4)]
point_cloud = neural_recon.multi_view_reconstruction(images)
```

### Camera Calibration

For better results, provide camera calibration images:

```python
# With checkerboard calibration
calibration_images = ["calib1.jpg", "calib2.jpg", "calib3.jpg"]
builder.reconstruct_from_images(
    image_paths=image_paths,
    calibration_images=calibration_images
)
```

## 🎛️ Configuration Options

### Mesh Reconstruction Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `poisson` | Smooth, watertight surfaces | Clean data, 3D printing |
| `alpha_shape` | Detailed surface preservation | Complex geometries |
| `ball_pivoting` | Robust to noise | Noisy point clouds |

### Feature Detection Parameters

```python
# Customize SIFT parameters
builder.detector = cv2.SIFT_create(
    nfeatures=10000,      # Maximum features
    contrastThreshold=0.04, # Filter weak features
    edgeThreshold=10       # Filter edge responses
)
```

### Neural Model Options

```python
# Use different device
neural_recon = NeuralReconstruction(device="cuda")  # or "cpu"

# Configure depth estimation
depth_map = neural_recon.estimate_depth(image)
```

## 📸 Image Guidelines

### For Best Results:

1. **Multiple Viewpoints**: Take 5-10 images around the object
2. **Good Overlap**: Ensure 60-80% overlap between consecutive images
3. **Consistent Lighting**: Avoid shadows and reflections
4. **Sharp Focus**: Use fast shutter speeds, avoid motion blur
5. **Plain Background**: Reduces noise in feature detection
6. **Stable Object**: Keep the object stationary between shots

### Supported Formats:
- JPEG, PNG, BMP, TIFF
- Recommended: High resolution (1080p or higher)

## 🧪 Demo and Examples

Run the comprehensive demo:

```bash
python demo.py
```

This will:
1. Create sample images for testing
2. Demonstrate traditional reconstruction
3. Show neural depth estimation
4. Compare different mesh methods
5. Generate visualizations

## 🔧 API Reference

### Model3DBuilder Class

#### Methods:

```python
# Core functionality
load_images(image_paths: List[str]) -> bool
calibrate_camera(calibration_images: List[str]) -> bool
extract_features() -> bool
build_point_cloud() -> bool
create_mesh(method: str) -> bool
save_results(filename_prefix: str) -> Dict[str, str]
visualize(show_point_cloud: bool, show_mesh: bool)

# Complete pipeline
reconstruct_from_images(
    image_paths: List[str],
    calibration_images: Optional[List[str]] = None,
    mesh_method: str = "poisson"
) -> bool
```

### NeuralReconstruction Class

#### Methods:

```python
# Depth estimation
estimate_depth(image: np.ndarray) -> np.ndarray
single_image_to_3d(image: np.ndarray) -> Tuple[PointCloud, np.ndarray]

# Multi-view reconstruction
multi_view_reconstruction(images: List[np.ndarray]) -> PointCloud
create_mesh_from_depth(images: List[np.ndarray]) -> TriangleMesh
```

## 🛠️ Dependencies

### Core Dependencies:
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- Open3D (`open3d`)
- scikit-image (`scikit-image`)
- matplotlib (`matplotlib`)

### Neural Network Dependencies:
- PyTorch (`torch`, `torchvision`)
- Transformers (`transformers`)
- Accelerate (`accelerate`)

### Web Interface Dependencies:
- Gradio (`gradio`)
- Plotly (`plotly`)

### Optional Dependencies:
- Trimesh (`trimesh`) - Additional mesh processing
- scikit-learn (`scikit-learn`) - Clustering algorithms

## 🚨 Troubleshooting

### Common Issues:

#### "No features detected"
- Ensure images have sufficient texture
- Try different lighting conditions
- Check image quality and sharpness

#### "Reconstruction failed"
- Verify image overlap between views
- Try different mesh reconstruction methods
- Check camera calibration

#### "Neural model loading failed"
- Install PyTorch: `pip install torch torchvision`
- Check internet connection for model download
- Verify CUDA availability for GPU acceleration

#### Memory issues
- Reduce image resolution
- Use fewer images
- Enable point cloud downsampling

### Performance Tips:

1. **GPU Acceleration**: Install CUDA-enabled PyTorch for neural methods
2. **Image Size**: Resize images to 1024x1024 for faster processing
3. **Feature Limit**: Reduce SIFT features for large datasets
4. **Mesh Quality**: Adjust Poisson depth parameter (lower = faster)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### Development Setup:

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black *.py

# Lint code
flake8 *.py
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Open3D**: 3D data processing library
- **OpenCV**: Computer vision algorithms
- **Intel DPT**: Depth estimation models
- **Hugging Face**: Transformer models
- **COLMAP**: Inspiration for SfM algorithms

## 📚 References

1. Lowe, D. G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints"
2. Ranftl, R., et al. (2021). "Vision Transformers for Dense Prediction"
3. Kazhdan, M., et al. (2006). "Poisson Surface Reconstruction"
4. Edelsbrunner, H., et al. (1983). "On the Shape of a Set of Points in the Plane"

## 🔗 Related Projects

- [COLMAP](https://colmap.github.io/) - Structure-from-Motion
- [OpenMVG](https://openmvg.readthedocs.io/) - Multiple View Geometry
- [MiDaS](https://github.com/isl-org/MiDaS) - Monocular Depth Estimation
- [NeRF](https://www.matthewtancik.com/nerf) - Neural Radiance Fields

---

Made with ❤️ for the 3D reconstruction community
