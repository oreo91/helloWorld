import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import pipeline, DPTForDepthEstimation, DPTImageProcessor
import numpy as np
import cv2
from PIL import Image
import open3d as o3d
from typing import List, Optional, Tuple
import logging
import os

class NeuralReconstruction:
    """
    Neural network-based 3D reconstruction using monocular depth estimation
    and multi-view synthesis techniques.
    """
    
    def __init__(self, device: str = "auto"):
        """Initialize neural reconstruction pipeline."""
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize depth estimation model
        self.depth_model = None
        self.depth_processor = None
        self._init_depth_model()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def _init_depth_model(self):
        """Initialize the depth estimation model."""
        try:
            self.logger.info("Loading depth estimation model...")
            
            # Use DPT (Dense Prediction Transformer) for depth estimation
            model_name = "Intel/dpt-large"
            self.depth_processor = DPTImageProcessor.from_pretrained(model_name)
            self.depth_model = DPTForDepthEstimation.from_pretrained(model_name)
            self.depth_model.to(self.device)
            self.depth_model.eval()
            
            self.logger.info("Depth estimation model loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to load advanced depth model: {e}")
            self.logger.info("Falling back to basic depth estimation")
            self.depth_model = None
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from a single image.
        
        Args:
            image: Input image as numpy array (H, W, 3)
            
        Returns:
            depth_map: Depth map as numpy array (H, W)
        """
        if self.depth_model is None:
            return self._estimate_depth_stereo(image)
        
        try:
            # Convert to PIL Image
            if image.dtype == np.uint8:
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray((image * 255).astype(np.uint8))
            
            # Process image
            inputs = self.depth_processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.depth_model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # Interpolate to original size
            original_size = image.shape[:2]
            depth_map = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=original_size,
                mode="bicubic",
                align_corners=False
            )
            
            # Convert to numpy
            depth_map = depth_map.squeeze().cpu().numpy()
            
            # Normalize depth values
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            
            return depth_map
            
        except Exception as e:
            self.logger.error(f"Error in depth estimation: {e}")
            return self._estimate_depth_stereo(image)
    
    def _estimate_depth_stereo(self, image: np.ndarray) -> np.ndarray:
        """Fallback depth estimation using stereo vision techniques."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Use Sobel edge detection as a proxy for depth
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        depth_proxy = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Invert and normalize
        depth_proxy = 1.0 - (depth_proxy / depth_proxy.max())
        depth_proxy = cv2.GaussianBlur(depth_proxy, (5, 5), 0)
        
        return depth_proxy
    
    def depth_to_point_cloud(self, image: np.ndarray, depth_map: np.ndarray, 
                           camera_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convert depth map to 3D point cloud.
        
        Args:
            image: RGB image (H, W, 3)
            depth_map: Depth map (H, W)
            camera_matrix: Camera intrinsic matrix (3, 3)
            
        Returns:
            points: 3D points with colors (N, 6) - [x, y, z, r, g, b]
        """
        h, w = depth_map.shape
        
        # Create camera matrix if not provided
        if camera_matrix is None:
            focal_length = max(w, h)
            camera_matrix = np.array([
                [focal_length, 0, w/2],
                [0, focal_length, h/2],
                [0, 0, 1]
            ])
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Convert to 3D points
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        
        # Scale depth values
        depth_scale = 10.0  # Adjust this based on scene scale
        z = depth_map * depth_scale
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Stack coordinates
        points_3d = np.stack([x, y, z], axis=-1)
        
        # Reshape to (N, 3)
        points_3d = points_3d.reshape(-1, 3)
        
        # Get colors
        colors = image.reshape(-1, 3) / 255.0
        
        # Combine points and colors
        points_with_colors = np.hstack([points_3d, colors])
        
        # Filter out invalid points (depth = 0 or too close/far)
        valid_mask = (points_3d[:, 2] > 0.1) & (points_3d[:, 2] < depth_scale * 0.9)
        points_with_colors = points_with_colors[valid_mask]
        
        return points_with_colors
    
    def multi_view_reconstruction(self, images: List[np.ndarray], 
                                camera_matrices: Optional[List[np.ndarray]] = None) -> o3d.geometry.PointCloud:
        """
        Perform multi-view 3D reconstruction using depth estimation.
        
        Args:
            images: List of RGB images
            camera_matrices: Optional list of camera matrices
            
        Returns:
            combined_point_cloud: Open3D point cloud
        """
        self.logger.info(f"Processing {len(images)} images for multi-view reconstruction...")
        
        all_points = []
        
        for i, image in enumerate(images):
            self.logger.info(f"Processing image {i+1}/{len(images)}")
            
            # Estimate depth
            depth_map = self.estimate_depth(image)
            
            # Get camera matrix
            camera_matrix = camera_matrices[i] if camera_matrices else None
            
            # Convert to point cloud
            points_with_colors = self.depth_to_point_cloud(image, depth_map, camera_matrix)
            
            # Apply view transformation if multiple views
            if len(images) > 1:
                # Simple rotation around Y-axis for demonstration
                angle = (2 * np.pi * i) / len(images)
                rotation_matrix = np.array([
                    [np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)]
                ])
                
                # Apply rotation to 3D points
                points_3d = points_with_colors[:, :3]
                rotated_points = (rotation_matrix @ points_3d.T).T
                points_with_colors[:, :3] = rotated_points
            
            all_points.append(points_with_colors)
        
        # Combine all point clouds
        combined_points = np.vstack(all_points)
        
        # Remove duplicates and outliers
        combined_points = self._filter_point_cloud(combined_points)
        
        # Create Open3D point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(combined_points[:, :3])
        point_cloud.colors = o3d.utility.Vector3dVector(combined_points[:, 3:6])
        
        # Estimate normals
        point_cloud.estimate_normals()
        
        self.logger.info(f"Created point cloud with {len(combined_points)} points")
        
        return point_cloud
    
    def _filter_point_cloud(self, points: np.ndarray, voxel_size: float = 0.01) -> np.ndarray:
        """Filter and downsample point cloud."""
        # Create temporary Open3D point cloud for filtering
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        temp_pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6])
        
        # Voxel downsampling
        temp_pcd = temp_pcd.voxel_down_sample(voxel_size)
        
        # Statistical outlier removal
        temp_pcd, _ = temp_pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )
        
        # Convert back to numpy
        filtered_points = np.asarray(temp_pcd.points)
        filtered_colors = np.asarray(temp_pcd.colors)
        
        return np.hstack([filtered_points, filtered_colors])
    
    def create_mesh_from_depth(self, images: List[np.ndarray], 
                             camera_matrices: Optional[List[np.ndarray]] = None,
                             method: str = "poisson") -> o3d.geometry.TriangleMesh:
        """
        Create 3D mesh from images using depth estimation.
        
        Args:
            images: List of RGB images
            camera_matrices: Optional list of camera matrices
            method: Mesh reconstruction method
            
        Returns:
            mesh: Open3D triangle mesh
        """
        # Get point cloud
        point_cloud = self.multi_view_reconstruction(images, camera_matrices)
        
        # Create mesh
        if method == "poisson":
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                point_cloud, depth=9
            )
        elif method == "alpha_shape":
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                point_cloud, alpha=0.03
            )
        elif method == "ball_pivoting":
            radii = [0.005, 0.01, 0.02, 0.04]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                point_cloud, o3d.utility.DoubleVector(radii)
            )
        else:
            raise ValueError(f"Unknown mesh method: {method}")
        
        # Clean mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        return mesh
    
    def single_image_to_3d(self, image: np.ndarray, 
                          camera_matrix: Optional[np.ndarray] = None) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """
        Create 3D representation from a single image.
        
        Args:
            image: RGB image
            camera_matrix: Optional camera matrix
            
        Returns:
            point_cloud: 3D point cloud
            depth_map: Estimated depth map
        """
        self.logger.info("Creating 3D model from single image...")
        
        # Estimate depth
        depth_map = self.estimate_depth(image)
        
        # Convert to point cloud
        points_with_colors = self.depth_to_point_cloud(image, depth_map, camera_matrix)
        
        # Create Open3D point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_with_colors[:, :3])
        point_cloud.colors = o3d.utility.Vector3dVector(points_with_colors[:, 3:6])
        
        # Estimate normals
        point_cloud.estimate_normals()
        
        return point_cloud, depth_map


# Integration with main Model3DBuilder class
def add_neural_reconstruction_to_builder():
    """Add neural reconstruction capabilities to the main Model3DBuilder class."""
    
    def __init_neural__(self):
        """Initialize neural reconstruction module."""
        self.neural_reconstruction = NeuralReconstruction()
    
    def reconstruct_with_neural_depth(self, image_paths: List[str], 
                                    camera_matrices: Optional[List[np.ndarray]] = None) -> bool:
        """
        Reconstruct 3D model using neural depth estimation.
        
        Args:
            image_paths: List of image file paths
            camera_matrices: Optional camera matrices for each image
            
        Returns:
            bool: Success status
        """
        if not hasattr(self, 'neural_reconstruction'):
            self.neural_reconstruction = NeuralReconstruction()
        
        # Load images
        if not self.load_images(image_paths):
            return False
        
        # Perform neural reconstruction
        self.point_cloud = self.neural_reconstruction.multi_view_reconstruction(
            self.images, camera_matrices
        )
        
        # Create mesh using Poisson reconstruction
        try:
            self.mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                self.point_cloud, depth=9
            )
            self.mesh.remove_degenerate_triangles()
            self.mesh.remove_duplicated_triangles()
            self.mesh.remove_duplicated_vertices()
            self.mesh.remove_non_manifold_edges()
            
            self.logger.info("Neural 3D reconstruction completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Mesh creation failed: {e}")
            return False
    
    # Add methods to Model3DBuilder class
    from model_builder_3d import Model3DBuilder
    Model3DBuilder.init_neural = __init_neural__
    Model3DBuilder.reconstruct_with_neural_depth = reconstruct_with_neural_depth


if __name__ == "__main__":
    # Example usage
    neural_recon = NeuralReconstruction()
    
    # Test with single image
    image_path = "test_image.jpg"
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        point_cloud, depth_map = neural_recon.single_image_to_3d(image_rgb)
        
        # Visualize
        o3d.visualization.draw_geometries([point_cloud])
    else:
        print(f"Test image not found: {image_path}")