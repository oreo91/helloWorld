import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import trimesh
from PIL import Image
import os
import json
from typing import List, Tuple, Optional, Dict, Any
import logging
from tqdm import tqdm

class Model3DBuilder:
    """
    A comprehensive 3D model builder from images using multiple approaches:
    1. Structure from Motion (SfM) with photogrammetry
    2. Stereo vision reconstruction
    3. Multi-view stereo (MVS)
    4. Neural reconstruction support
    """
    
    def __init__(self, output_dir: str = "output_3d"):
        """Initialize the 3D model builder."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Store image data
        self.images = []
        self.image_paths = []
        self.keypoints_list = []
        self.descriptors_list = []
        
        # 3D reconstruction data
        self.point_cloud = None
        self.mesh = None
        
        # Feature detector and matcher
        self.detector = cv2.SIFT_create(nfeatures=5000)
        self.matcher = cv2.BFMatcher()
        
    def load_images(self, image_paths: List[str]) -> bool:
        """Load and preprocess images for 3D reconstruction."""
        self.logger.info(f"Loading {len(image_paths)} images...")
        
        self.images = []
        self.image_paths = image_paths
        
        for path in tqdm(image_paths, desc="Loading images"):
            if not os.path.exists(path):
                self.logger.warning(f"Image not found: {path}")
                continue
                
            img = cv2.imread(path)
            if img is None:
                self.logger.warning(f"Could not load image: {path}")
                continue
                
            # Convert to RGB for consistency
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.images.append(img_rgb)
            
        self.logger.info(f"Successfully loaded {len(self.images)} images")
        return len(self.images) > 0
    
    def calibrate_camera(self, calibration_images: Optional[List[str]] = None) -> bool:
        """
        Calibrate camera using checkerboard pattern or estimate from images.
        If no calibration images provided, estimates camera parameters.
        """
        if calibration_images and len(calibration_images) > 0:
            return self._calibrate_with_checkerboard(calibration_images)
        else:
            return self._estimate_camera_parameters()
    
    def _calibrate_with_checkerboard(self, calibration_images: List[str]) -> bool:
        """Calibrate camera using checkerboard pattern."""
        # Checkerboard dimensions
        pattern_size = (9, 6)
        
        # Prepare object points
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        
        # Arrays to store object points and image points
        objpoints = []
        imgpoints = []
        
        for img_path in calibration_images:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
        
        if len(objpoints) > 0:
            img_shape = gray.shape[::-1]
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, img_shape, None, None
            )
            
            if ret:
                self.camera_matrix = mtx
                self.dist_coeffs = dist
                self.logger.info("Camera calibration successful")
                return True
        
        self.logger.warning("Camera calibration failed, using estimated parameters")
        return self._estimate_camera_parameters()
    
    def _estimate_camera_parameters(self) -> bool:
        """Estimate camera parameters from image dimensions."""
        if not self.images:
            return False
        
        h, w = self.images[0].shape[:2]
        
        # Estimate focal length (typical assumption)
        focal_length = max(w, h)
        
        # Create camera matrix
        self.camera_matrix = np.array([
            [focal_length, 0, w/2],
            [0, focal_length, h/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Assume no distortion
        self.dist_coeffs = np.zeros((4, 1))
        
        self.logger.info("Camera parameters estimated")
        return True
    
    def extract_features(self) -> bool:
        """Extract SIFT features from all images."""
        self.logger.info("Extracting features from images...")
        
        self.keypoints_list = []
        self.descriptors_list = []
        
        for i, img in enumerate(tqdm(self.images, desc="Feature extraction")):
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Detect keypoints and descriptors
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)
            
            self.keypoints_list.append(keypoints)
            self.descriptors_list.append(descriptors)
            
            self.logger.info(f"Image {i+1}: {len(keypoints) if keypoints else 0} features")
        
        return len(self.keypoints_list) > 0
    
    def match_features(self, img1_idx: int, img2_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Match features between two images."""
        if (img1_idx >= len(self.descriptors_list) or 
            img2_idx >= len(self.descriptors_list) or
            self.descriptors_list[img1_idx] is None or
            self.descriptors_list[img2_idx] is None):
            return np.array([]), np.array([])
        
        # Match descriptors
        matches = self.matcher.knnMatch(
            self.descriptors_list[img1_idx],
            self.descriptors_list[img2_idx],
            k=2
        )
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            return np.array([]), np.array([])
        
        # Extract matched keypoints
        pts1 = np.float32([
            self.keypoints_list[img1_idx][m.queryIdx].pt for m in good_matches
        ]).reshape(-1, 1, 2)
        
        pts2 = np.float32([
            self.keypoints_list[img2_idx][m.trainIdx].pt for m in good_matches
        ]).reshape(-1, 1, 2)
        
        return pts1, pts2
    
    def triangulate_points(self, img1_idx: int, img2_idx: int) -> np.ndarray:
        """Triangulate 3D points from two views."""
        # Match features
        pts1, pts2 = self.match_features(img1_idx, img2_idx)
        
        if len(pts1) == 0:
            return np.array([])
        
        # Find essential matrix
        essential_matrix, mask = cv2.findEssentialMat(
            pts1, pts2, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        
        if essential_matrix is None:
            return np.array([])
        
        # Recover pose
        _, R, t, mask = cv2.recoverPose(essential_matrix, pts1, pts2, self.camera_matrix)
        
        # Create projection matrices
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = np.hstack([R, t])
        P1 = self.camera_matrix @ P1
        P2 = self.camera_matrix @ P2
        
        # Filter points using mask
        pts1_filtered = pts1[mask.ravel() == 1]
        pts2_filtered = pts2[mask.ravel() == 1]
        
        if len(pts1_filtered) == 0:
            return np.array([])
        
        # Triangulate points
        points_4d = cv2.triangulatePoints(P1, P2, pts1_filtered.T, pts2_filtered.T)
        points_3d = points_4d[:3] / points_4d[3]
        
        return points_3d.T
    
    def build_point_cloud(self) -> bool:
        """Build 3D point cloud from multiple views."""
        self.logger.info("Building 3D point cloud...")
        
        all_points = []
        
        # Process all image pairs
        num_images = len(self.images)
        for i in tqdm(range(num_images), desc="Processing image pairs"):
            for j in range(i + 1, min(i + 3, num_images)):  # Process with next 2 images
                points_3d = self.triangulate_points(i, j)
                if len(points_3d) > 0:
                    all_points.append(points_3d)
        
        if not all_points:
            self.logger.error("No 3D points could be triangulated")
            return False
        
        # Combine all points
        combined_points = np.vstack(all_points)
        
        # Remove outliers using statistical filtering
        combined_points = self._remove_outliers(combined_points)
        
        # Create Open3D point cloud
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(combined_points)
        
        # Estimate normals
        self.point_cloud.estimate_normals()
        
        self.logger.info(f"Point cloud created with {len(combined_points)} points")
        return True
    
    def _remove_outliers(self, points: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        """Remove outliers from point cloud using statistical methods."""
        if len(points) < 10:
            return points
        
        # Remove points that are too far from the mean
        mean = np.mean(points, axis=0)
        distances = np.linalg.norm(points - mean, axis=1)
        threshold = np.percentile(distances, 95)
        
        inliers = points[distances < threshold]
        
        # Use DBSCAN to remove isolated points
        if len(inliers) > 10:
            clustering = DBSCAN(eps=np.std(distances) * 2, min_samples=5)
            labels = clustering.fit_predict(inliers)
            
            # Keep the largest cluster
            if len(np.unique(labels)) > 1:
                largest_cluster = np.argmax(np.bincount(labels[labels >= 0]))
                inliers = inliers[labels == largest_cluster]
        
        return inliers
    
    def create_mesh(self, method: str = "poisson") -> bool:
        """Create 3D mesh from point cloud."""
        if self.point_cloud is None:
            self.logger.error("No point cloud available for mesh generation")
            return False
        
        self.logger.info(f"Creating mesh using {method} method...")
        
        try:
            if method == "poisson":
                # Poisson surface reconstruction
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    self.point_cloud, depth=9
                )
            elif method == "alpha_shape":
                # Alpha shape reconstruction
                alpha = 0.03
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    self.point_cloud, alpha
                )
            elif method == "ball_pivoting":
                # Ball pivoting algorithm
                radii = [0.005, 0.01, 0.02, 0.04]
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    self.point_cloud, o3d.utility.DoubleVector(radii)
                )
            else:
                self.logger.error(f"Unknown mesh creation method: {method}")
                return False
            
            # Clean up mesh
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
            
            self.mesh = mesh
            self.logger.info(f"Mesh created with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
            return True
            
        except Exception as e:
            self.logger.error(f"Mesh creation failed: {str(e)}")
            return False
    
    def save_results(self, filename_prefix: str = "model_3d") -> Dict[str, str]:
        """Save the 3D model in multiple formats."""
        output_files = {}
        
        try:
            if self.point_cloud is not None:
                # Save point cloud
                pc_file = os.path.join(self.output_dir, f"{filename_prefix}_pointcloud.ply")
                o3d.io.write_point_cloud(pc_file, self.point_cloud)
                output_files["point_cloud"] = pc_file
                self.logger.info(f"Point cloud saved to {pc_file}")
            
            if self.mesh is not None:
                # Save mesh in multiple formats
                mesh_ply = os.path.join(self.output_dir, f"{filename_prefix}_mesh.ply")
                mesh_obj = os.path.join(self.output_dir, f"{filename_prefix}_mesh.obj")
                mesh_stl = os.path.join(self.output_dir, f"{filename_prefix}_mesh.stl")
                
                o3d.io.write_triangle_mesh(mesh_ply, self.mesh)
                o3d.io.write_triangle_mesh(mesh_obj, self.mesh)
                o3d.io.write_triangle_mesh(mesh_stl, self.mesh)
                
                output_files["mesh_ply"] = mesh_ply
                output_files["mesh_obj"] = mesh_obj
                output_files["mesh_stl"] = mesh_stl
                
                self.logger.info(f"Mesh saved in multiple formats")
            
            # Save metadata
            metadata = {
                "num_input_images": len(self.images),
                "point_cloud_size": len(self.point_cloud.points) if self.point_cloud else 0,
                "mesh_vertices": len(self.mesh.vertices) if self.mesh else 0,
                "mesh_triangles": len(self.mesh.triangles) if self.mesh else 0,
                "camera_matrix": self.camera_matrix.tolist() if self.camera_matrix is not None else None
            }
            
            metadata_file = os.path.join(self.output_dir, f"{filename_prefix}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            output_files["metadata"] = metadata_file
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
        
        return output_files
    
    def visualize(self, show_point_cloud: bool = True, show_mesh: bool = True):
        """Visualize the 3D reconstruction results."""
        geometries = []
        
        if show_point_cloud and self.point_cloud is not None:
            # Color point cloud
            self.point_cloud.paint_uniform_color([0.5, 0.5, 1.0])
            geometries.append(self.point_cloud)
        
        if show_mesh and self.mesh is not None:
            # Color mesh
            self.mesh.paint_uniform_color([0.8, 0.8, 0.8])
            geometries.append(self.mesh)
        
        if geometries:
            o3d.visualization.draw_geometries(
                geometries,
                window_name="3D Model Reconstruction",
                width=1024,
                height=768
            )
        else:
            self.logger.warning("No 3D geometry to visualize")
    
    def reconstruct_from_images(self, image_paths: List[str], 
                              calibration_images: Optional[List[str]] = None,
                              mesh_method: str = "poisson") -> bool:
        """
        Complete 3D reconstruction pipeline from images.
        
        Args:
            image_paths: List of paths to input images
            calibration_images: Optional list of calibration images with checkerboard
            mesh_method: Method for mesh creation ('poisson', 'alpha_shape', 'ball_pivoting')
        
        Returns:
            bool: True if reconstruction successful
        """
        self.logger.info("Starting 3D reconstruction pipeline...")
        
        # Step 1: Load images
        if not self.load_images(image_paths):
            return False
        
        # Step 2: Camera calibration
        if not self.calibrate_camera(calibration_images):
            return False
        
        # Step 3: Feature extraction
        if not self.extract_features():
            return False
        
        # Step 4: Build point cloud
        if not self.build_point_cloud():
            return False
        
        # Step 5: Create mesh
        if not self.create_mesh(mesh_method):
            self.logger.warning("Mesh creation failed, but point cloud is available")
        
        self.logger.info("3D reconstruction completed successfully!")
        return True


if __name__ == "__main__":
    # Example usage
    builder = Model3DBuilder()
    
    # Example image paths (replace with actual paths)
    image_paths = [
        "sample_images/img1.jpg",
        "sample_images/img2.jpg",
        "sample_images/img3.jpg",
        "sample_images/img4.jpg"
    ]
    
    # Perform reconstruction
    success = builder.reconstruct_from_images(image_paths)
    
    if success:
        # Save results
        output_files = builder.save_results("example_model")
        print("Output files:", output_files)
        
        # Visualize results
        builder.visualize()
    else:
        print("Reconstruction failed!")