#!/usr/bin/env python3
"""
Demo script for 3D Model Builder from Images
"""

import os
import numpy as np
import cv2
from model_builder_3d import Model3DBuilder
from neural_reconstruction import NeuralReconstruction
import matplotlib.pyplot as plt
from PIL import Image

def create_sample_images():
    """Create sample images for testing (synthetic cube)."""
    print("Creating sample images for demonstration...")
    
    os.makedirs("sample_images", exist_ok=True)
    
    # Create a simple 3D scene with a textured cube
    def draw_cube(img, center, size, angle, color):
        """Draw a simple cube projection."""
        h, w = img.shape[:2]
        
        # Define cube vertices
        vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Back face
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # Front face
        ]) * size / 2
        
        # Rotation around Y-axis
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
        
        # Apply rotation
        rotated_vertices = vertices @ rotation.T
        
        # Project to 2D (simple perspective projection)
        focal_length = max(w, h)
        projected = []
        for vertex in rotated_vertices:
            x, y, z = vertex
            z += 3  # Move away from camera
            if z > 0:
                px = int(focal_length * x / z + center[0])
                py = int(focal_length * y / z + center[1])
                projected.append((px, py))
            else:
                projected.append(None)
        
        # Draw cube faces
        faces = [
            [0, 1, 2, 3],  # Back face
            [4, 5, 6, 7],  # Front face
            [0, 1, 5, 4],  # Bottom
            [2, 3, 7, 6],  # Top
            [0, 3, 7, 4],  # Left
            [1, 2, 6, 5]   # Right
        ]
        
        face_colors = [
            (color[0] * 0.6, color[1] * 0.6, color[2] * 0.6),  # Back (darker)
            color,  # Front
            (color[0] * 0.8, color[1] * 0.8, color[2] * 0.8),  # Bottom
            (color[0] * 1.2, color[1] * 1.2, color[2] * 1.2),  # Top (brighter)
            (color[0] * 0.7, color[1] * 0.7, color[2] * 0.7),  # Left
            (color[0] * 0.9, color[1] * 0.9, color[2] * 0.9),  # Right
        ]
        
        for face_idx, face in enumerate(faces):
            points = []
            for vertex_idx in face:
                if projected[vertex_idx] is not None:
                    points.append(projected[vertex_idx])
            
            if len(points) >= 3:
                pts = np.array(points, dtype=np.int32)
                fc = face_colors[face_idx]
                cv2.fillPoly(img, [pts], (int(fc[2]), int(fc[1]), int(fc[0])))
                cv2.polylines(img, [pts], True, (0, 0, 0), 2)
        
        return img
    
    # Generate images from different viewpoints
    num_images = 6
    image_size = (512, 512)
    
    for i in range(num_images):
        # Create blank image
        img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 200
        
        # Add some texture/noise to background
        noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        # Calculate viewpoint angle
        angle = (2 * np.pi * i) / num_images
        
        # Draw cube
        center = (image_size[0] // 2, image_size[1] // 2)
        cube_size = 100
        color = (100, 150, 200)  # BGR format
        
        img = draw_cube(img, center, cube_size, angle, color)
        
        # Add some additional features (circles for better feature matching)
        for j in range(5):
            x = np.random.randint(50, image_size[0] - 50)
            y = np.random.randint(50, image_size[1] - 50)
            radius = np.random.randint(10, 30)
            color_circle = (
                np.random.randint(50, 255),
                np.random.randint(50, 255),
                np.random.randint(50, 255)
            )
            cv2.circle(img, (x, y), radius, color_circle, -1)
            cv2.circle(img, (x, y), radius, (0, 0, 0), 2)
        
        # Save image
        filename = f"sample_images/img_{i+1:02d}.jpg"
        cv2.imwrite(filename, img)
        print(f"Created {filename}")
    
    print(f"Created {num_images} sample images in 'sample_images/' directory")
    return [f"sample_images/img_{i+1:02d}.jpg" for i in range(num_images)]

def demo_traditional_reconstruction():
    """Demonstrate traditional photogrammetry reconstruction."""
    print("\n" + "="*60)
    print("TRADITIONAL PHOTOGRAMMETRY RECONSTRUCTION DEMO")
    print("="*60)
    
    # Create sample images if they don't exist
    image_paths = []
    for i in range(6):
        path = f"sample_images/img_{i+1:02d}.jpg"
        if not os.path.exists(path):
            image_paths = create_sample_images()
            break
        image_paths.append(path)
    
    if not image_paths:
        image_paths = create_sample_images()
    
    # Initialize builder
    builder = Model3DBuilder(output_dir="demo_output_traditional")
    
    # Perform reconstruction
    print("\nStarting traditional 3D reconstruction...")
    success = builder.reconstruct_from_images(
        image_paths=image_paths,
        mesh_method="poisson"
    )
    
    if success:
        # Save results
        output_files = builder.save_results("traditional_demo")
        
        print("\n✅ Traditional reconstruction completed!")
        print(f"Point cloud: {len(builder.point_cloud.points)} points")
        if builder.mesh:
            print(f"Mesh: {len(builder.mesh.vertices)} vertices, {len(builder.mesh.triangles)} triangles")
        
        print("\nOutput files:")
        for file_type, file_path in output_files.items():
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"  {file_type}: {file_path} ({file_size:.1f} KB)")
        
        # Visualize
        print("\nPress any key in the visualization window to continue...")
        builder.visualize()
        
        return True
    else:
        print("❌ Traditional reconstruction failed!")
        return False

def demo_neural_reconstruction():
    """Demonstrate neural network-based reconstruction."""
    print("\n" + "="*60)
    print("NEURAL NETWORK RECONSTRUCTION DEMO")
    print("="*60)
    
    try:
        # Initialize neural reconstruction
        neural_recon = NeuralReconstruction()
        
        # Create sample images if they don't exist
        image_paths = []
        for i in range(3):  # Use fewer images for neural demo
            path = f"sample_images/img_{i+1:02d}.jpg"
            if not os.path.exists(path):
                image_paths = create_sample_images()
                break
            image_paths.append(path)
        
        if not image_paths:
            image_paths = create_sample_images()[:3]
        
        # Load images
        images = []
        for path in image_paths:
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img_rgb)
        
        print(f"\nProcessing {len(images)} images with neural reconstruction...")
        
        # Perform neural reconstruction
        point_cloud = neural_recon.multi_view_reconstruction(images)
        
        # Create mesh
        mesh = neural_recon.create_mesh_from_depth(images, method="poisson")
        
        print("\n✅ Neural reconstruction completed!")
        print(f"Point cloud: {len(point_cloud.points)} points")
        print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        
        # Save results
        os.makedirs("demo_output_neural", exist_ok=True)
        
        pc_file = "demo_output_neural/neural_pointcloud.ply"
        mesh_file = "demo_output_neural/neural_mesh.ply"
        
        import open3d as o3d
        o3d.io.write_point_cloud(pc_file, point_cloud)
        o3d.io.write_triangle_mesh(mesh_file, mesh)
        
        print(f"\nSaved point cloud to: {pc_file}")
        print(f"Saved mesh to: {mesh_file}")
        
        # Visualize
        print("\nPress any key in the visualization window to continue...")
        o3d.visualization.draw_geometries([point_cloud, mesh])
        
        return True
        
    except Exception as e:
        print(f"❌ Neural reconstruction failed: {e}")
        print("This might be due to missing neural network dependencies.")
        print("Try installing: pip install torch torchvision transformers")
        return False

def demo_single_image_depth():
    """Demonstrate single image depth estimation."""
    print("\n" + "="*60)
    print("SINGLE IMAGE DEPTH ESTIMATION DEMO")
    print("="*60)
    
    try:
        neural_recon = NeuralReconstruction()
        
        # Use the first sample image
        image_path = "sample_images/img_01.jpg"
        if not os.path.exists(image_path):
            create_sample_images()
        
        # Load image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        print(f"Processing single image: {image_path}")
        
        # Estimate depth and create 3D model
        point_cloud, depth_map = neural_recon.single_image_to_3d(img_rgb)
        
        print(f"\n✅ Single image processing completed!")
        print(f"Generated point cloud with {len(point_cloud.points)} points")
        
        # Save depth map visualization
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(img_rgb)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(depth_map, cmap='viridis')
        plt.title("Estimated Depth Map")
        plt.axis('off')
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        # Show point cloud projection
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)
        
        plt.scatter(points[:, 0], points[:, 1], c=colors, s=1)
        plt.title("Point Cloud (X-Y projection)")
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig("demo_output_neural/depth_visualization.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        # Save point cloud
        os.makedirs("demo_output_neural", exist_ok=True)
        pc_file = "demo_output_neural/single_image_pointcloud.ply"
        
        import open3d as o3d
        o3d.io.write_point_cloud(pc_file, point_cloud)
        print(f"Saved point cloud to: {pc_file}")
        
        # Visualize 3D
        print("\nPress any key in the 3D visualization window to continue...")
        o3d.visualization.draw_geometries([point_cloud])
        
        return True
        
    except Exception as e:
        print(f"❌ Single image depth estimation failed: {e}")
        return False

def compare_methods():
    """Compare different reconstruction methods."""
    print("\n" + "="*60)
    print("METHOD COMPARISON")
    print("="*60)
    
    methods = ["poisson", "alpha_shape", "ball_pivoting"]
    
    # Create sample images if needed
    image_paths = []
    for i in range(4):
        path = f"sample_images/img_{i+1:02d}.jpg"
        if not os.path.exists(path):
            image_paths = create_sample_images()
            break
        image_paths.append(path)
    
    if not image_paths:
        image_paths = create_sample_images()[:4]
    
    results = {}
    
    for method in methods:
        print(f"\nTesting {method} method...")
        
        builder = Model3DBuilder(output_dir=f"demo_output_comparison/{method}")
        
        success = builder.reconstruct_from_images(
            image_paths=image_paths,
            mesh_method=method
        )
        
        if success:
            results[method] = {
                "point_cloud_size": len(builder.point_cloud.points) if builder.point_cloud else 0,
                "mesh_vertices": len(builder.mesh.vertices) if builder.mesh else 0,
                "mesh_triangles": len(builder.mesh.triangles) if builder.mesh else 0
            }
            
            # Save results
            builder.save_results(f"{method}_comparison")
            
            print(f"✅ {method}: {results[method]}")
        else:
            print(f"❌ {method}: Failed")
            results[method] = None
    
    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    for method, result in results.items():
        if result:
            print(f"{method:15}: {result['point_cloud_size']:8d} points, "
                  f"{result['mesh_vertices']:6d} vertices, "
                  f"{result['mesh_triangles']:6d} triangles")
        else:
            print(f"{method:15}: Failed")

def main():
    """Main demo function."""
    print("🏗️  3D MODEL BUILDER FROM IMAGES - DEMO")
    print("="*60)
    print("This demo will showcase different 3D reconstruction methods:")
    print("1. Traditional photogrammetry (SIFT + triangulation)")
    print("2. Neural network depth estimation") 
    print("3. Single image to 3D conversion")
    print("4. Method comparison")
    print("="*60)
    
    # Create output directories
    os.makedirs("demo_output_traditional", exist_ok=True)
    os.makedirs("demo_output_neural", exist_ok=True)
    os.makedirs("demo_output_comparison", exist_ok=True)
    
    # Run demos
    try:
        # Traditional reconstruction
        demo_traditional_reconstruction()
        
        # Neural reconstruction (optional, may fail if dependencies missing)
        demo_neural_reconstruction()
        
        # Single image depth estimation
        demo_single_image_depth()
        
        # Method comparison
        compare_methods()
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Check the following directories for output files:")
        print("- demo_output_traditional/")
        print("- demo_output_neural/")
        print("- demo_output_comparison/")
        print("\nYou can view the .ply files in any 3D viewer or CAD software.")
        print("The .obj files can be imported into Blender, Maya, etc.")
        print("The .stl files are ready for 3D printing!")
        
    except KeyboardInterrupt:
        print("\n❌ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()