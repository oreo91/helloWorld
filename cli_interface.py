#!/usr/bin/env python3
"""
Command-line interface for 3D Model Builder from Images
"""

import argparse
import os
import sys
import glob
from pathlib import Path
from model_builder_3d import Model3DBuilder

def main():
    parser = argparse.ArgumentParser(
        description="Build 3D models from images using photogrammetry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with image directory
  python cli_interface.py --input-dir ./images --output-dir ./output
  
  # Specify individual images
  python cli_interface.py --images img1.jpg img2.jpg img3.jpg --output-dir ./output
  
  # Use alpha shape reconstruction with calibration
  python cli_interface.py --input-dir ./images --mesh-method alpha_shape --calibration ./calib
  
  # Generate only point cloud
  python cli_interface.py --input-dir ./images --no-mesh --visualize
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dir", "-i",
        type=str,
        help="Directory containing input images"
    )
    input_group.add_argument(
        "--images",
        nargs="+",
        help="List of image file paths"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="output_3d",
        help="Output directory for 3D models (default: output_3d)"
    )
    
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="model",
        help="Prefix for output files (default: model)"
    )
    
    # Reconstruction options
    parser.add_argument(
        "--mesh-method",
        choices=["poisson", "alpha_shape", "ball_pivoting"],
        default="poisson",
        help="Mesh reconstruction method (default: poisson)"
    )
    
    parser.add_argument(
        "--no-mesh",
        action="store_true",
        help="Skip mesh generation, create only point cloud"
    )
    
    # Camera calibration
    parser.add_argument(
        "--calibration",
        type=str,
        help="Directory containing camera calibration images with checkerboard pattern"
    )
    
    # Visualization and output
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show 3D visualization after reconstruction"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    # File filtering
    parser.add_argument(
        "--image-extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
        help="Image file extensions to process (default: .jpg .jpeg .png .bmp .tiff)"
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    try:
        # Get image paths
        if args.input_dir:
            image_paths = get_images_from_directory(args.input_dir, args.image_extensions)
            if not image_paths:
                print(f"Error: No images found in directory '{args.input_dir}'")
                return 1
        else:
            image_paths = args.images
            # Validate image files exist
            for path in image_paths:
                if not os.path.exists(path):
                    print(f"Error: Image file not found: {path}")
                    return 1
        
        print(f"Found {len(image_paths)} images for processing")
        
        # Get calibration images if specified
        calibration_images = None
        if args.calibration:
            if os.path.isdir(args.calibration):
                calibration_images = get_images_from_directory(args.calibration, args.image_extensions)
                print(f"Found {len(calibration_images)} calibration images")
            else:
                print(f"Warning: Calibration directory not found: {args.calibration}")
        
        # Initialize 3D model builder
        builder = Model3DBuilder(output_dir=args.output_dir)
        
        print("Starting 3D reconstruction...")
        
        # Perform reconstruction
        success = builder.reconstruct_from_images(
            image_paths=image_paths,
            calibration_images=calibration_images,
            mesh_method=args.mesh_method if not args.no_mesh else None
        )
        
        if not success:
            print("Error: 3D reconstruction failed!")
            return 1
        
        # Save results
        print("Saving results...")
        output_files = builder.save_results(args.output_prefix)
        
        # Print summary
        print("\n" + "="*50)
        print("3D RECONSTRUCTION COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        if builder.point_cloud:
            print(f"Point Cloud: {len(builder.point_cloud.points)} points")
        
        if builder.mesh:
            print(f"Mesh: {len(builder.mesh.vertices)} vertices, {len(builder.mesh.triangles)} triangles")
        
        print(f"\nOutput files saved to: {args.output_dir}")
        for file_type, file_path in output_files.items():
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"  {file_type}: {os.path.basename(file_path)} ({file_size:.2f} MB)")
        
        # Visualization
        if args.visualize:
            print("\nOpening 3D visualization...")
            builder.visualize()
        
        print(f"\n✅ 3D model reconstruction completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def get_images_from_directory(directory: str, extensions: list) -> list:
    """Get all image files from a directory with specified extensions."""
    image_paths = []
    
    for ext in extensions:
        # Handle both with and without dot
        if not ext.startswith('.'):
            ext = '.' + ext
        
        pattern = os.path.join(directory, f"*{ext}")
        pattern_upper = os.path.join(directory, f"*{ext.upper()}")
        
        image_paths.extend(glob.glob(pattern))
        image_paths.extend(glob.glob(pattern_upper))
    
    # Sort for consistent ordering
    image_paths.sort()
    
    return image_paths


def validate_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'cv2', 'numpy', 'open3d', 'matplotlib', 
        'sklearn', 'trimesh', 'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            if package == 'cv2':
                missing_packages.append('opencv-python')
            elif package == 'PIL':
                missing_packages.append('Pillow')
            elif package == 'sklearn':
                missing_packages.append('scikit-learn')
            else:
                missing_packages.append(package)
    
    if missing_packages:
        print("Error: Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


if __name__ == "__main__":
    # Check requirements first
    if not validate_requirements():
        sys.exit(1)
    
    # Run main application
    sys.exit(main())