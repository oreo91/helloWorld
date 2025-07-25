import gradio as gr
import os
import tempfile
import shutil
from model_builder_3d import Model3DBuilder
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

class WebInterface:
    def __init__(self):
        self.builder = None
        self.temp_dir = None
    
    def process_images(self, uploaded_files, mesh_method="poisson", progress=gr.Progress()):
        """Process uploaded images and create 3D model."""
        if not uploaded_files or len(uploaded_files) < 2:
            return None, "Please upload at least 2 images for 3D reconstruction.", None, None
        
        try:
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp()
            
            # Save uploaded files
            image_paths = []
            progress(0.1, desc="Saving uploaded images...")
            
            for i, file in enumerate(uploaded_files):
                temp_path = os.path.join(self.temp_dir, f"image_{i}.jpg")
                shutil.copy(file.name, temp_path)
                image_paths.append(temp_path)
            
            # Initialize builder
            self.builder = Model3DBuilder(output_dir=os.path.join(self.temp_dir, "output"))
            
            progress(0.2, desc="Loading images...")
            
            # Perform reconstruction
            success = self.builder.reconstruct_from_images(
                image_paths, 
                mesh_method=mesh_method
            )
            
            if not success:
                return None, "3D reconstruction failed. Please try with different images.", None, None
            
            progress(0.8, desc="Saving results...")
            
            # Save results
            output_files = self.builder.save_results("web_model")
            
            progress(1.0, desc="Complete!")
            
            # Generate visualization
            point_cloud_plot = self.create_point_cloud_plot()
            mesh_plot = self.create_mesh_plot()
            
            # Prepare download files
            download_files = []
            for file_type, file_path in output_files.items():
                if os.path.exists(file_path):
                    download_files.append(file_path)
            
            status_msg = f"""
            ✅ 3D Reconstruction Successful!
            
            📊 Results:
            - Point Cloud: {len(self.builder.point_cloud.points) if self.builder.point_cloud else 0} points
            - Mesh: {len(self.builder.mesh.vertices) if self.builder.mesh else 0} vertices, {len(self.builder.mesh.triangles) if self.builder.mesh else 0} triangles
            - Output files: {len(download_files)} files generated
            """
            
            return download_files, status_msg, point_cloud_plot, mesh_plot
            
        except Exception as e:
            return None, f"Error during reconstruction: {str(e)}", None, None
    
    def create_point_cloud_plot(self):
        """Create a Plotly 3D scatter plot of the point cloud."""
        if not self.builder or not self.builder.point_cloud:
            return None
        
        points = np.asarray(self.builder.point_cloud.points)
        
        # Sample points if too many for visualization
        if len(points) > 10000:
            indices = np.random.choice(len(points), 10000, replace=False)
            points = points[indices]
        
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=points[:, 2],  # Color by Z coordinate
                colorscale='Viridis',
                opacity=0.8
            ),
            name='Point Cloud'
        )])
        
        fig.update_layout(
            title="3D Point Cloud",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_mesh_plot(self):
        """Create a Plotly 3D mesh plot."""
        if not self.builder or not self.builder.mesh:
            return None
        
        vertices = np.asarray(self.builder.mesh.vertices)
        triangles = np.asarray(self.builder.mesh.triangles)
        
        # Sample triangles if too many for visualization
        if len(triangles) > 5000:
            indices = np.random.choice(len(triangles), 5000, replace=False)
            triangles = triangles[indices]
        
        fig = go.Figure(data=[go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            opacity=0.8,
            colorscale='Viridis'
        )])
        
        fig.update_layout(
            title="3D Mesh",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


def create_interface():
    """Create the Gradio interface."""
    web_interface = WebInterface()
    
    with gr.Blocks(title="3D Model Builder from Images", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🏗️ 3D Model Builder from Images
        
        Upload multiple images of an object from different angles to create a 3D model using photogrammetry and computer vision techniques.
        
        ## Instructions:
        1. **Upload Images**: Select 3-10 images of the same object taken from different viewpoints
        2. **Choose Method**: Select the mesh reconstruction method
        3. **Process**: Click "Generate 3D Model" and wait for processing
        4. **Download**: Get your 3D model files in various formats
        
        ### Tips for Best Results:
        - Take photos around the object in a circular pattern
        - Ensure good lighting and avoid shadows
        - Include overlapping features between consecutive images
        - Use a plain background if possible
        - Keep the object stationary between shots
        """)
        
        with gr.Row():
            with gr.Column():
                # Input section
                gr.Markdown("### 📤 Input Images")
                
                image_input = gr.File(
                    label="Upload Images (3-10 images)",
                    file_count="multiple",
                    file_types=["image"],
                    height=200
                )
                
                mesh_method = gr.Dropdown(
                    choices=["poisson", "alpha_shape", "ball_pivoting"],
                    value="poisson",
                    label="Mesh Reconstruction Method",
                    info="Poisson: Smooth surfaces, Alpha Shape: Detailed surfaces, Ball Pivoting: Good for noisy data"
                )
                
                process_btn = gr.Button(
                    "🚀 Generate 3D Model",
                    variant="primary",
                    size="lg"
                )
        
        with gr.Row():
            # Status and download section
            status_output = gr.Markdown(
                "Upload images and click 'Generate 3D Model' to start",
                label="Status"
            )
        
        with gr.Row():
            # Download section
            download_files = gr.File(
                label="📥 Download 3D Model Files",
                file_count="multiple",
                visible=False
            )
        
        with gr.Row():
            with gr.Column():
                # Point cloud visualization
                point_cloud_plot = gr.Plot(
                    label="3D Point Cloud Visualization",
                    visible=False
                )
            
            with gr.Column():
                # Mesh visualization
                mesh_plot = gr.Plot(
                    label="3D Mesh Visualization",
                    visible=False
                )
        
        # Advanced options (collapsible)
        with gr.Accordion("🔧 Advanced Options", open=False):
            gr.Markdown("""
            ### Camera Calibration
            For better results, you can provide camera calibration images with a checkerboard pattern.
            This is optional - the system will estimate camera parameters automatically if not provided.
            """)
            
            calibration_images = gr.File(
                label="Camera Calibration Images (Optional)",
                file_count="multiple",
                file_types=["image"]
            )
        
        # Technical details (collapsible)
        with gr.Accordion("📚 Technical Details", open=False):
            gr.Markdown("""
            ### How it Works:
            
            1. **Feature Detection**: SIFT features are detected in each image
            2. **Feature Matching**: Corresponding features are matched between image pairs
            3. **Camera Pose Estimation**: Essential matrix and camera poses are computed
            4. **Triangulation**: 3D points are triangulated from matched features
            5. **Point Cloud Generation**: All 3D points are combined and filtered
            6. **Mesh Generation**: Surface reconstruction creates a 3D mesh
            
            ### Supported Output Formats:
            - **PLY**: Point cloud and mesh format
            - **OBJ**: Wavefront 3D model format
            - **STL**: Stereolithography format for 3D printing
            - **JSON**: Metadata and parameters
            
            ### Algorithms Used:
            - **SIFT**: Scale-Invariant Feature Transform for feature detection
            - **RANSAC**: Random Sample Consensus for robust estimation
            - **Poisson Reconstruction**: Smooth surface reconstruction
            - **Alpha Shapes**: Detailed surface reconstruction
            - **Ball Pivoting**: Robust mesh generation
            """)
        
        # Event handlers
        def on_process(images, method, progress=gr.Progress()):
            files, status, pc_plot, mesh_plot = web_interface.process_images(
                images, method, progress
            )
            
            # Update visibility based on results
            download_visible = files is not None and len(files) > 0
            plot_visible = pc_plot is not None or mesh_plot is not None
            
            return (
                files,
                status,
                gr.update(value=pc_plot, visible=plot_visible),
                gr.update(value=mesh_plot, visible=plot_visible),
                gr.update(visible=download_visible)
            )
        
        process_btn.click(
            fn=on_process,
            inputs=[image_input, mesh_method],
            outputs=[
                download_files,
                status_output,
                point_cloud_plot,
                mesh_plot,
                download_files
            ]
        )
        
        # Cleanup on app close
        app.load(lambda: web_interface.cleanup)
    
    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )