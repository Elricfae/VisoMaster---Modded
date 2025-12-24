#!/usr/bin/env python3
"""
VisoMaster Gradio Web Interface
Converts the desktop Qt application to a web-based interface using Gradio
"""

import os
import sys
import gradio as gr
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Optional, Tuple
import tempfile
import traceback

# Ensure no Qt/Display is required
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Mock Qt objects to avoid import errors
class MockQObject:
    def __init__(self, *args, **kwargs):
        pass

class MockSignal:
    def __init__(self, *args):
        pass

    def emit(self, *args, **kwargs):
        pass

    def connect(self, *args, **kwargs):
        pass

# Mock PySide6 before importing app modules
sys.modules['PySide6'] = type(sys)('PySide6')
sys.modules['PySide6.QtCore'] = type(sys)('PySide6.QtCore')
sys.modules['PySide6.QtWidgets'] = type(sys)('PySide6.QtWidgets')
sys.modules['PySide6.QtGui'] = type(sys)('PySide6.QtGui')

# Add mock classes to QtCore
sys.modules['PySide6.QtCore'].QObject = MockQObject
sys.modules['PySide6.QtCore'].Signal = MockSignal
sys.modules['PySide6.QtCore'].Slot = lambda *args: (lambda f: f)
sys.modules['PySide6.QtCore'].QTimer = MockQObject
sys.modules['PySide6.QtCore'].QEventLoop = MockQObject

# Now we can import app modules
from app.processors.models_processor import ModelsProcessor
from app.processors.face_detectors import FaceDetectors
from app.processors.face_swappers import FaceSwappers
from app.processors.face_editors import FaceEditors
from app.processors.face_restorers import FaceRestorers
from app.processors.frame_enhancers import FrameEnhancers


class VisoMasterGradio:
    """Web interface for VisoMaster face processing"""

    def __init__(self):
        print("Initializing VisoMaster Gradio...")

        # Initialize models processor (parent=None for no Qt)
        self.models_processor = ModelsProcessor(parent=None)

        # Initialize processor helpers
        self.face_detectors = FaceDetectors(self.models_processor)
        self.face_swappers = FaceSwappers(self.models_processor)
        self.face_editors = FaceEditors(self.models_processor)
        self.face_restorers = FaceRestorers(self.models_processor)
        self.frame_enhancers = FrameEnhancers(self.models_processor)

        # Default control settings
        self.control = {
            'ExecutionProviderSelection': 'CUDA' if torch.cuda.is_available() else 'CPU',
            'DetectorModelSelection': 'RetinaFace',
            'DetectorScoreSlider': 50,
            'MaxFacesToDetectSlider': 1,
            'SimilarityTypeSelection': 'Optimal',
            'RecognitionModelSelection': 'Inswapper128ArcFace',
            'LandmarkDetectToggle': False,
            'LandmarkDetectModelSelection': '5',
            'LandmarkDetectScoreSlider': 50,
            'DetectFromPointsToggle': False,
            'AutoRotationToggle': False,
            'ManualRotationEnableToggle': False,
            'ManualRotationAngleSlider': 0,
            'SwapOnlyBestMatchEnableToggle': True,
        }

        print(f"Using device: {self.models_processor.device}")
        print("Initialization complete!")

    def process_image_swap(
        self,
        target_image: np.ndarray,
        source_image: Optional[np.ndarray],
        swap_model: str,
        similarity_threshold: int,
        enable_restorer: bool,
        restorer_model: str,
    ) -> Tuple[np.ndarray, str]:
        """Process face swapping"""

        try:
            if target_image is None:
                return None, "Please upload a target image"

            if source_image is None:
                return None, "Please upload a source face image"

            status_messages = []

            # Convert images to torch tensors and resize if needed
            target_img = torch.from_numpy(target_image.astype('uint8')).to(self.models_processor.device)
            target_img = target_img.permute(2, 0, 1)  # HWC -> CHW

            source_img = torch.from_numpy(source_image.astype('uint8')).to(self.models_processor.device)
            source_img = source_img.permute(2, 0, 1)

            # Detect faces in target
            status_messages.append("Detecting faces in target image...")
            bboxes_target, kpss_5_target, kpss_target = self.face_detectors.run_detect(
                target_img,
                detect_mode=self.control['DetectorModelSelection'],
                max_num=self.control['MaxFacesToDetectSlider'],
                score=self.control['DetectorScoreSlider'] / 100.0,
                input_size=(512, 512),
                use_landmark_detection=False,
                landmark_detect_mode='5',
                landmark_score=0.5,
                from_points=False,
                rotation_angles=[0]
            )

            if len(kpss_5_target) == 0:
                return target_image, "No faces detected in target image"

            status_messages.append(f"Found {len(kpss_5_target)} face(s) in target")

            # Detect face in source
            status_messages.append("Detecting face in source image...")
            bboxes_source, kpss_5_source, kpss_source = self.face_detectors.run_detect(
                source_img,
                detect_mode=self.control['DetectorModelSelection'],
                max_num=1,
                score=self.control['DetectorScoreSlider'] / 100.0,
                input_size=(512, 512),
                use_landmark_detection=False,
                landmark_detect_mode='5',
                landmark_score=0.5,
                from_points=False,
                rotation_angles=[0]
            )

            if len(kpss_5_source) == 0:
                return target_image, "No face detected in source image"

            status_messages.append("Extracting source face embedding...")

            # Get source embedding
            source_kps = kpss_5_source[0]
            arcface_model = self.models_processor.get_arcface_model(swap_model)
            source_embedding, _ = self.face_swappers.run_recognize_direct(
                source_img,
                source_kps,
                similarity_type=self.control['SimilarityTypeSelection'],
                arcface_model=arcface_model
            )

            # Load swapper model if needed
            if not self.models_processor.models.get(swap_model):
                status_messages.append(f"Loading {swap_model} model...")
                self.models_processor.load_swapper_model(swap_model)

            # Process each detected face in target
            output_img = target_img.clone()

            for i, target_kps in enumerate(kpss_5_target):
                # Get target embedding
                target_embedding, _ = self.face_swappers.run_recognize_direct(
                    target_img,
                    target_kps,
                    similarity_type=self.control['SimilarityTypeSelection'],
                    arcface_model=arcface_model
                )

                # Check similarity
                similarity = self.models_processor.findCosineDistance(source_embedding, target_embedding)

                status_messages.append(f"Face {i+1} similarity: {similarity:.2f}")

                if similarity < similarity_threshold:
                    status_messages.append(f"Face {i+1} skipped (below threshold)")
                    continue

                # Create parameters for this swap
                params = {
                    'SwapModelSelection': swap_model,
                    'SwapperResSelection': '512',
                    'SimilarityThresholdSlider': similarity_threshold,
                    'FaceRestorerEnableToggle': enable_restorer,
                    'FaceRestorerTypeSelection': restorer_model,
                }

                # Perform swap
                status_messages.append(f"Swapping face {i+1}...")
                output_img = self.face_swappers.run_swap_simple(
                    output_img,
                    target_kps,
                    source_embedding,
                    self.models_processor.models[swap_model],
                    swap_model,
                    params
                )

            # Apply face restoration if enabled
            if enable_restorer:
                status_messages.append(f"Applying {restorer_model}...")

                if not self.models_processor.models.get(restorer_model):
                    status_messages.append(f"Loading {restorer_model}...")
                    self.models_processor.load_restorer_model(restorer_model)

                for target_kps in kpss_5_target:
                    output_img = self.face_restorers.apply_simple(
                        output_img,
                        target_kps,
                        self.models_processor.models[restorer_model],
                        restorer_model
                    )

            # Convert back to numpy
            output_img = output_img.permute(1, 2, 0).cpu().numpy()

            return output_img, " | ".join(status_messages)

        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return target_image if target_image is not None else None, error_msg

    def create_interface(self):
        """Create Gradio interface"""

        with gr.Blocks(title="VisoMaster - Face Processing Web Interface") as interface:
            gr.Markdown("# VisoMaster - Face Animation & Swapping")
            gr.Markdown("Web interface for LivePortrait-based face animation and swapping")

            with gr.Tabs():
                # Face Swapping Tab
                with gr.Tab("Face Swap"):
                    with gr.Row():
                        with gr.Column():
                            target_img_input = gr.Image(
                                label="Target Image",
                                type="numpy"
                            )
                            source_img_input = gr.Image(
                                label="Source Face",
                                type="numpy"
                            )

                        with gr.Column():
                            output_img = gr.Image(
                                label="Result",
                                type="numpy"
                            )
                            status_text = gr.Textbox(
                                label="Status",
                                lines=3
                            )

                    with gr.Row():
                        swap_model = gr.Dropdown(
                            choices=[
                                'Inswapper128',
                                'InStyleSwapper256 Version A',
                                'InStyleSwapper256 Version B',
                                'InStyleSwapper256 Version C',
                                'SimSwap512'
                            ],
                            value='Inswapper128',
                            label="Swapper Model"
                        )

                        similarity = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=58,
                            step=1,
                            label="Similarity Threshold"
                        )

                    with gr.Row():
                        enable_restorer = gr.Checkbox(
                            label="Enable Face Restoration",
                            value=False
                        )

                        restorer_model = gr.Dropdown(
                            choices=[
                                'GFPGAN v1.4',
                                'GFPGAN 1024',
                                'GPEN-BFR-256',
                                'GPEN-BFR-512',
                                'GPEN-BFR-1024',
                                'GPEN-BFR-2048'
                            ],
                            value='GFPGAN v1.4',
                            label="Restorer Model"
                        )

                    swap_btn = gr.Button("Swap Faces", variant="primary")

                    swap_btn.click(
                        fn=self.process_image_swap,
                        inputs=[
                            target_img_input,
                            source_img_input,
                            swap_model,
                            similarity,
                            enable_restorer,
                            restorer_model
                        ],
                        outputs=[output_img, status_text]
                    )

                # Info Tab
                with gr.Tab("Info"):
                    gr.Markdown(f"""
                    ## VisoMaster Web Interface

                    **Device:** {self.models_processor.device}
                    **CUDA Available:** {torch.cuda.is_available()}

                    ### Features
                    - Face Swapping with multiple models
                    - Face Restoration (GFPGAN, GPEN)
                    - GPU acceleration (when available)

                    ### Models Directory
                    Models are stored in: `model_assets/`

                    If models are missing, they will be auto-downloaded on first use.

                    ### Usage
                    1. Upload a target image containing the face(s) to modify
                    2. Upload a source face image
                    3. Select swapper model and adjust parameters
                    4. Click "Swap Faces" to process

                    ### Requirements
                    - NVIDIA GPU with CUDA support (recommended)
                    - At least 4GB VRAM for basic models
                    - 8GB+ VRAM for higher resolution processing
                    """)

        return interface


def main():
    """Main entry point"""

    print("="*60)
    print("VisoMaster - Gradio Web Interface")
    print("="*60)

    # Create app instance
    app = VisoMasterGradio()

    # Create and launch interface
    interface = app.create_interface()

    # Launch with public sharing disabled by default
    # Set share=True to create a public link
    interface.launch(
        server_name="0.0.0.0",  # Listen on all interfaces
        server_port=7860,
        share=False,  # Set to True for public link
        inbrowser=False,  # Don't auto-open browser
    )


if __name__ == "__main__":
    main()
