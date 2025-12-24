"""
Gradio-compatible wrapper for VisoMaster processing pipeline.
This module provides a simplified interface to the core processing functionality
without requiring Qt/PySide6 dependencies.
"""

import os
import numpy as np
import cv2
import torch
from typing import Dict, Optional, Tuple
from pathlib import Path
import copy

# Import core processors (these are Qt-independent)
from app.processors.models_processor import ModelsProcessor
from app.processors.face_detectors import FaceDetectors
from app.processors.face_swappers import FaceSwappers
from app.processors.face_editors import FaceEditors
from app.processors.face_restorers import FaceRestorers
from app.processors.frame_enhancers import FrameEnhancers
from app.helpers.miscellaneous import ParametersDict


class GradioProcessor:
    """
    Simplified processor for Gradio interface that mimics the Qt application's
    processing pipeline without UI dependencies.
    """

    def __init__(self):
        # Initialize models processor (without Qt parent)
        self.models_processor = ModelsProcessor(parent=None)

        # Default parameters matching the Qt app defaults
        self.default_parameters = self._get_default_parameters()

        # Current processing parameters
        self.parameters = {}
        self.control = {}

        # Initialize with defaults
        self.reset_parameters()

    def _get_default_parameters(self) -> Dict:
        """Get default parameters matching the Qt application."""
        return {
            # Face Swap parameters
            'SwapModelSelection': 'Inswapper128',
            'SwapperResSelection': '512',
            'SwapperResAutoSelectEnableToggle': False,
            'SimilarityThresholdSlider': 58,
            'FaceSwapStrengthMultiplierDecimalSlider': 1.0,
            'FaceLikenessDecimalSlider': 0.0,

            # Face Editor parameters
            'FaceEditorEnableToggle': False,
            'FaceEditorTypeSelection': 'Human-Face',
            'FaceEditorCropScaleDecimalSlider': 2.50,
            'FaceEditorVYRatioDecimalSlider': -0.125,
            'EyesOpenRatioDecimalSlider': 0.0,
            'LipsOpenRatioDecimalSlider': 0.0,
            'HeadPitchSlider': 0,
            'HeadYawSlider': 0,
            'HeadRollSlider': 0,
            'XAxisSlider': 0.0,
            'YAxisSlider': 0.0,
            'ZAxisSlider': 0.0,

            # Face Restorer parameters
            'FaceRestorerEnableToggle': False,
            'FaceRestorerTypeSelection': 'GFPGAN v1.4',
            'FaceRestorerBlendDecimalSlider': 0.5,

            # Frame Enhancer parameters
            'FrameEnhancerEnableToggle': False,
            'FrameEnhancerTypeSelection': 'RealESRGAN x2',

            # Detection parameters
            'FaceDetectorTypeSelection': 'RetinaFace',
            'FaceDetectionScoreSlider': 50,
            'MaxFacesSlider': 1,
        }

    def reset_parameters(self):
        """Reset parameters to defaults."""
        self.parameters = {0: copy.deepcopy(self.default_parameters)}
        self.control = {
            'ExecutionProviderSelection': 'CUDA',
            'ExecutionThreadsSlider': 2,
            'FrameEnhancerEnableToggle': False,
        }

    def update_parameter(self, face_id: int, param_name: str, value):
        """Update a specific parameter for a face."""
        if face_id not in self.parameters:
            self.parameters[face_id] = copy.deepcopy(self.default_parameters)
        self.parameters[face_id][param_name] = value

    def process_image(
        self,
        target_image: np.ndarray,
        source_face_image: Optional[np.ndarray] = None,
        enable_swap: bool = False,
        enable_edit: bool = False,
        swap_model: str = 'Inswapper128',
        similarity_threshold: int = 58,
        pitch: int = 0,
        yaw: int = 0,
        roll: int = 0,
        eyes_open: float = 0.0,
        lips_open: float = 0.0,
        enable_restorer: bool = False,
        restorer_type: str = 'GFPGAN v1.4',
    ) -> Tuple[np.ndarray, str]:
        """
        Process a single image with face swapping and/or editing.

        Args:
            target_image: Input image as numpy array (RGB)
            source_face_image: Source face for swapping (RGB), optional
            enable_swap: Enable face swapping
            enable_edit: Enable face editing (pose/expression)
            swap_model: Swapper model to use
            similarity_threshold: Similarity threshold for face matching
            pitch, yaw, roll: Head pose adjustments
            eyes_open, lips_open: Expression adjustments
            enable_restorer: Enable face restoration
            restorer_type: Face restorer model

        Returns:
            Tuple of (processed_image, status_message)
        """
        try:
            # Update parameters
            face_id = 0
            self.update_parameter(face_id, 'SwapModelSelection', swap_model)
            self.update_parameter(face_id, 'SimilarityThresholdSlider', similarity_threshold)
            self.update_parameter(face_id, 'HeadPitchSlider', pitch)
            self.update_parameter(face_id, 'HeadYawSlider', yaw)
            self.update_parameter(face_id, 'HeadRollSlider', roll)
            self.update_parameter(face_id, 'EyesOpenRatioDecimalSlider', eyes_open)
            self.update_parameter(face_id, 'LipsOpenRatioDecimalSlider', lips_open)
            self.update_parameter(face_id, 'FaceRestorerEnableToggle', enable_restorer)
            self.update_parameter(face_id, 'FaceRestorerTypeSelection', restorer_type)
            self.update_parameter(face_id, 'FaceEditorEnableToggle', enable_edit)

            # Ensure image is RGB and numpy array
            if isinstance(target_image, str):
                target_image = cv2.imread(target_image)
                target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

            # Convert to numpy if needed
            if not isinstance(target_image, np.ndarray):
                target_image = np.array(target_image)

            # Ensure RGB format
            if target_image.shape[2] == 4:  # RGBA
                target_image = target_image[:, :, :3]

            output_frame = target_image.copy()

            # Step 1: Face Detection
            detector_type = self.parameters[face_id].get('FaceDetectorTypeSelection', 'RetinaFace')
            detection_score = self.parameters[face_id].get('FaceDetectionScoreSlider', 50) / 100.0
            max_faces = self.parameters[face_id].get('MaxFacesSlider', 1)

            # Detect faces in target image
            face_detectors = FaceDetectors()
            detected_faces = face_detectors.run_detect(
                output_frame,
                detector_type,
                detection_score,
                max_faces
            )

            if not detected_faces:
                return target_image, "No faces detected in target image"

            status_parts = [f"Detected {len(detected_faces)} face(s)"]

            # Step 2: Face Swapping (if enabled and source provided)
            if enable_swap and source_face_image is not None:
                # Load swapper model
                swapper_model_name = swap_model
                if not self.models_processor.models.get(swapper_model_name):
                    status_parts.append(f"Loading {swapper_model_name}...")
                    self.models_processor.load_swapper_model(swapper_model_name)

                # Extract source face embedding
                if isinstance(source_face_image, str):
                    source_face_image = cv2.imread(source_face_image)
                    source_face_image = cv2.cvtColor(source_face_image, cv2.COLOR_BGR2RGB)

                if not isinstance(source_face_image, np.ndarray):
                    source_face_image = np.array(source_face_image)

                # Detect face in source image
                source_faces = face_detectors.run_detect(
                    source_face_image,
                    detector_type,
                    detection_score,
                    1  # Only need one face from source
                )

                if not source_faces:
                    return target_image, "No face detected in source image"

                source_face = source_faces[0]

                # Swap faces
                face_swappers = FaceSwappers()
                for i, target_face in enumerate(detected_faces):
                    output_frame = face_swappers.run_swap(
                        source_face,
                        target_face,
                        output_frame,
                        self.models_processor.models[swapper_model_name],
                        swapper_model_name,
                        self.parameters[face_id]
                    )

                status_parts.append("Face swapping complete")

            # Step 3: Face Editing (if enabled)
            if enable_edit:
                face_editors = FaceEditors()

                # Load LivePortrait models if needed
                required_models = [
                    'MotionExtractor_Human',
                    'AppearanceFeatureExtractor',
                    'WarpingModule',
                    'StitchingModule_Eye',
                    'StitchingModule_Lip'
                ]

                for model_name in required_models:
                    if not self.models_processor.models.get(model_name):
                        status_parts.append(f"Loading {model_name}...")
                        self.models_processor.load_model(model_name)

                # Apply face editing to each detected face
                for target_face in detected_faces:
                    # Extract motion features
                    motion = face_editors.lp_motion_extractor(
                        output_frame,
                        target_face,
                        self.models_processor.models,
                        self.parameters[face_id].get('FaceEditorTypeSelection', 'Human-Face')
                    )

                    # Modify motion based on parameters
                    if motion is not None:
                        # Apply pose adjustments
                        motion['pitch'] += pitch
                        motion['yaw'] += yaw
                        motion['roll'] += roll

                        # Apply expression adjustments
                        if 'expression' in motion and motion['expression'] is not None:
                            # Eyes open/close
                            if abs(eyes_open) > 0.01:
                                motion['expression'][0] += eyes_open
                            # Lips open/close
                            if abs(lips_open) > 0.01:
                                motion['expression'][1] += lips_open

                        # Apply editing
                        output_frame = face_editors.apply_editing(
                            output_frame,
                            target_face,
                            motion,
                            self.models_processor.models,
                            self.parameters[face_id]
                        )

                status_parts.append("Face editing complete")

            # Step 4: Face Restoration (if enabled)
            if enable_restorer:
                restorer_model_name = restorer_type
                if not self.models_processor.models.get(restorer_model_name):
                    status_parts.append(f"Loading {restorer_model_name}...")
                    self.models_processor.load_restorer_model(restorer_model_name)

                face_restorers = FaceRestorers()
                for target_face in detected_faces:
                    output_frame = face_restorers.apply_facerestorer(
                        output_frame,
                        target_face,
                        self.models_processor.models[restorer_model_name],
                        restorer_model_name,
                        self.parameters[face_id]
                    )

                status_parts.append("Face restoration complete")

            return output_frame, " | ".join(status_parts)

        except Exception as e:
            import traceback
            error_msg = f"Error processing image: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return target_image, error_msg

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'models_processor'):
            # Unload models
            for model_name in list(self.models_processor.models.keys()):
                self.models_processor.models[model_name] = None

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
