from app.ui.widgets.actions import control_actions
import cv2
from app.helpers.typing_helper import LayoutDictTypes
SETTINGS_LAYOUT_DATA: LayoutDictTypes = {
    'Appearance': {
        'ThemeSelection': {
            'level': 1,
            'label': 'Theme',
            'options': ['Dark', 'Dark-Blue', 'Light'],
            'default': 'Dark',
            'help': 'Select the theme to be used',
            'exec_function': control_actions.change_theme,
            'exec_function_args': [],
        },
    },
    'General': {
        'ProvidersPrioritySelection': {
            'level': 1,
            'label': 'Providers Priority',
            'options': ['CUDA', 'TensorRT', 'TensorRT-Engine', 'CPU'],
            'default': 'TensorRT-Engine',
            'help': 'Select the providers priority to be used with the system.',
            'exec_function': control_actions.change_execution_provider,
            'exec_function_args': [],
        },
        'nThreadsSlider': {
            'level': 1,
            'label': 'Number of Threads',
            'min_value': '1',
            'max_value': '30',
            'default': '1',
            'step': 1,
            'help': 'Set number of execution threads while playing and recording. Depends strongly on GPU VRAM.',
            'exec_function': control_actions.change_threads_number,
            'exec_function_args': [],
        },
    },
    'Video Settings': {
        'VideoPlaybackCustomFpsToggle': {
            'level': 1,
            'label': 'Set Custom Video Playback FPS',
            'default': False,
            'help': 'Manually set the FPS to be used when playing the video',
            'exec_function': control_actions.set_video_playback_fps,
            'exec_function_args': [],
        },
        'VideoPlaybackCustomFpsSlider': {
            'level': 2,
            'label': 'Video Playback FPS',
            'min_value': '1',
            'max_value': '120',
            'default': '30',
            'parentToggle': 'VideoPlaybackCustomFpsToggle',
            'requiredToggleValue': True,
            'step': 1,
            'help': 'Set the maximum FPS of the video when playing'
        },
    },
    'Auto Swap':{
        'AutoSwapToggle': {
            'level': 1,
            'label': 'Auto Swap',
            'default': False,
            'help': 'Automatically Swap all faces using selected Source Faces/Embeddings when loading an video/image file'
        },
    },
    'Best Swap': {
        'SwapOnlyBestMatchEnableToggle': {
            'level': 3,
            'label': 'Swap only best match',
            'default': False,           
            'help': 'only swap highest face match per face (not every match above treshold)'
        }
    },
    'Detectors': {
        'DetectorModelSelection': {
            'level': 1,
            'label': 'Face Detect Model',
            'options': ['RetinaFace', 'Yolov8', 'SCRFD'],
            'default': 'RetinaFace',
            'help': 'Select the face detection model to use for detecting faces in the input image or video.'
        },
        'DetectorScoreSlider': {
            'level': 1,
            'label': 'Detect Score',
            'min_value': '1',
            'max_value': '100',
            'default': '48',
            'step': 1,
            'help': 'Set the confidence score threshold for face detection. Higher values ensure more confident detections but may miss some faces.'
        },
        'MaxFacesToDetectSlider': {
            'level': 1,
            'label': 'Max No of Faces to Detect',
            'min_value': '1',
            'max_value': '50',
            'default': '2',
            'step': 1,     
            'help': 'Set the maximum number of faces to detect in a frame'
   
        },
        'AutoRotationToggle': {
            'level': 1,
            'label': 'Auto Rotation',
            'default': False,
            'help': 'Automatically rotate the input to detect faces in various orientations.'
        },
        'ManualRotationEnableToggle': {
            'level': 1,
            'label': 'Manual Rotation',
            'default': False,
            'help': 'Rotate the face detector to better detect faces at different angles.'
        },
        'ManualRotationAngleSlider': {
            'level': 2,
            'label': 'Rotation Angle',
            'min_value': '0',
            'max_value': '270',
            'default': '0',
            'step': 90,
            'parentToggle': 'ManualRotationEnableToggle',
            'requiredToggleValue': True,
            'help': 'Set this to the angle of the input face angle to help with laying down/upside down/etc. Angles are read clockwise.'
        },
        'LandmarkDetectToggle': {
            'level': 1,
            'label': 'Enable Landmark Detection',
            'default': False,
            'help': 'Enable or disable facial landmark detection, which is used to refine face alignment.'
        },
        'LandmarkDetectModelSelection': {
            'level': 2,
            'label': 'Landmark Detect Model',
            'options': ['5', '68', '3d68', '98', '106', '203', '478'],
            'default': '203',
            'parentToggle': 'LandmarkDetectToggle',
            'requiredToggleValue': True,
            'help': 'Select the landmark detection model, where different models detect varying numbers of facial landmarks.'
        },
        'LandmarkDetectScoreSlider': {
            'level': 2,
            'label': 'Landmark Detect Score',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'LandmarkDetectToggle',
            'requiredToggleValue': True,
            'help': 'Set the confidence score threshold for facial landmark detection.'
        },
        'DetectFromPointsToggle': {
            'level': 2,
            'label': 'Detect From Points',
            'default': False,
            'parentToggle': 'LandmarkDetectToggle',
            'requiredToggleValue': True,
            'help': 'Enable detection of faces from specified landmark points.'
        },
        'ShowLandmarksEnableToggle': {
            'level': 1,
            'label': 'Show Landmarks',
            'default': False,
            'help': 'Show Landmarks in realtime.'
        },
        'ShowAllDetectedFacesBBoxToggle': {
            'level': 1,
            'label': 'Show Bounding Boxes',
            'default': False,
            'help': 'Draw bounding boxes to all detected faces in the frame'
        }
    },
    'DFM Settings':{
        'MaxDFMModelsSlider':{
            'level': 1,
            'label': 'Maximum DFM Models to use',
            'min_value': '1',
            'max_value': '5',
            'default': '1',
            'step': 1,
            'help': "Set the maximum number of DFM Models to keep in memory at a time. Set this based on your GPU's VRAM",
        }
    },
    'Frame Enhancer':{
        'FrameEnhancerEnableToggle':{
            'level': 1,
            'label': 'Enable Frame Enhancer',
            'default': False,
            'help': 'Enable frame enhancement for video inputs to improve visual quality.'
        },
        'FrameEnhancerTypeSelection':{
            'level': 2,
            'label': 'Frame Enhancer Type',
            'options': ['RealEsrgan-x2-Plus', 'RealEsrgan-x4-Plus', 'RealEsr-General-x4v3', 'BSRGan-x2', 'BSRGan-x4', 'UltraSharp-x4', 'UltraMix-x4', 'DDColor-Artistic', 'DDColor', 'DeOldify-Artistic', 'DeOldify-Stable', 'DeOldify-Video'],
            'default': 'RealEsrgan-x2-Plus',
            'parentToggle': 'FrameEnhancerEnableToggle',
            'requiredToggleValue': True,
            'help': 'Select the type of frame enhancement to apply, based on the content and resolution requirements.'
        },
        'FrameEnhancerBlendSlider': {
            'level': 2,
            'label': 'Blend',
            'min_value': '0',
            'max_value': '100',
            'default': '100',
            'step': 1,
            'parentToggle': 'FrameEnhancerEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blends the enhanced results back into the original frame.'
        },
        'FrameEnhancerDownToggle':{
            'level': 1,
            'label': 'Frame Downscale to 1920*1080',
            'default': False,
            #'parentToggle': 'FrameEnhancerEnableToggle',
            #'requiredToggleValue': True,
            'help': 'Select to downscale the video to 1920*1080'
        },
    },
    'Face Recognition': {
        'RecognitionModelSelection': {
            'level': 1,
            'label': 'Recognition Model',
            'options': ['Inswapper128ArcFace', 'SimSwapArcFace'],
            'default': 'Inswapper128ArcFace',
            'help': 'Choose the ArcFace model to be used for comparing the similarity of faces.'
        },
        'SimilarityTypeSelection': {
            'level': 1,
            'label': 'Swapping Similarity Type',
            'options': ['Opal', 'Pearl', 'Optimal'],
            'default': 'Opal',
            'help': 'Choose the type of similarity calculation for face detection and matching during the face swapping process.'
        },
    },
    'Embedding Merge Method':{
        'EmbMergeMethodSelection':{
            'level': 1,
            'label': 'Embedding Merge Method',
            'options': ['Mean','Median'],
            'default': 'Mean',
            'help': 'Select the method to merge facial embeddings. "Mean" averages the embeddings, while "Median" selects the middle value, providing more robustness to outliers.'
        }
    },
    'Media Selection':{
        'TargetMediaFolderRecursiveToggle':{
            'level': 1,
            'label': 'Target Media Include Subfolders',
            'default': False,
            'help': 'Include all files from Subfolders when choosing Target Media Folder'
        },
        'InputFacesFolderRecursiveToggle':{
            'level': 1,
            'label': 'Input Faces Include Subfolders',
            'default': False,
            'help': 'Include all files from Subfolders when choosing Input Faces Folder'
        }
    }
}
