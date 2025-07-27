from app.helpers.typing_helper import LayoutDictTypes
import app.ui.widgets.actions.layout_actions as layout_actions
import app.ui.widgets.actions.control_actions as control_actions

COMMON_LAYOUT_DATA: LayoutDictTypes = {
    'Face Restorer': {
        'FaceRestorerEnableToggle': {
            'level': 1,
            'label': 'Enable Face Restorer',
            'default': False,
            'help': 'Enable the use of a face restoration model to improve the quality of the face after swapping.'
        },
        'FaceRestorerTypeSelection': {
            'level': 2,
            'label': 'Restorer Type',
            'options': ['GFPGAN-v1.4','GFPGAN-1024', 'GPEN-256', 'GPEN-512', 'GPEN-1024', 'GPEN-2048'],
            'default': 'GFPGAN-1024',
            'parentToggle': 'FaceRestorerEnableToggle',
            'requiredToggleValue': True,
            'help': 'Select the model type for face restoration.'
        },
        'FaceRestorerDetTypeSelection': {
            'level': 2,
            'label': 'Alignment',
            'options': ['Original', 'Blend', 'Reference'],
            'default': 'Original',
            'parentToggle': 'FaceRestorerEnableToggle',
            'requiredToggleValue': True,
            'help': 'Select the alignment method for restoring the face to its original or blended position.'
        },
        'FaceRestorerBlendSlider': {
            'level': 2,
            'label': 'Blend',
            'min_value': '0',
            'max_value': '100',
            'default': '100',
            'step': 1,
            'parentToggle': 'FaceRestorerEnableToggle',
            'requiredToggleValue': True,
            'help': 'Control the blend ratio between the restored face and the swapped face.'
        },
        'FaceRestorerEnable2Toggle': {
            'level': 1,
            'label': 'Enable Face Restorer 2',
            'default': False,
            'help': 'Enable the use of a face restoration model to improve the quality of the face after swapping.'
        },
        'FaceRestorerType2Selection': {
            'level': 2,
            'label': 'Restorer Type',
            'options': ['GFPGAN-v1.4', 'GFPGAN-1024', 'GPEN-256', 'GPEN-512', 'GPEN-1024', 'GPEN-2048'],
            'default': 'GPEN-2048',
            'parentToggle': 'FaceRestorerEnable2Toggle',
            'requiredToggleValue': True,
            'help': 'Select the model type for face restoration.'
        },
        'FaceRestorerDetType2Selection': {
            'level': 2,
            'label': 'Alignment',
            'options': ['Original', 'Blend', 'Reference'],
            'default': 'Original',
            'parentToggle': 'FaceRestorerEnable2Toggle',
            'requiredToggleValue': True,
            'help': 'Select the alignment method for restoring the face to its original or blended position.'
        },
        'FaceRestorerBlend2Slider': {
            'level': 2,
            'label': 'Blend',
            'min_value': '0',
            'max_value': '100',
            'default': '100',
            'step': 1,
            'parentToggle': 'FaceRestorerEnable2Toggle',
            'requiredToggleValue': True,
            'help': 'Control the blend ratio between the restored face and the swapped face.'
        },
        'FaceExpressionEnableToggleBoth': {
            'level': 1,
            'label': 'Enable Face Expression Restorer',
            'default': False,
            'help': 'Enabled the use of the LivePortrait face expression model to restore facial expressions after swapping.'
        },
        'FaceExpressionCropScaleDecimalSliderBoth': {
            'level': 2,
            'label': 'Crop Scale',
            'min_value': '2.0',
            'max_value': '3.0',
            'default': '2.5',
            'step': 0.1,
            'decimals': 1,
            'parentToggle': 'FaceExpressionEnableToggleBoth',
            'requiredToggleValue': True,
            'help': 'Changes swap crop scale. Increase the value to capture the face more distantly.'
        },
        'FaceExpressionVYRatioDecimalSliderBoth': {
            'level': 2,
            'label': 'VY Ratio',
            'min_value': '-0.125',
            'max_value': '-0.100',
            'default': '-0.100',
            'step': 0.001,
            'decimals': 3,
            'parentToggle': 'FaceExpressionEnableToggleBoth',
            'requiredToggleValue': True,
            'help': 'Changes the vy ratio for crop scale. Increase the value to capture the face more distantly.'
        },
        'FaceExpressionEyesToggle': {
            'level': 2,
            'label': 'Restore the eyes',
            'default': False,
            'parentToggle': 'FaceExpressionEnableToggleBoth',
            'requiredToggleValue': True,
            'help': 'Activate the eyes face expression restorer'
        },
        'FaceExpressionFriendlyFactorDecimalSliderEyes': {
            'level': 3,
            'label': 'Expression Friendly Factor Eyes',
            'min_value': '0.0',
            'max_value': '1.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'FaceExpressionEnableToggleBoth & FaceExpressionEyesToggle',
            'requiredToggleValue': True,
            'help': 'Control the expression similarity between the driving face and the swapped face of the eyes.'
        },
        'FaceExpressionRetargetingEyesEnableToggleBoth': {
            'level': 3,
            'label': 'Retargeting Eyes',
            'default': False,
            'parentToggle': 'FaceExpressionEnableToggleBoth & FaceExpressionEyesToggle',
            'requiredToggleValue': True,
            'help': 'Adjusting or redirecting the gaze or movement of the eyes during the facial restoration process.'
        },
        'FaceExpressionRetargetingEyesMultiplierDecimalSliderBoth': {
            'level': 4,
            'label': 'Retargeting Eyes Multiplier',
            'min_value': '0.0',
            'max_value': '2.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'FaceExpressionRetargetingEyesEnableToggleBoth & FaceExpressionEnableToggleBoth & FaceExpressionEyesToggle',
            'requiredToggleValue': True,
            'help': 'Multiplier value for Retargeting Eyes.'
        },
        'FaceExpressionNormalizeEyesEnableToggleBoth': {
            'level': 4,
            'label': 'Normalize Eyes',
            'default': True,
            'parentToggle': 'FaceExpressionEnableToggleBoth & FaceExpressionEyesToggle & FaceExpressionRetargetingEyesEnableToggleBoth',
            'requiredToggleValue': True,
            'help': 'Normalize the Eyes during the facial restoration process.'
        },
        'FaceExpressionNormalizeEyesThresholdDecimalSliderBoth': {
            'level': 5,
            'label': 'Normalize Eyes Threshold',
            'min_value': '0.00',
            'max_value': '1.00',
            'default': '0.40',
            'decimals': 2,
            'step': 0.01,
            'parentToggle': 'FaceExpressionNormalizeEyesEnableToggleBoth & FaceExpressionEnableToggleBoth & FaceExpressionEyesToggle & FaceExpressionRetargetingEyesEnableToggleBoth',
            'requiredToggleValue': True,
            'help': 'Threshold value for Normalize Eyes.'
        },     
        'FaceExpressionLipsToggle': {
            'level': 2,
            'label': 'Restore the lips',
            'default': False,
            'parentToggle': 'FaceExpressionEnableToggleBoth',
            'requiredToggleValue': True,
            'help': 'Activate the lips face expression restorer'
        },
        'FaceExpressionFriendlyFactorDecimalSliderLips': {
            'level': 3,
            'label': 'Expression Friendly Factor Lips',
            'min_value': '0.0',
            'max_value': '1.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'FaceExpressionEnableToggleBoth & FaceExpressionLipsToggle',
            'requiredToggleValue': True,
            'help': 'Control the expression similarity between the driving face and the swapped face of the lips.'
        },
        'FaceExpressionNormalizeLipsEnableToggleBoth': {
            'level': 3,
            'label': 'Normalize Lips',
            'default': True,
            'parentToggle': 'FaceExpressionEnableToggleBoth & FaceExpressionLipsToggle',
            'requiredToggleValue': True,
            'help': 'Normalize the lips during the facial restoration process.'
        },
        'FaceExpressionNormalizeLipsThresholdDecimalSliderBoth': {
            'level': 4,
            'label': 'Normalize Lips Threshold',
            'min_value': '0.00',
            'max_value': '0.20',
            'default': '0.03',
            'decimals': 2,
            'step': 0.01,
            'parentToggle': 'FaceExpressionNormalizeLipsEnableToggleBoth & FaceExpressionEnableToggleBoth & FaceExpressionLipsToggle',
            'requiredToggleValue': True,
            'help': 'Threshold value for Normalize Lips.'
        },        
        'FaceExpressionRetargetingLipsEnableToggleBoth': {
            'level': 3,
            'label': 'Retargeting Lips',
            'default': False,
            'parentToggle': 'FaceExpressionEnableToggleBoth & FaceExpressionLipsToggle',
            'requiredToggleValue': True,
            'help': 'Adjusting or modifying the position, shape, or movement of the lips during the facial restoration process.'
        },
        'FaceExpressionRetargetingLipsMultiplierDecimalSliderBoth': {
            'level': 4,
            'label': 'Retargeting Lips Multiplier',
            'min_value': '0.0',
            'max_value': '2.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'FaceExpressionRetargetingLipsEnableToggleBoth & FaceExpressionEnableToggleBoth & FaceExpressionLipsToggle',
            'requiredToggleValue': True,
            'help': 'Multiplier value for Retargeting Lips.'
        },
    },
    'ReF-LDM Denoiser': {
        'ReferenceKVTensorsSelection': {
            'level': 1, # Or your desired layout level
            'widget_type': 'SelectionBox',
            'label': 'Reference K/V Tensors',
            'control_name': 'ReferenceKVTensorsSelection',
            'options': [], # Will be populated by _populate_reference_kv_tensors
            'default': "", # Or a default filename if applicable
            # Add any 'condition_control' or 'parentToggle' if needed
            'help': 'Select a Reference K/V Tensor file (*.pt). Files must be in "model_assets/reference_kv_data/".',
            'exec_function': lambda mw, val: mw.handle_reference_kv_file_change(val), # Trigger loading on UI change
            'exec_function_args': [] # No extra args needed
        },
        'UseReferenceExclusivePathToggle': { # New ToggleButton
            'level': 1,
            'widget_type': 'ToggleButton',
            'label': 'Exclusive Reference Path',
            'control_name': 'UseReferenceExclusivePathToggle',
            'default': True,
            'help': 'If enabled, forces the UNet to use only reference K/V for attention, maximizing focus on the reference features.'
        },
        'DenoiserBaseSeedSlider': {
            'level': 1,
            'widget_type': 'ParameterSlider',
            'label': 'Base Seed',
            'control_name': 'DenoiserBaseSeedSlider',
            'min_value': '0', 'max_value': '300', 'default': '0', 'step': 1,
            'help': 'Set a fixed base seed for the denoiser. This seed will be used for all frames and both denoiser passes (if applicable) to ensure consistent noise patterns.'
        },
        'DenoiserUNetEnableBeforeRestorersToggle': {
            'level': 1,
            'widget_type': 'ToggleButton',
            'label': 'Enable Denoiser before Restorers',
            'control_name': 'DenoiserUNetEnableBeforeRestorersToggle',
            'default': False,
            'help': 'Enable UNet-based image denoising. This is applied to the 512x512 aligned/swapped face before other restorers.',
            'exec_function': control_actions.handle_denoiser_state_change,
            'exec_function_args': ['DenoiserUNetEnableBeforeRestorersToggle'],
        },
        'DenoiserModeSelectionBefore': {
            'level': 2,
            'widget_type': 'SelectionBox',
            'label': 'Denoiser Mode (Before)',
            'control_name': 'DenoiserModeSelectionBefore',
            'options': ["Single Step (Fast)", "Full Restore (DDIM)"],
            'default': "Full Restore (DDIM)",
            'parentToggle': 'DenoiserUNetEnableBeforeRestorersToggle',
            'requiredToggleValue': True,
            'help': 'Denoising mode for the pass before restorers. Single Step is generally faster.'
        },
        'DenoiserSingleStepTimestepSliderBefore': {
            'level': 3,
            'widget_type': 'ParameterSlider',
            'label': 'Single Step Timestep (t) (Before)',
            'control_name': 'DenoiserSingleStepTimestepSliderBefore',
            'min_value': '0', 'max_value': '500', 'default': '500', 'step': 1, # Max value was 200, can be higher for single step
            'parentToggle': 'DenoiserUNetEnableBeforeRestorersToggle',
            'requiredToggleValue': True,
            'parentSelection': 'DenoiserModeSelectionBefore',
            'requiredSelectionValue': "Single Step (Fast)",
            'help': 'Timestep for single-step denoising (Before Restorers). Lower values mean less noise added/removed.'
        },
        'DenoiserDDIMStepsSliderBefore': {
            'level': 3,
            'widget_type': 'ParameterSlider',
            'label': 'DDIM Steps (Before)',
            'control_name': 'DenoiserDDIMStepsSliderBefore',
            'min_value': '5', 'max_value': '50', 'default': '5', 'step': 1,
            'parentToggle': 'DenoiserUNetEnableBeforeRestorersToggle',
            'requiredToggleValue': True,
            'parentSelection': 'DenoiserModeSelectionBefore',
            'requiredSelectionValue': "Full Restore (DDIM)",
            'help': "Number of DDIM steps for full restoration (Before Restorers). Higher = more detail, slower."
        },
        'DenoiserCFGScaleDecimalSliderBefore': {
            'level': 3,
            'widget_type': 'ParameterDecimalSlider',
            'label': 'CFG Scale (Before)',
            'control_name': 'DenoiserCFGScaleDecimalSliderBefore',
            'min_value': '0.0', 'max_value': '10.0', 'default': '1.0', 'step': 0.1, 'decimals': 1,
            'parentToggle': 'DenoiserUNetEnableBeforeRestorersToggle',
            'requiredToggleValue': True,
            'parentSelection': 'DenoiserModeSelectionBefore',
            'requiredSelectionValue': "Full Restore (DDIM)",
            'help': "Classifier-Free Guidance scale for DDIM (Before Restorers). Higher = stronger adherence to K/V."
        },
        'DenoiserAfterFirstRestorerToggle': {
            'level': 1,
            'widget_type': 'ToggleButton',
            'label': 'Enable Denoiser After First Restorer',
            'control_name': 'DenoiserAfterFirstRestorerToggle',
            'default': False,
            'help': 'Apply the UNet Denoiser again after the first face restorer has been applied. Uses the same UNet model and step settings.',
            'exec_function': control_actions.handle_denoiser_state_change,
            'exec_function_args': ['DenoiserAfterFirstRestorerToggle'],
        },
        'DenoiserModeSelectionAfterFirst': {
            'level': 2,
            'widget_type': 'SelectionBox',
            'label': 'Denoiser Mode (After)',
            'control_name': 'DenoiserModeSelectionAfterFirst',
            'options': ["Single Step (Fast)", "Full Restore (DDIM)"],
            'default': "Single Step (Fast)",
            'parentToggle': 'DenoiserAfterFirstRestorerToggle',
            'requiredToggleValue': True,
            'help': 'Denoising mode for the pass after first restorer. Single Step is generally faster.'
        },
        'DenoiserSingleStepTimestepSliderAfterFirst': {
            'level': 3,
            'widget_type': 'ParameterSlider',
            'label': 'Single Step Timestep (t) (After)',
            'control_name': 'DenoiserSingleStepTimestepSliderAfterFirst',
            'min_value': '0', 'max_value': '500', 'default': '500', 'step': 1, # Max value was 200
            'parentToggle': 'DenoiserAfterFirstRestorerToggle',
            'requiredToggleValue': True,
            'parentSelection': 'DenoiserModeSelectionAfterFirst',
            'requiredSelectionValue': "Single Step (Fast)",
            'help': 'Timestep for single-step denoising (After first Restorer). Lower values mean less noise added/removed.'
        },
        'DenoiserDDIMStepsSliderAfterFirst': {
            'level': 3,
            'widget_type': 'ParameterSlider',
            'label': 'DDIM Steps (After First)',
            'control_name': 'DenoiserDDIMStepsSliderAfterFirst',
            'min_value': '5', 'max_value': '50', 'default': '5', 'step': 1,
            'parentToggle': 'DenoiserAfterFirstRestorerToggle',
            'requiredToggleValue': True,
            'parentSelection': 'DenoiserModeSelectionAfterFirst',
            'requiredSelectionValue': "Full Restore (DDIM)",
            'help': "Number of DDIM steps for full restoration (After First Restorer). Higher = more detail, slower."
        },
        'DenoiserCFGScaleDecimalSliderAfterFirst': {
            'level': 3,
            'widget_type': 'ParameterDecimalSlider',
            'label': 'CFG Scale (After First)',
            'control_name': 'DenoiserCFGScaleDecimalSliderAfterFirst',
            'min_value': '0.0', 'max_value': '10.0', 'default': '1.0', 'step': 0.1, 'decimals': 1,
            'parentToggle': 'DenoiserAfterFirstRestorerToggle',
            'requiredToggleValue': True,
            'parentSelection': 'DenoiserModeSelectionAfterFirst',
            'requiredSelectionValue': "Full Restore (DDIM)",
            'help': "Classifier-Free Guidance scale for DDIM (After First Restorer). Higher = stronger adherence to K/V."
        },
        'DenoiserAfterRestorersToggle': {
            'level': 1,
            'widget_type': 'ToggleButton',
            'label': 'Enable Denoiser After Restorers',
            'control_name': 'DenoiserAfterRestorersToggle',
            'default': False,
            'help': 'Apply the UNet Denoiser again after face restorers have been applied. Uses the same UNet model and step settings.',
            'exec_function': control_actions.handle_denoiser_state_change,
            'exec_function_args': ['DenoiserAfterRestorerToggle'],
        },
        'DenoiserModeSelectionAfter': {
            'level': 2,
            'widget_type': 'SelectionBox',
            'label': 'Denoiser Mode (After)',
            'control_name': 'DenoiserModeSelectionAfter',
            'options': ["Single Step (Fast)", "Full Restore (DDIM)"],
            'default': "Single Step (Fast)",
            'parentToggle': 'DenoiserAfterRestorersToggle',
            'requiredToggleValue': True,
            'help': 'Denoising mode for the pass after restorers. Single Step is generally faster.'
        },
        'DenoiserSingleStepTimestepSliderAfter': {
            'level': 3,
            'widget_type': 'ParameterSlider',
            'label': 'Single Step Timestep (t) (After)',
            'control_name': 'DenoiserSingleStepTimestepSliderAfter',
            'min_value': '0', 'max_value': '500', 'default': '500', 'step': 1, # Max value was 200
            'parentToggle': 'DenoiserAfterRestorersToggle',
            'requiredToggleValue': True,
            'parentSelection': 'DenoiserModeSelectionAfter',
            'requiredSelectionValue': "Single Step (Fast)",
            'help': 'Timestep for single-step denoising (After Restorers). Lower values mean less noise added/removed.'
        },
        'DenoiserDDIMStepsSliderAfter': {
            'level': 3,
            'widget_type': 'ParameterSlider',
            'label': 'DDIM Steps (After)',
            'control_name': 'DenoiserDDIMStepsSliderAfter',
            'min_value': '5', 'max_value': '50', 'default': '5', 'step': 1,
            'parentToggle': 'DenoiserAfterRestorersToggle',
            'requiredToggleValue': True,
            'parentSelection': 'DenoiserModeSelectionAfter',
            'requiredSelectionValue': "Full Restore (DDIM)",
            'help': "Number of DDIM steps for full restoration (After Restorers). Higher = more detail, slower."
        },
        'DenoiserCFGScaleDecimalSliderAfter': {
            'level': 3,
            'widget_type': 'ParameterDecimalSlider',
            'label': 'CFG Scale (After)',
            'control_name': 'DenoiserCFGScaleDecimalSliderAfter',
            'min_value': '0.0', 'max_value': '10.0', 'default': '1.0', 'step': 0.1, 'decimals': 1,
            'parentToggle': 'DenoiserAfterRestorersToggle',
            'requiredToggleValue': True,
            'parentSelection': 'DenoiserModeSelectionAfter',
            'requiredSelectionValue': "Full Restore (DDIM)",
            'help': "Classifier-Free Guidance scale for DDIM (After Restorers). Higher = stronger adherence to K/V."
        }
    }
}