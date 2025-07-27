from app.helpers import miscellaneous as misc_helpers
from app.ui.widgets.actions import layout_actions
from app.helpers.typing_helper import LayoutDictTypes

# Widgets in Face Swap tab are created from this Layout
SWAPPER_LAYOUT_DATA: LayoutDictTypes = {
    'Swapper': {
        'SwapModelSelection': {
            'level': 1,
            'label': 'Swapper Model',
            'options': ['Inswapper128', 'InStyleSwapper256 Version A', 'InStyleSwapper256 Version B', 'InStyleSwapper256 Version C', 'DeepFaceLive (DFM)', 'SimSwap512', 'Hyperswap256 Version A', 'Hyperswap256 Version B', 'Hyperswap256 Version C'],
            'default': 'Inswapper128',
            'help': 'Choose which swapper model to use for face swapping.'
        },
        'SwapperResSelection': {
            'level': 2,
            'label': 'Swapper Resolution',
            'options': ['128', '256', '384', '512'],
            'default': '256',
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'Inswapper128',
            'help': 'Select the resolution for the swapped face in pixels. Higher values offer better quality but are slower to process.'
        },
        'InStyleResAEnableToggle': {
            'level': 2,
            'label': '512 Resolution',
            'default': False,            
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'InStyleSwapper256 Version A',
            'help': 'Like the inswapper Resolution (512) for InStyleSwappers. i dont know to hide it with 3 selections possible :(.'
        }, 
        'InStyleResBEnableToggle': {
            'level': 2,
            'label': '512 Resolution',
            'default': False,            
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'InStyleSwapper256 Version B',
            'help': 'Like the inswapper Resolution (512) for InStyleSwappers. i dont know to hide it with 3 selections possible :(.'
        }, 
        'InStyleResCEnableToggle': {
            'level': 2,
            'label': '512 Resolution',
            'default': False,            
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'InStyleSwapper256 Version C',
            'help': 'Like the inswapper Resolution (512) for InStyleSwappers. i dont know to hide it with 3 selections possible :(.'
        },
        'HyperAEnableToggle': {
            'level': 2,
            'label': '512 Resolution',
            'default': False,            
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'Hyperswap256 Version A',
            'help': 'Like the inswapper Resolution (512) for Hyperswap. i dont know to hide it with 3 selections possible :(.'
        }, 
        'HyperBEnableToggle': {
            'level': 2,
            'label': '512 Resolution',
            'default': False,            
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'Hyperswap256 Version B',
            'help': 'Like the inswapper Resolution (512) for Hyperswap. i dont know to hide it with 3 selections possible :(.'
        }, 
        'HyperCEnableToggle': {
            'level': 2,
            'label': '512 Resolution',
            'default': False,            
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'Hyperswap256 Version C',
            'help': 'Like the inswapper Resolution (512) for Hyperswap. i dont know to hide it with 3 selections possible :(.'
        },
        'DFMModelSelection': {
            'level': 2,
            'label': 'DFM Model',
            'options': misc_helpers.get_dfm_models_selection_values,
            'default': misc_helpers.get_dfm_models_default_value,
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'DeepFaceLive (DFM)',
            'help': 'Select which pretrained DeepFaceLive (DFM) Model to use for swapping.'
        },
        'DFMAmpMorphSlider': {
            'level': 2,
            'label': 'AMP Morph Factor',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'DeepFaceLive (DFM)',
            'help': 'AMP Morph Factor for DFM AMP Models',
        },
        'DFMRCTColorToggle': {
            'level': 2,
            'label': 'RCT Color Transfer',
            'default': False,
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'DeepFaceLive (DFM)',
            'help': 'RCT Color Transfer for DFM Models',
        },
        'PreSwapSharpnessDecimalSlider': {
            'level': 1,
            'label': 'Pre Swap Sharpness (1.0)',
            'min_value': '0.0',
            'max_value': '2.0',
            'default': '1.0',
            'step': 0.1,
            'decimals': 1,
            'help': 'Sharpens the original face befor swapping. can sometimes be usefull. care it can tamper with "Auto Face Restorer"!'
        }
    },
    'Face Similarity': {
        'SimilarityThresholdSlider': {
            'level': 1,
            'label': 'Similarity Threshold',
            'min_value': '1',
            'max_value': '100',
            'default': '58',
            'step': 1,
            'help': 'Set the similarity threshold to control how similar the detected face should be to the reference (target) face.'
        },
        'StrengthEnableToggle': {
            'level': 1,
            'label': 'Strength',
            'default': False,
            'help': 'Apply additional swapping iterations to increase the strength of the result, which may increase likeness.'
        },
        'StrengthAmountSlider': {
            'level': 2,
            'label': 'Amount',
            'min_value': '0',
            'max_value': '500',
            'default': '100',
            'step': 25,
            'parentToggle': 'StrengthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Increase up to 5x additional swaps (500%). 200% is generally a good result. Set to 0 to turn off swapping but allow the rest of the pipeline to apply to the original image.'
        },
        'FaceLikenessEnableToggle': {
            'level': 1,
            'label': 'Face Likeness',
            'default': False,
            'help': 'This is a feature to perform direct adjustments to likeness of faces.'
        },
        'FaceLikenessFactorDecimalSlider': {
            'level': 2,
            'label': 'Amount',
            'min_value': '-1.00',
            'max_value': '1.00',
            'default': '0.00',
            'decimals': 2,
            'step': 0.05,
            'parentToggle': 'FaceLikenessEnableToggle',
            'requiredToggleValue': True,
            'help': 'Determines the factor of likeness between the source and assigned faces.'
        },
        'AutoColorEnableToggle': {
            'level': 1,
            'label': 'AutoColor Transfer',
            'default': False,
            'help': 'Enable AutoColor Transfer: 1. Hans Test without mask, 2. Hans Test with mask, 3. DFL Method without mask, 4. DFL Original Method.'
        },
        'AutoColorTransferTypeSelection':{
            'level': 2,
            'label': 'Transfer Type',
            'options': ['Test', 'Test_Mask', 'DFL_Test', 'DFL_Orig'],
            'default': 'DFL_Orig',
            'parentToggle': 'AutoColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Select the AutoColor transfer method type.'
        },
        'AutoColorBlendAmountSlider': {
            'level': 2,
            'label': 'Blend Amount',
            'min_value': '0',
            'max_value': '100',
            'default': '95',
            'step': 5,
            'parentToggle': 'AutoColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the blend value.'
        },
        'ColorEnableToggle': {
            'level': 1,
            'label': 'Color Adjustments',
            'default': False,
            'help': 'Fine-tune the RGB color values of the swap.'
        },
        'ColorRedSlider': {
            'level': 1,
            'label': 'Red',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Red color adjustments'
        },
        'ColorGreenSlider': {
            'level': 1,
            'label': 'Green',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Green color adjustments'
        },
        'ColorBlueSlider': {
            'level': 1,
            'label': 'Blue',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blue color adjustments'
        },
        'ColorBrightnessDecimalSlider': {
            'level': 1,
            'label': 'Brightness',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the Brightness.'
        },
        'ColorContrastDecimalSlider': {
            'level': 1,
            'label': 'Contrast',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the Contrast.'
        },
        'ColorSaturationDecimalSlider': {
            'level': 1,
            'label': 'Saturation',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the Saturation.'
        },
        'ColorSharpnessDecimalSlider': {
            'level': 1,
            'label': 'Sharpness',
            'min_value': '0.0',
            'max_value': '2.0',
            'default': '1.0',
            'step': 0.1,
            'decimals': 1,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the Sharpness.'
        },
        'ColorHueDecimalSlider': {
            'level': 1,
            'label': 'Hue',
            'min_value': '-0.50',
            'max_value': '0.50',
            'default': '0.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the Hue.'
        },
        'ColorGammaDecimalSlider': {
            'level': 1,
            'label': 'Gamma',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the Gamma.'
        },
        'ColorNoiseDecimalSlider': {
            'level': 1,
            'label': 'Noise',
            'min_value': '0.0',
            'max_value': '20.0',
            'default': '0.0',
            'step': 0.5,
            'decimals': 1,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Add noise to swapped face.'
        },
        'JPEGCompressionEnableToggle': {
            'level': 1,
            'label': 'JPEG Compression',
            'default': False,
            'help': 'Apply JPEG Compression to the swapped face to make output more realistic',
        },
        'JPEGCompressionAmountSlider': {
            'level': 2,
            'label': 'Compression',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'JPEGCompressionEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the JPEG Compression amount'
        }
    },
    'Face Mask':{
        'MaskShowSelection':{
            'level': 1,
            'label': 'mask show',
            'options': ['swap_mask', 'texture'],
            'default': 'swap_mask',
            'help': 'select what mask is shown in "view face mask".'
        },    
        'BorderBottomSlider':{
            'level': 1,
            'label': 'Bottom Border',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'help': 'A rectangle with adjustable bottom, left, right, top, and sides that masks the swapped face result back into the original image.'
        },
        'BorderLeftSlider':{
            'level': 1,
            'label': 'Left Border',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'help': 'A rectangle with adjustable bottom, left, right, top, and sides that masks the swapped face result back into the original image.'
        },
        'BorderRightSlider':{
            'level': 1,
            'label': 'Right Border',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'help': 'A rectangle with adjustable bottom, left, right, top, and sides that masks the swapped face result back into the original image.'
        },
        'BorderTopSlider':{
            'level': 1,
            'label': 'Top Border',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'help': 'A rectangle with adjustable bottom, left, right, top, and sides that masks the swapped face result back into the original image.'
        },
        'BorderBlurSlider':{
            'level': 1,
            'label': 'Border Blur',
            'min_value': '0',
            'max_value': '100',
            'default': '6',
            'step': 1,
            'help': 'Border mask blending distance.'
        },
        'OccluderEnableToggle': {
            'level': 1,
            'label': 'Occlusion Mask',
            'default': False,
            'help': 'Allow objects occluding the face to show up in the swapped image.'
        },
        'OccluderSizeSlider': {
            'level': 2,
            'label': 'Size',
            'min_value': '-25',
            'max_value': '25',
            'default': '0',
            'step': 1,
            'parentToggle': 'OccluderEnableToggle',
            'requiredToggleValue': True,
            'help': 'Grows or shrinks the occluded region'
        },
        'DFLXSegEnableToggle': {
            'level': 1,
            'label': 'DFL XSeg Mask',
            'default': False,
            'help': 'Allow objects occluding the face to show up in the swapped image.'
        },
        'DFLXSegSizeSlider': {
            'level': 2,
            'label': 'Size',
            'min_value': '-20',
            'max_value': '20',
            'default': '0',
            'step': 1,
            'parentToggle': 'DFLXSegEnableToggle',
            'requiredToggleValue': True,
            'help': 'Grows or shrinks the occluded region.'
        },
        'OccluderXSegBlurSlider': {
            'level': 1,
            'label': 'Occluder/DFL XSeg Blur',
            'min_value': '0',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'OccluderEnableToggle | DFLXSegEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend value for Occluder and XSeg.'
        },
        'DFLXSeg2EnableToggle': {
            'level': 1,
            'label': 'Xseg 2',
            'default': False,
            'help': 'Enable second XSeg Mask for special regions.'
        },
        'DFLXSeg2SizeSlider': {
            'level': 2,
            'label': 'Size2',
            'min_value': '-30',
            'max_value': '30',
            'default': '-1',
            'step': 1,
            'parentToggle': 'DFLXSeg2EnableToggle',
            'requiredToggleValue': True,
            'help': 'Grows or shrinks the occluded region of second Xseg Mask (for BG and mouth).'
        },
        'XSeg2BlurSlider': {
            'level': 2,
            'label': 'XSeg2 Blur',
            'min_value': '0',
            'max_value': '100',
            'default': '4',
            'step': 1,
            'parentToggle': 'DFLXSeg2EnableToggle',
            'requiredToggleValue': True,
            'help': 'Blur for second XSeg Mask'
        },
        'XSegMouthEnableToggle': {
            'level': 2,
            'label': 'Xseg Mouth',
            'default': False,
            'parentToggle': 'DFLXSeg2EnableToggle',
            'requiredToggleValue': True,
            'help': 'Use second Xseg Mask on mouth region.'
        },       
        'XsegUpperLipParserSlider': {
            'level': 3,
            'label': 'Upper Lip',
            'min_value': '0',
            'max_value': '30',
            'default': '2',
            'step': 1,
            'parentToggle': 'DFLXSeg2EnableToggle',
            'requiredToggleValue': True,
            'help': 'Grow Upper Lip Region (uses Faceparser on swap)'
        },
        'XsegMouthParserSlider': {
            'level': 3,
            'label': 'Mouth',
            'min_value': '0',
            'max_value': '30',
            'default': '2',
            'step': 1,
            'parentToggle': 'DFLXSeg2EnableToggle',
            'requiredToggleValue': True,
            'help': 'Grow Mouth Region (uses Faceparser on swap)'
        },
        'XsegLowerLipParserSlider': {
            'level': 3,
            'label': 'Lower Lip',
            'min_value': '0',
            'max_value': '30',
            'default': '10',
            'step': 1,
            'parentToggle': 'DFLXSeg2EnableToggle',
            'requiredToggleValue': True,
            'help': 'Grow Lower Lip Region (uses Faceparser on swap)'
        },
        'DFLXSegBGEnableToggle': {
            'level': 2,
            'label': 'Xseg 2 Background',
            'default': False,
            'parentToggle': 'DFLXSeg2EnableToggle',
            'requiredToggleValue': True,            
            'help': 'Enable second XSeg Mask for Inside the Face. not working well atm. (uses Faceparser on swap)'
        },
        'OccluderMaskBgSlider': {
            'level': 2,
            'label': 'Xseg 2 Background Adjust',
            'min_value': '-40',
            'max_value': '40',
            'default': '-10',
            'step': 1,
            'parentToggle': 'DFLXSeg2EnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust where the second Xseg Mask gets applied.'
        },
        'TransferTextureEnableToggle': {
            'level': 1,
            'label': 'Transfer Texture',
            'default': False,
            'help': 'Enable Texture Transfer'
        },
        'TransferTextureBlendAmountSlider': {
            'level': 2,
            'label': 'Texture Strength Amount',
            'min_value': '0',
            'max_value': '100',
            'default': '30',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'Define how strong texture Transfer is applied'
        },              
        'TransferTexturePreGammaDecimalSlider': {
            'level': 2,
            'label': 'Texture Gamma adjust',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'decimals': 2,
            'step': 0.05,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'adjusting can sometimes be good. Gamma adjust of original Face for Texture Transfer'
        },            
        'TransferTexturePreContrastDecimalSlider': {
            'level': 2,
            'label': 'Texture Contrast adjust',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'decimals': 2,
            'step': 0.05,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'adjusting can sometimes be good. Contrast adjust of original Face for Texture Transfer'
        },
        'TransferTextureClaheEnableToggle': {
            'level': 2,
            'label': 'CLAHE',
            'default': False,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'for some extra wums. (Contrast Limited Adaptive Histogram Equalization). enhancing local contrast while preventing noise overamplification. handle with care, changes color dynamics.'
        },                   
        'TransferTextureClipLimitDecimalSlider': {
            'level': 2,
            'label': 'CLAHE Limit',
            'min_value': '0.0',
            'max_value': '5.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': ''
        },             
        'TransferTextureAlphaClaheDecimalSlider': {
            'level': 2,
            'label': 'CLAHE Blend',
            'min_value': '0.00',
            'max_value': '1.00',
            'default': '0.40',
            'decimals': 2,
            'step': 0.05,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'lower blend for additional texture strength without making colors too extreme'
        },        
        'ExcludeOriginalVGGMaskEnableToggle': {
            'level': 2,
            'label': 'VGG Mask Exclude',
            'default': False,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,            
            'help': 'Original Exclude Mask with VGG Model (without lower limit / upper limit / strenght  manipulation)'
        },    
        'TextureBlendAmountSlider': {
            'level': 2,
            'label': 'VGG Mask Blur Amount',
            'min_value': '0',
            'max_value': '20',
            'default': '2',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blur the VGG Mask for smoother transitions.'
        },
        'ExcludeVGGMaskEnableToggle': {
            'level': 3,
            'label': 'VGG Mask Manipulation',
            'default': False,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,            
            'help': 'Exclude Mask with VGG Model and lower limit / upper limit / strenght'
        }, 
        'TextureLowerLimitThreshSlider': {
            'level': 3,
            'label': 'Face Features Lower Limit',
            'min_value': '0',
            'max_value': '100',
            'default': '30',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'Defines lower Limit for transfer mask. Check "View Face Mask" with "mask show -> texture" to see effect.'
        },   
        'TextureUpperLimitThreshSlider': {
            'level': 3,
            'label': 'Face Features Upper Limit',
            'min_value': '0',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'Defines upper Limit for transfer mask. Check "View Face Mask" with "mask show -> texture" to see effect.'
        },                                     
        'TextureMiddleLimitValueSlider': {
            'level': 3,
            'label': 'Lower Strength',
            'min_value': '0',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'Defines the strenght of parts under "Face Features Lower Limit" in transfer mask. Check "View Face Mask" with "mask show -> texture" to see effect.'
        },       
        'TextureUpperLimitValueSlider': {
            'level': 3,
            'label': 'Upper Strength',
            'min_value': '0',
            'max_value': '100',
            'default': '100',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'Defines the strenght of parts over "Face Features Upper Limit" in transfer mask. Check "View Face Mask" with "mask show -> texture" to see effect.'
        },
        'ExcludeMaskEnableToggle': {
            'level': 2,
            'label': 'Mask Features Exclude',
            'default': False,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,            
            'help': 'Exclude Faceparts from Texture Transfere and uses the original Swap there. Combineable with VGG Mask'
        }, 
        'FaceParserTextureSlider': {
            'level': 3,
            'label': 'Face (is also Blend value)',
            'min_value': '0',
            'max_value': '10',
            'default': '0',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend amount of rest of Face (without features)'
        },        
        'EyebrowParserTextureSlider': {
            'level': 3,
            'label': 'Eyebrows',
            'min_value': '0',
            'max_value': '10',
            'default': '1',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'Exclude Faceparts (Eyes, Eyebrows, Nose, Mouth, Lips, Neck), 0=whole face is used, 1= Parts not included, 1+ = increase Parts size. Most of the time should be 1/1+. try 0 on low quality/artefacted targets'
        },         
        'EyeParserTextureSlider': {
            'level': 3,
            'label': 'Eyes',
            'min_value': '-10',
            'max_value': '10',
            'default': '1',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'Exclude Faceparts (Eyes, Eyebrows, Nose, Mouth, Lips, Neck), 0=whole face is used, 1= Parts not included, 1+ = increase Parts size. Most of the time should be 1/1+. try 0 on low quality/artefacted targets'
        },        
        'NoseParserTextureSlider': {
            'level': 3,
            'label': 'Nose',
            'min_value': '0',
            'max_value': '10',
            'default': '1',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'Exclude Faceparts (Eyes, Eyebrows, Nose, Mouth, Lips, Neck), 0=whole face is used, 1= Parts not included, 1+ = increase Parts size. Most of the time should be 1/1+. try 0 on low quality/artefacted targets'
        },        
        'MouthParserTextureSlider': {
            'level': 3,
            'label': 'Mouth',
            'min_value': '0',
            'max_value': '10',
            'default': '1',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'Exclude Faceparts (Eyes, Eyebrows, Nose, Mouth, Lips, Neck), 0=whole face is used, 1= Parts not included, 1+ = increase Parts size. Most of the time should be 1/1+. try 0 on low quality/artefacted targets'
        },        
        'NeckParserTextureSlider': {
            'level': 3,
            'label': 'Neck',
            'min_value': '0',
            'max_value': '10',
            'default': '0',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'Exclude Faceparts (Eyes, Eyebrows, Nose, Mouth, Lips, Neck), 0=whole face is used, 1= Parts not included, 1+ = increase Parts size. Most of the time should be 1/1+. try 0 on low quality/artefacted targets'
        },        
        'BackgroundParserTextureSlider': {
            'level': 3,
            'label': 'Background',
            'min_value': '-20',
            'max_value': '0',
            'default': '0',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'Decrease Background Area for Texture Transfer.'
        },  
        'FaceParserBlendTextureSlider': {
            'level': 3,
            'label': 'Excluded Texture Blend adjust',
            'min_value': '-50',
            'max_value': '50',
            'default': '0',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend Amount of Excluded Feature Areas'
        },         
        'FaceParserBlurTextureSlider': {
            'level': 3,
            'label': 'Texture Mask Blur',
            'min_value': '0',
            'max_value': '10',
            'default': '4',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'Mask Blur on excluded Area Edges.'
        },      
        'BgExcludeEnableToggle': {
            'level': 2,
            'label': 'Background Exclude',
            'default': False,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'Background reduce for Texture Transfer Mask, usefull if xseg > 0'
        },
        'DFLXSeg3SizeSlider': {
            'level': 3,
            'label': 'BG XSeg Adjust',
            'min_value': '-30',
            'max_value': '0',
            'default': '0',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'Background reduce based on XSEG model on original face'
        },
        'BGExcludeBlurAmountSlider': {
            'level': 3,
            'label': 'BG Blur',
            'min_value': '0',
            'max_value': '50',
            'default': '0',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'Background reduce based on faceparser model on original face'
        },
        'FaceParserEnableToggle': {
            'level': 1,
            'label': 'Face Parser Mask',
            'default': False,
            'help': 'Allow the unprocessed background from the orginal image to show in the final swap.'
        },
        'FaceParserSlider': {
            'level': 2,
            'label': 'Face',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the entire face.'
        },
        'LeftEyebrowParserSlider': {
            'level': 2,
            'label': 'Left Eyebrow',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the left eyebrow.'
        },
        'RightEyebrowParserSlider': {
            'level': 2,
            'label': 'Right Eyebrow',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the right eyebrow.'
        },
        'LeftEyeParserSlider': {
            'level': 2,
            'label': 'Left Eye',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the left eye.'
        },
        'RightEyeParserSlider': {
            'level': 2,
            'label': 'Right Eye',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the right eye.'
        },
        'EyeGlassesParserSlider': {
            'level': 2,
            'label': 'EyeGlasses',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the eyeglasses.'
        },
        'NoseParserSlider': {
            'level': 2,
            'label': 'Nose',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the nose.'
        },
        'MouthParserSlider': {
            'level': 2,
            'label': 'Mouth',
            'min_value': '0',
            'max_value': '20',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the inside of the mouth, including the tongue.'
        },
        'MouthParserInsideToggle': {
            'level': 2,
            'label': 'Mouth Inside toggle',
            'default': True,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Mask the inside of the mouth of the swapped face.'
        },
        'UpperLipParserSlider': {
            'level': 2,
            'label': 'Upper Lip',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the upper lip.'
        },
        'LowerLipParserSlider': {
            'level': 2,
            'label': 'Lower Lip',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the lower lip.'
        },
        'NeckParserSlider': {
            'level': 2,
            'label': 'Neck',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the neck.'
        },
        'HairParserSlider': {
            'level': 2,
            'label': 'Hair',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the hair.'
        },
        'BackgroundBlurParserSlider': {
            'level': 2,
            'label': 'Background Blur',
            'min_value': '0',
            'max_value': '100',
            'default': '5',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend the value for Background Parser'
        },
        'FaceBlurParserSlider': {
            'level': 2,
            'label': 'Face Blur',
            'min_value': '0',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend the value for Face Parser'
        },
        'FaceParserBlendSlider': {
            'level': 2,
            'label': 'Face Blend',
            'min_value': '0',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend the value for Face Parser'
        },
        'FaceParserHairMakeupEnableToggle': {
            'level': 2,
            'label': 'Hair Makeup',
            'default': False,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Enable hair makeup'
        },
        'FaceParserHairMakeupRedSlider': {
            'level': 3,
            'label': 'Red',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle & FaceParserHairMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Red color adjustments'
        },
        'FaceParserHairMakeupGreenSlider': {
            'level': 3,
            'label': 'Green',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 3,
            'parentToggle': 'FaceParserEnableToggle & FaceParserHairMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Green color adjustments'
        },
        'FaceParserHairMakeupBlueSlider': {
            'level': 3,
            'label': 'Blue',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle & FaceParserHairMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blue color adjustments'
        },
        'FaceParserHairMakeupBlendAmountDecimalSlider': {
            'level': 3,
            'label': 'Blend Amount',
            'min_value': '0.1',
            'max_value': '1.0',
            'default': '0.2',
            'step': 0.1,
            'decimals': 1,
            'parentToggle': 'FaceParserEnableToggle & FaceParserHairMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend the value: 0.0 represents the original color, 1.0 represents the full target color.'
        },
        'FaceParserLipsMakeupEnableToggle': {
            'level': 2,
            'label': 'Lips Makeup',
            'default': False,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Enable lips makeup'
        },
        'FaceParserLipsMakeupRedSlider': {
            'level': 3,
            'label': 'Red',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle & FaceParserLipsMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Red color adjustments'
        },
        'FaceParserLipsMakeupGreenSlider': {
            'level': 3,
            'label': 'Green',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 3,
            'parentToggle': 'FaceParserEnableToggle & FaceParserLipsMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Green color adjustments'
        },
        'FaceParserLipsMakeupBlueSlider': {
            'level': 3,
            'label': 'Blue',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle & FaceParserLipsMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blue color adjustments'
        },
        'FaceParserLipsMakeupBlendAmountDecimalSlider': {
            'level': 3,
            'label': 'Blend Amount',
            'min_value': '0.1',
            'max_value': '1.0',
            'default': '0.2',
            'step': 0.1,
            'decimals': 1,
            'parentToggle': 'FaceParserEnableToggle & FaceParserLipsMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend the value: 0.0 represents the original color, 1.0 represents the full target color.'
        }
    },    
    'Blend Adjustments':{
        'FinalBlendAdjEnableToggle': {
            'level': 1,
            'label': 'Final Blend',
            'default': False,
            'help': 'Blend at the end of pipeline.'
        },
        'FinalBlendAmountSlider': {
            'level': 2,
            'label': 'Final Blend Amount',
            'min_value': '1',
            'max_value': '50',
            'default': '1',
            'step': 1,
            'parentToggle': 'FinalBlendAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the final blend value.'
        },
        'OverallMaskBlendAmountSlider': {
            'level': 1,
            'label': 'Overall Mask Blend Amount',
            'min_value': '0',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'help': 'Combined masks blending distance. It is not applied to the border masks.'
        },        
    },
}