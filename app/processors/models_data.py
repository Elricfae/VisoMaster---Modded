from pathlib import Path

models_dir = str(Path(__file__).resolve().parent.parent.parent / 'model_assets') # More robust path
assets_repo = "https://github.com/visomaster/visomaster-assets/releases/download"

try:
    import tensorrt as trt
    models_trt_list = [
        {
            'model_name': 'LivePortraitMotionExtractor', 
            'local_path': f'{models_dir}/liveportrait_onnx/motion_extractor.' + trt.__version__ + '.trt', 
            'hash': '8cab6d8fe093a07ee59e14bf83b9fbc90732ce7a6c1732b88b59f4457bea6204'
        },
        {
            'model_name': 'LivePortraitAppearanceFeatureExtractor', 
            'local_path': f'{models_dir}/liveportrait_onnx/appearance_feature_extractor.' + trt.__version__ + '.trt', 
            'hash': '7fea0c28948a5f0d21ae0712301084a0b4a0b1fdef48983840d58d8711da90af'
        },
        {
            'model_name': 'LivePortraitStitchingEye', 
            'local_path': f'{models_dir}/liveportrait_onnx/stitching_eye.' + trt.__version__ + '.trt', 
            'hash': '266afbccd79f2f5ae277242b19dd9299815b24dc453b22f6fd79fbf8f3a1e593'
        },
        {
            'model_name': 'LivePortraitStitchingLip', 
            'local_path': f'{models_dir}/liveportrait_onnx/stitching_lip.' + trt.__version__ + '.trt', 
            'hash': '2ac2e57eb2edd5aec70dc45023113e2ccc0495a16579c6c5d56fa30b74edc4f5'
        },
        {
            'model_name': 'LivePortraitStitching', 
            'local_path': f'{models_dir}/liveportrait_onnx/stitching.' + trt.__version__ + '.trt', 
            'hash': '8448de922a824b7b11eb7f470805ec22cf4ee541f7d66afeb2965094f96fd3ab'
        },
        {
            'model_name': 'LivePortraitWarpingSpadeFix', 
            'local_path': f'{models_dir}/liveportrait_onnx/warping_spade-fix.' + trt.__version__ + '.trt', 
            'hash': '24acdb6379b28fbefefb6339b3605693e00f1703c21ea5b8fec0215e521f6912'
        }
    ]
except ModuleNotFoundError:
    models_trt_list = []

arcface_mapping_model_dict = {
    'Inswapper128': 'Inswapper128ArcFace',
    'InStyleSwapper256 Version A': 'Inswapper128ArcFace',
    'InStyleSwapper256 Version B': 'Inswapper128ArcFace',
    'InStyleSwapper256 Version C': 'Inswapper128ArcFace',
    'DeepFaceLive (DFM)': 'Inswapper128ArcFace',
    'SimSwap512': 'SimSwapArcFace',
}

detection_model_mapping = {
    'RetinaFace': 'RetinaFace',
    'SCRFD': 'SCRFD2.5g',
    'Yolov8': 'YoloFace8n',
    'Yunet': 'YunetN',
}

landmark_model_mapping = {
    '5': 'FaceLandmark5',
    '68': 'FaceLandmark68',
    '3d68': 'FaceLandmark3d68',
    '98': 'FaceLandmark98',
    '106': 'FaceLandmark106',
    '203': 'FaceLandmark203',
    '478': 'FaceLandmark478',
}

models_list = [
    {
        "model_name": "Inswapper128",
        "local_path": f"{models_dir}/inswapper_128.fp16.onnx",
        "hash": "6d51a9278a1f650cffefc18ba53f38bf2769bf4bbff89267822cf72945f8a38b",
        "url": f"{assets_repo}/v0.1.0/inswapper_128.fp16.onnx"
    },
    {
        
        "model_name": "InStyleSwapper256 Version A",
        "local_path": f"{models_dir}/InStyleSwapper256_Version_A.fp16.onnx",
        "url": f"{assets_repo}/v0.1.0/InStyleSwapper256_Version_A.fp16.onnx",
        "hash": '0e0ef024b935abca69fd367a385200ed46b83a3cc618287ffe89440e2cc646da',

    },
    {
        
        "model_name": "InStyleSwapper256 Version B",
        "local_path": f"{models_dir}/InStyleSwapper256_Version_B.fp16.onnx",
        "url": f"{assets_repo}/v0.1.0/InStyleSwapper256_Version_B.fp16.onnx",
        "hash": '0870b6c75eaea239bdd72b6c6d0910cb285310736e356c17a2cd67a961738116',
    },
    {
        
        "model_name": "InStyleSwapper256 Version C",
        "local_path": f"{models_dir}/InStyleSwapper256_Version_C.fp16.onnx",
        "url": f"{assets_repo}/v0.1.0/InStyleSwapper256_Version_C.fp16.onnx",
        "hash": '6eaefc04cfb1461222ab72a814ad5b5673ab1af4267f7eb9054e308797567cde',
    },
    {
        "model_name": "SimSwap512",
        "local_path": f"{models_dir}/simswap_512_unoff.onnx",
        "hash": "08c6ca9c0a65eff119bea42686a4574337141de304b9d26e2f9d11e78d9e8e86",
        "url": f"{assets_repo}/v0.1.0/simswap_512_unoff.onnx"

    },
    {
        "model_name": "RetinaFace",
        "local_path": f"{models_dir}/det_10g.onnx",
        "hash": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
        "url": f"{assets_repo}/v0.1.0/det_10g.onnx"

    },
    {
        "model_name": "SCRFD2.5g",
        "local_path": f"{models_dir}/scrfd_2.5g_bnkps.onnx",
        "hash": "bc24bb349491481c3ca793cf89306723162c280cb284c5a5e49df3760bf5c2ce",
        "url": f"{assets_repo}/v0.1.0/scrfd_2.5g_bnkps.onnx"

    },
    {
        "model_name": "YoloFace8n",
        "local_path": f"{models_dir}/yoloface_8n.onnx",
        "hash": "84d5bb985b0ea75fc851d7454483897b1494c71c211759b4fec3a22ac196d206",
        "url": f"{assets_repo}/v0.1.0/yoloface_8n.onnx"
    },
    {
        "model_name": "FaceLandmark5",
        "local_path": f"{models_dir}/res50.onnx",
        "hash": "025db4efa3f7bef9911adc8eb92663608c682696a843cc7e1116d90c223354b5",
        "url": f"{assets_repo}/v0.1.0/res50.onnx"

    },
    {
        "model_name": "FaceLandmark68",
        "local_path": f"{models_dir}/2dfan4.onnx",
        "hash": "1ceedb108439c7d7b3f92cfa2b25bdc69a1f5f6c8b41da228cb283ca98d4181d",
        "url": f"{assets_repo}/v0.1.0/2dfan4.onnx"

    },
    {
        "model_name": "FaceLandmark3d68",
        "local_path": f"{models_dir}/1k3d68.onnx",
        "hash": "df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc",
        "url": f"{assets_repo}/v0.1.0/1k3d68.onnx"

    },
    {
        "model_name": "FaceLandmark98",
        "local_path": f"{models_dir}/peppapig_teacher_Nx3x256x256.onnx",
        "hash": "d4aa6dbd0081763a6eef04bf51484175b6a133ed12999bdc83b681a03f3f87d2",
        "url": f"{assets_repo}/v0.1.0/peppapig_teacher_Nx3x256x256.onnx"

    },
    {
        "model_name": "FaceLandmark106",
        "local_path": f"{models_dir}/2d106det.onnx",
        "hash": "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf",
        "url": f"{assets_repo}/v0.1.0/2d106det.onnx"

    },
    {
        "model_name": "FaceLandmark203",
        "local_path": f"{models_dir}/landmark.onnx",
        "hash": "31d22a5041326c31f19b78886939a634a5aedcaa5ab8b9b951a1167595d147db",
        "url": f"{assets_repo}/v0.1.0/landmark.onnx"

    },
    {
        "model_name": "FaceLandmark478",
        "local_path": f"{models_dir}/face_landmarks_detector_Nx3x256x256.onnx",
        "hash": "6d7932bdefc38871f57dd915b8c723d855e599f29cf4cdf19616fb35d0ed572e",
        "url": f"{assets_repo}/v0.1.0/face_landmarks_detector_Nx3x256x256.onnx"

    },
    {
        "model_name": "FaceBlendShapes",
        "local_path": f"{models_dir}/face_blendshapes_Nx146x2.onnx",
        "hash": "79065a18016da3b95f71247ff9ade3fe09b9124903a26a1af85af6d9e2a4faf3",
        "url": f"{assets_repo}/v0.1.0/face_blendshapes_Nx146x2.onnx"

    },
    {
        "model_name": "Inswapper128ArcFace",
        "local_path": f"{models_dir}/w600k_r50.onnx",
        "hash": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
        "url": f"{assets_repo}/v0.1.0/w600k_r50.onnx"

    },
    {
        "model_name": "SimSwapArcFace",
        "local_path": f"{models_dir}/simswap_arcface_model.onnx",
        "hash": "58949c864ab4a89012aaefc117f1ab8548c5f470bbc3889474bca13a412fc843",
        "url": f"{assets_repo}/v0.1.0/simswap_arcface_model.onnx"

    },
    {
        "model_name": "GFPGANv1.4",
        "local_path": f"{models_dir}/GFPGANv1.4.onnx",
        "hash": "6548e54cbcf248af385248f0c1193b359c37a0f98b836282b09cf48af4fd2b73",
        "url": f"{assets_repo}/v0.1.0/GFPGANv1.4.onnx"

    },
    {
        "model_name": "GFPGAN1024",
        "local_path": f"{models_dir}/gfpgan-1024.onnx",
        "hash": "ee8dd6415e388b3a410689d5d9395a2bf50b5973b588421ebfa57bc266f19e24",
        "url": "https://github.com/Glat0s/GFPGAN-1024-onnx/releases/download/v0.0.1/gfpgan-1024.onnx"

    },
    {
        "model_name": "GPENBFR256",
        "local_path": f"{models_dir}/GPEN-BFR-256.onnx",
        "hash": "aa5bd3ab238640a378c59e4a560f7a7150627944cf2129e6311ae4720e833271",
        "url": f"{assets_repo}/v0.1.0/GPEN-BFR-256.onnx"

    },
    {
        "model_name": "GPENBFR512",
        "local_path": f"{models_dir}/GPEN-BFR-512.onnx",
        "hash": "0960f836488735444d508b588e44fb5dfd19c68fde9163ad7878aa24d1d5115e",
        "url": f"{assets_repo}/v0.1.0/GPEN-BFR-512.onnx"

    },
    {
        "model_name": "GPENBFR1024",
        "local_path": f"{models_dir}/GPEN-BFR-1024.onnx",
        "hash": "cec8892093d7b99828acde97bf231fb0964d3fb11b43f3b0951e36ef1e192a3e",
        "url": f"{assets_repo}/v0.1.0/GPEN-BFR-1024.onnx"

    },
    {
        "model_name": "GPENBFR2048",
        "local_path": f"{models_dir}/GPEN-BFR-2048.onnx",
        "hash": "d0229ff43f979c360bd19daa9cd0ce893722d59f41a41822b9223ebbe4f89b3e",
        "url": f"{assets_repo}/v0.1.0/GPEN-BFR-2048.onnx"

    },
    {
        "model_name": "RealEsrganx2Plus",
        "local_path": f"{models_dir}/RealESRGAN_x2plus.fp16.onnx",
        "hash": "0b1770bcb31b3a9021d4251b538da4eb47c84f42706504d44a76d17e8c267606",
        "url": f"{assets_repo}/v0.1.0/RealESRGAN_x2plus.fp16.onnx"

    },
    {
        "model_name": "RealEsrganx4Plus",
        "local_path": f"{models_dir}/RealESRGAN_x4plus.fp16.onnx",
        "hash": "0a06c68f463a14bf5563b78d77d61ba4394024e148383c4308d6d3783eac2dc5",
        "url": f"{assets_repo}/v0.1.0/RealESRGAN_x4plus.fp16.onnx"

    },
    {
        "model_name": "RealEsrx4v3",
        "local_path": f"{models_dir}/realesr-general-x4v3.onnx",
        "hash": "09b757accd747d7e423c1d352b3e8f23e77cc5742d04bae958d4eb8082b76fa4",
        "url": f"{assets_repo}/v0.1.0/realesr-general-x4v3.onnx"

    },
    {
        "model_name": "BSRGANx2",
        "local_path": f"{models_dir}/BSRGANx2.fp16.onnx",
        "hash": "ba3a43613f5d2434c853201411b87e75c25ccb5b5918f38af504e4cf3bd4df9a",
        "url": f"{assets_repo}/v0.1.0/BSRGANx2.fp16.onnx"

    },
    {
        "model_name": "BSRGANx4",
        "local_path": f"{models_dir}/BSRGANx4.fp16.onnx",
        "hash": "e1467fbe60d2846919480f55a12ddbd5c516e343685bcdeac50ddcfa1dde2f46",
        "url": f"{assets_repo}/v0.1.0/BSRGANx4.fp16.onnx"

    },
    {
        "model_name": "UltraSharpx4",
        "local_path": f"{models_dir}/4x-UltraSharp.fp16.onnx",
        "hash": "d801b7f6081746e0b2cccef407c7a8acdb95e284c89298684582a8f2b35ad0f9",
        "url": f"{assets_repo}/v0.1.0/4x-UltraSharp.fp16.onnx"

    },
    {
        "model_name": "UltraMixx4",
        "local_path": f"{models_dir}/4x-UltraMix_Smooth.fp16.onnx",
        "hash": "3b96d63c239121b1ad5992e42a2089d6b4e1185c493c6440adfeafc0a20591eb",
        "url": f"{assets_repo}/v0.1.0/4x-UltraMix_Smooth.fp16.onnx"

    },
    {
        "model_name": "DeoldifyArt",
        "local_path": f"{models_dir}/ColorizeArtistic.fp16.onnx",
        "hash": "c8ad5c54b1b333361e959fdc6591828931b731f6652055f891d6118532cad081",
        "url": f"{assets_repo}/v0.1.0/ColorizeArtistic.fp16.onnx"

    },
    {
        "model_name": "DeoldifyStable",
        "local_path": f"{models_dir}/ColorizeStable.fp16.onnx",
        "hash": "666811485bfd37b236fdef695dbf50de7d3a430b10dbf5a3001d1609de06ad88",
        "url": f"{assets_repo}/v0.1.0/ColorizeStable.fp16.onnx"

    },
    {
        "model_name": "DeoldifyVideo",
        "local_path": f"{models_dir}/ColorizeVideo.fp16.onnx",
        "hash": "4d93b3cca8aa514bdf18a0ed00b25e36de5a9cc70b7aec7e60132632f6feced3",
        "url": f"{assets_repo}/v0.1.0/ColorizeVideo.fp16.onnx"

    },
    {
        "model_name": "DDColorArt",
        "local_path": f"{models_dir}/ddcolor_artistic.onnx",
        "hash": "2f2510323e59995051eeac4f1ef8c267130eabf6187535defa55c11929b2b31c",
        "url": f"{assets_repo}/v0.1.0/ddcolor_artistic.onnx"

    },
    {
        "model_name": "DDcolor",
        "local_path": f"{models_dir}/ddcolor.onnx",
        "hash": "4e8b8a8d7c346ea7df08fc0bc985d30c67f5835cd1b81b6728f6bbe8b7658ae1",
        "url": f"{assets_repo}/v0.1.0/ddcolor.onnx"

    },
    {
        "model_name": "Occluder",
        "local_path": f"{models_dir}/occluder.onnx",
        "hash": "79f5c2edf10b83458693d122dd51488b210fb80c059c5d56347a047710d44a78",
        "url": f"{assets_repo}/v0.1.0/occluder.onnx"

    },
    {
        "model_name": "XSeg",
        "local_path": f"{models_dir}/XSeg_model.onnx",
        "hash": "4381395dcbec1eef469fa71cfb381f00ac8aadc3e5decb4c29c36b6eb1f38ad9",
        "url": f"{assets_repo}/v0.1.0/XSeg_model.onnx"

    },
    {
        "model_name": "FaceParser",
        "local_path": f"{models_dir}/faceparser_resnet34.onnx",
        "hash": "5b805bba7b5660ab7070b5a381dcf75e5b3e04199f1e9387232a77a00095102e",
        "url": f"{assets_repo}/v0.1.0/faceparser_resnet34.onnx"

    },    
    {
        "model_name": "combo_relu3_3_relu3_1",
        "local_path": f"{models_dir}/vgg_combo_relu3_3_relu3_1.onnx",
        "hash": "1068EE41E3C67DCFBBECCBC93E539EB06F89BBA08618951BB33E9BE2C1FBC986",
        "url": f"{assets_repo}/v0.1.0_lp/vgg_combo_relu3_3_relu3_1.onnx"

    },
    {
        "model_name": "LivePortraitMotionExtractor",
        "local_path": f"{models_dir}/liveportrait_onnx/motion_extractor.onnx",
        "hash": "99d4b3c9dd3fd301910de9415a29560e38c0afaa702da51398281376cc36fdd3",
        "url": f"{assets_repo}/v0.1.0_lp/motion_extractor.onnx"

    },
    {
        "model_name": "LivePortraitAppearanceFeatureExtractor",
        "local_path": f"{models_dir}/liveportrait_onnx/appearance_feature_extractor.onnx",
        "hash": "dbbbb44e4bba12302d7137bdee6a0f249b45fb6dd879509fd5baa27d70c40e32",
        "url": f"{assets_repo}/v0.1.0_lp/appearance_feature_extractor.onnx"

    },
    {
        "model_name": "LivePortraitStitchingEye",
        "local_path": f"{models_dir}/liveportrait_onnx/stitching_eye.onnx",
        "hash": "251004fe4a994c57c8cd9f2c50f3d89feb289fb42e6bc3af74470a3a9fa7d83b",
        "url": f"{assets_repo}/v0.1.0_lp/stitching_eye.onnx"

    },
    {
        "model_name": "LivePortraitStitchingLip",
        "local_path": f"{models_dir}/liveportrait_onnx/stitching_lip.onnx",
        "hash": "1ca793eac4b0dc5464f1716cdaa62e595c2c2272c9971a444e39c164578dc34b",
        "url": f"{assets_repo}/v0.1.0_lp/stitching_lip.onnx"

    },
    {
        "model_name": "LivePortraitStitching",
        "local_path": f"{models_dir}/liveportrait_onnx/stitching.onnx",
        "hash": "43598e9747a19f4c55d8e1604fb7d7fa70ab22377d129cb7d1fe38c9a737cc79",
        "url": f"{assets_repo}/v0.1.0_lp/stitching.onnx"

    },
    {
        "model_name": "LivePortraitWarpingSpade",
        "local_path": f"{models_dir}/liveportrait_onnx/warping_spade.onnx",
        "hash": "d6ee9af4352b47e88e0521eba6b774c48204afddc8d91c671a5f7b8a0dfb4971",
        "url": f"{assets_repo}/v0.1.0_lp/warping_spade.onnx"

    },
    {
        "model_name": "LivePortraitWarpingSpadeFix",
        "local_path": f"{models_dir}/liveportrait_onnx/warping_spade-fix.onnx",
        "hash": "a6164debbf1e851c3dcefa622111c42a78afd9bb8f1540e7d01172ddf642c3b5",
        "url": f"{assets_repo}/v0.1.0_lp/warping_spade-fix.onnx"

    },
    {
        "model_name": "RefLDMVAEEncoder",
        "local_path": f"{models_dir}/ref_ldm_vae_encoder.onnx",
        "hash": "b88d18e79bb0dc2a0d2763e4fd806d6ce7f885a6503a828ab862a7c284d456fc",
        "url": "https://github.com/Glat0s/ref-ldm-onnx/releases/download/v0.0.1/ref_ldm_vae_encoder.onnx"
    },
    {
        "model_name": "RefLDMVAEDecoder",
        "local_path": f"{models_dir}/ref_ldm_vae_decoder.onnx",
        "hash": "eca3065e6a40f4f73a0a14bc810769d07563a351964a0830ad59a481aa00b4f5",
        "url": "https://github.com/Glat0s/ref-ldm-onnx/releases/download/v0.0.1/ref_ldm_vae_decoder.onnx"
    },
    {
        "model_name": "RefLDM_UNET_EXTERNAL_KV",
        "local_path": f"{models_dir}/ref_ldm_unet_external_kv.onnx",
        "hash": "56edbea2aaf0361607645bbe0f35ce07ff8ddce80ee0ef617af305d50d251154",
        "url": "https://github.com/Glat0s/ref-ldm-onnx/releases/download/v0.0.1/ref_ldm_unet_external_kv.onnx"
    }
]