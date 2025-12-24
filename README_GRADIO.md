# VisoMaster - Gradio Web Interface Conversion

This document describes the conversion of VisoMaster from a desktop Qt/PySide6 application to a web-based Gradio interface for cloud GPU deployment.

## Status

🚧 **Work in Progress** - Framework created, core integration in progress

### Completed
- ✅ Analyzed existing codebase structure
- ✅ Created Gradio web interface framework (`app.py`)
- ✅ Created web-friendly `requirements.txt`
- ✅ Qt dependency mocking system for headless operation
- ✅ Models processor integration structure

### In Progress
- 🔄 Face swapping pipeline integration
- 🔄 Simplified parameter system
- 🔄 Testing and debugging

### To Do
- ⏳ Complete face swapping implementation
- ⏳ Add LivePortrait face editing interface
- ⏳ Video processing support
- ⏳ Add examples and sample images

## Architecture

### Original Desktop App
```
main.py
  └─> PySide6/Qt UI (main_ui.py)
      ├─> VideoProcessor (threading, Qt signals)
      ├─> FrameWorker (complex processing pipeline)
      └─> ModelsProcessor (model loading, inference)
```

### Web Conversion
```
app.py
  └─> Gradio Interface
      ├─> MockQt (bypass Qt dependencies)
      ├─> ModelsProcessor (reused, Qt-mocked)
      ├─> Face Processors (reused: detectors, swappers, editors)
      └─> Simplified processing pipeline
```

## Files Created/Modified

### New Files
- **`app.py`** - Main Gradio web interface
- **`requirements.txt`** - Web deployment dependencies (no Qt/Desktop deps)
- **`gradio_processor.py`** - Simplified processor wrapper (WIP)
- **`README_GRADIO.md`** - This file

### Key Design Decisions

1. **Qt Mocking**: Instead of removing Qt dependencies, we mock PySide6 modules to avoid refactoring the entire codebase
2. **Processor Reuse**: The core `app/processors/` modules are reused as-is
3. **Simplified Interface**: Focus on essential features first (face swap, detection)
4. **Headless Operation**: Set `QT_QPA_PLATFORM=offscreen` for no-display environments

## Installation

### For Cloud GPU (RunPod, Paperspace, etc.)

```bash
# Clone the repository
git clone <your-repo-url>
cd VisoMaster---Modded

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Download required models (optional, will auto-download on first use)
python download_models.py
```

### For Local Development

```bash
# Same as above, or use CPU-only PyTorch
pip install torch torchvision torchaudio
```

## Usage

### Launch the Web Interface

```bash
python app.py
```

The interface will be available at:
- Local: http://localhost:7860
- Network: http://0.0.0.0:7860

### For Public Access (RunPod, etc.)

Edit `app.py` and change:
```python
interface.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,  # Change to True for public Gradio link
)
```

## Current Functionality

### Face Detection ✅
- Upload an image
- Detect faces with RetinaFace, SCRFD, Yolov8, or Yunet
- Adjustable detection threshold

### Face Swapping 🔄 (In Progress)
- Upload target image and source face
- Select swapper model (Inswapper128, InStyleSwapper, SimSwap512)
- Similarity threshold adjustment
- Optional face restoration (GFPGAN, GPEN)

### Face Editing ⏳ (Planned)
- LivePortrait-based pose/expression editing
- Head pitch/yaw/roll adjustment
- Eye/lip expression control

## Development Notes

### Challenges Addressed

1. **Qt Dependencies**: The original app heavily uses Qt signals/slots. Solution: Mock Qt modules
2. **Complex Processing Pipeline**: 800+ lines in `frame_worker.py`. Solution: Simplified interface with gradual feature addition
3. **Model Management**: 50+ models with TensorRT support. Solution: Lazy loading, optional TensorRT

### Code Structure

The Gradio app follows this flow:
```python
User uploads images
  └─> Convert to torch tensors
      └─> Detect faces (FaceDetectors)
          └─> Extract embeddings (FaceSwappers)
              └─> Perform swap (via models_processor)
                  └─> Apply restoration (FaceRestorers)
                      └─> Convert back to numpy
                          └─> Return to user
```

### Qt Mocking Strategy

```python
# Before importing app modules, mock Qt
sys.modules['PySide6'] = MockModule()
sys.modules['PySide6.QtCore'].QObject = MockQObject
sys.modules['PySide6.QtCore'].Signal = MockSignal

# Now app.processors can import without errors
from app.processors.models_processor import ModelsProcessor
```

## Next Steps

### Immediate (To Make It Work)
1. Implement simplified face swapping wrapper methods
2. Add error handling for missing models
3. Test basic face detection and swapping
4. Create sample images for testing

### Short Term
1. Add LivePortrait face editing interface
2. Video processing support (frame-by-frame)
3. Batch processing queue
4. Better progress indicators

### Long Term
1. Optimize for cloud deployment (memory management)
2. Add authentication for multi-user support
3. Save/load parameter presets
4. Integration tests and CI/CD

## Troubleshooting

### CUDA/GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
nvidia-smi
```

### Model Download Failures
Models are auto-downloaded from GitHub releases. If downloads fail:
```bash
# Manually download models
python download_models.py

# Or download from: https://github.com/asdf31jsa/VisoMaster-Experimental/releases
```

### Import Errors
Make sure PySide6 is NOT installed for the web version:
```bash
pip uninstall PySide6
```

The Qt mocking system requires that PySide6 is not present.

## Performance Considerations

### GPU Memory
- Face swapping: ~2-4GB VRAM
- With restoration: +1-2GB VRAM
- LivePortrait editing: +2-3GB VRAM
- TensorRT optimization: Can reduce by 20-30%

### Recommended Specs
- **Minimum**: 4GB VRAM (GTX 1650 or better)
- **Recommended**: 8GB VRAM (RTX 3060 or better)
- **Optimal**: 12GB+ VRAM (RTX 3080 or better)

## Contributing

To continue development:

1. Test the current face detection functionality
2. Implement the simplified swap pipeline
3. Add unit tests for core functions
4. Document API endpoints
5. Create example notebooks

## License

Same as original VisoMaster-Experimental project.

## Acknowledgments

- Original VisoMaster-Experimental by asdf31jsa
- LivePortrait team for the face animation models
- Gradio team for the excellent web framework
