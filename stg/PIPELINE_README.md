# STG Pipeline Testing Script

A complete, step-by-step implementation of the Selective Tampering Generation (STG) pipeline with intermediate result saving for easy debugging and testing.

## Features

✅ **Modular Pipeline**: Each step runs independently with saved intermediates  
✅ **Easy Debugging**: All intermediate results saved for inspection  
✅ **Fallback Support**: Works even without CRAFT or doxapy  
✅ **Visualization**: Bounding boxes and results visualized  
✅ **Configurable**: All parameters easily adjustable  
✅ **Production Ready**: Error handling and logging included  

## Pipeline Steps

1. **Load Images** - Read all images from input folder
2. **CRAFT OCR** - Detect text bounding boxes using CRAFT
3. **Binary Masks** - Generate text/background masks using doxapy
4. **Generate Tampering** - Apply STG algorithm
5. **Post-Processing** - Apply JPEG compression and other enhancements

## Installation

### Required Dependencies

```bash
# Basic dependencies
pip install opencv-python numpy tqdm pillow

# For CRAFT OCR (recommended)
pip install craft-text-detector

# For binary mask generation (recommended)
pip install doxapy
```

### Optional: Install CRAFT from source

If `craft-text-detector` doesn't work, install from source:

```bash
git clone https://github.com/clovaai/CRAFT-pytorch.git
cd CRAFT-pytorch
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```bash
# Run with default settings
python pipeline_test.py --input_dir ./test_images --output_dir ./output

# Run with custom parameters
python pipeline_test.py \
    --input_dir ./my_documents \
    --output_dir ./results \
    --max_tamperings 15 \
    --jpeg_quality 90
```

### Command Line Options

```
--input_dir PATH        Input directory containing images (default: test_images)
--output_dir PATH       Output directory for results (default: pipeline_output)
--max_tamperings INT    Maximum tamperings per image (default: 10)
--no_cuda              Disable CUDA for CRAFT
--jpeg_quality INT      JPEG compression quality 1-100 (default: 85)
--quiet                Disable verbose output
```

## Directory Structure

After running, the output directory will contain:

```
pipeline_output/
└── run_20231014_153045/          # Timestamped run
    ├── 1_ocr/                     # OCR results
    │   ├── ocr_results.pk         # Pickle format
    │   └── ocr_results.json       # Human-readable
    ├── 2_masks/                   # Binary masks
    │   ├── image1.png
    │   └── image2.png
    ├── 3_tampered/                # Tampered images
    │   ├── tampered_000001.jpg
    │   ├── mask_000001.png
    │   └── ...
    ├── 4_postprocessed/           # Post-processed images
    │   └── postprocessed_000001.jpg
    └── debug/                     # Debug visualizations
        ├── bbox_image1.jpg        # Images with bounding boxes
        └── bbox_image2.jpg
```

## Configuration

Edit the `PipelineConfig` class in the script to customize:

### OCR Settings (CRAFT)
```python
config.craft_text_threshold = 0.7      # Text detection threshold
config.craft_link_threshold = 0.4      # Link detection threshold
config.craft_use_cuda = True           # Use GPU acceleration
config.craft_canvas_size = 1280        # Max image size
```

### Binary Mask Settings (doxapy)
```python
config.sauvola_window = 75             # Sauvola window size
config.sauvola_k = 0.2                 # Sauvola k parameter
config.invert_mask = True              # Text = white
```

### STG Settings
```python
config.max_tamperings_per_image = 10   # Tamperings per image
config.size_tolerance = 0.1            # ±10% size matching
config.foreground_mean_tolerance = 20  # ±20 intensity units
config.foreground_std_tolerance = 4    # ±4 intensity units
config.background_mean_tolerance = 20  # ±20 intensity units
config.background_std_tolerance = 4    # ±4 intensity units
```

### Post-Processing Settings
```python
config.enable_poisson_blending = False
config.enable_noise_matching = False
config.enable_jpeg_compression = True
config.jpeg_quality = 85
```

## Examples

### Example 1: Test with Sample Images

```bash
# Create test directory
mkdir test_images

# Copy some document images to test_images/
# Then run:
python pipeline_test.py
```

### Example 2: Process Large Dataset

```bash
python pipeline_test.py \
    --input_dir /path/to/large/dataset \
    --output_dir ./large_results \
    --max_tamperings 20 \
    --no_cuda
```

### Example 3: High Quality Output

```bash
python pipeline_test.py \
    --input_dir ./documents \
    --jpeg_quality 95 \
    --max_tamperings 5
```

## Fallback Mode

If CRAFT or doxapy are not available, the script automatically uses fallback methods:

- **CRAFT OCR** → Simple contour-based text detection
- **doxapy Sauvola** → OpenCV Otsu thresholding

These work but produce lower quality results.

## Debugging

### Check Intermediate Results

1. **OCR Results**: Check `1_ocr/ocr_results.json` to see detected text regions
2. **Binary Masks**: Inspect images in `2_masks/` to verify text segmentation
3. **Bounding Boxes**: See `debug/bbox_*.jpg` for visualized detections
4. **Tampered Images**: View `3_tampered/` to see generated forgeries
5. **Ground Truth**: Check `3_tampered/mask_*.png` for tampered region masks

### Common Issues

**No text regions detected:**
- Check OCR results in `1_ocr/`
- Try adjusting `craft_text_threshold` (lower = more detections)
- Use visualization in `debug/` folder

**Poor quality masks:**
- Check masks in `2_masks/`
- Adjust `sauvola_window` (larger = smoother)
- Adjust `sauvola_k` (higher = more text detected)

**No tampering generated:**
- Check if images have enough compatible text regions
- Lower `size_tolerance` or statistical tolerances
- Enable verbose mode to see detailed logs

**Out of memory:**
- Use `--no_cuda` to disable GPU
- Reduce `craft_canvas_size`
- Process images in smaller batches

## Testing Individual Steps

You can test each step independently by modifying the script:

```python
# Test only OCR
image_paths = load_images(config)
ocr_results = run_craft_ocr(image_paths, config)
# Stop here, check results

# Test only masks
mask_paths = generate_binary_masks(image_paths, config)
# Stop here, check results
```

## Performance

**Typical Performance** (on modern CPU):
- OCR: ~2-5 seconds per image (with CRAFT)
- Masks: ~0.5-1 second per image
- Tampering: ~5-10 seconds per image
- Total: ~10-20 images per minute

**With GPU** (CUDA enabled):
- OCR: ~0.5-1 second per image
- Total: ~30-50 images per minute

## Extending the Pipeline

### Add Custom OCR Backend

```python
def run_custom_ocr(image_paths, config):
    """Add your custom OCR implementation"""
    ocr_results = {}
    for img_path in image_paths:
        # Your OCR code here
        boxes = your_ocr_function(img_path)
        ocr_results[img_path] = boxes
    return ocr_results
```

### Add Custom Post-Processing

```python
def custom_postprocessing(tampered_images, config):
    """Add your custom post-processing"""
    for item in tampered_images:
        img = cv2.imread(item['image'])
        # Your processing here
        cv2.imwrite(item['image'], img)
```

### Add New Pipeline Step

```python
# In run_pipeline(), add after step 5:
# Step 6: Custom processing
print("STEP 6: Custom Processing")
custom_results = your_custom_function(tampered_images, config)
```

## Troubleshooting

### Import Errors

```bash
# If craft_text_detector not found:
pip install craft-text-detector

# If doxapy not found:
pip install doxapy

# If cv2 not found:
pip install opencv-python
```

### CRAFT Model Download

First run will download CRAFT model (~80MB). If it fails:
1. Download manually from: https://drive.google.com/file/d/1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ
2. Place in: `~/.craft_text_detector/weights/craft_mlt_25k.pth`

### Memory Issues

Reduce memory usage:
```python
config.craft_canvas_size = 640  # Instead of 1280
config.max_tamperings_per_image = 5  # Instead of 10
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@inproceedings{qu2023towards,
  title={Towards Robust Tampered Text Detection in Document Image: New Dataset and New Solution},
  author={Qu, Chenfan and Liu, Chongyu and Liu, Yuliang and Chen, Xinhong and Peng, Dezhi and Guo, Fengjun and Jin, Lianwen},
  booktitle={CVPR},
  year={2023}
}
```

## License

This code is provided for research purposes. The DocTamper dataset and methodology are subject to the original paper's license.

## Support

For issues or questions:
1. Check the debugging section above
2. Review intermediate results in output directories
3. Enable verbose mode: `--quiet` flag removed
4. Check the IMPROVEMENTS.md for algorithm details
