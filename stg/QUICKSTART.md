# STG Complete Implementation - Quick Start Guide

## 📁 Files Overview

Your STG implementation now includes:

### Core Files
- **`stg.py`** - Original toy code with bug fixes and detailed annotations
- **`stg_improved.py`** - Production-ready modular implementation
- **`pipeline_test.py`** - Complete end-to-end testing pipeline ⭐ **START HERE**
- **`test_dependencies.py`** - Dependency checker

### Documentation
- **`IMPROVEMENTS.md`** - Detailed explanation of all fixes and improvements
- **`PIPELINE_README.md`** - Complete pipeline documentation
- **`Readme.md`** - Original STG description

## 🚀 Quick Start (3 Steps)

### Step 1: Check Dependencies

```bash
cd e:\signate\DocTamper\stg
python test_dependencies.py
```

This will show you what's installed and what's missing.

### Step 2: Install Missing Packages

```bash
# Essential (required)
pip install opencv-python numpy pillow tqdm

# Optional but recommended
pip install craft-text-detector doxapy

# For GPU support (optional)
pip install torch torchvision
```

### Step 3: Run the Pipeline

```bash
# Create test directory and add images
mkdir test_images
# Copy some document images to test_images/

# Run the pipeline
python pipeline_test.py --input_dir test_images --output_dir output
```

## 📊 What Each File Does

### `pipeline_test.py` - **Best for Testing** ⭐

**Use when:** You want to test the complete pipeline step-by-step

**Features:**
- ✅ Automatic CRAFT OCR detection
- ✅ Automatic binary mask generation
- ✅ Automatic tampering generation
- ✅ All intermediate results saved
- ✅ Visualizations for debugging
- ✅ Works with or without CRAFT/doxapy

**Example:**
```bash
python pipeline_test.py \
    --input_dir ./my_documents \
    --output_dir ./results \
    --max_tamperings 15 \
    --jpeg_quality 90
```

### `stg_improved.py` - **Best for Production**

**Use when:** You have OCR results and masks ready

**Features:**
- ✅ Clean modular code
- ✅ Configuration class
- ✅ Complete pseudo-code for all missing functions
- ✅ Ready to extend

**Example:**
```python
from stg_improved import STGConfig, generate_tampering

config = STGConfig()
config.OCR_PICKLE_PATH = 'my_ocr.pk'
config.MAX_TAMPERINGS_PER_IMAGE = 15

generate_tampering(config)
```

### `stg.py` - **Best for Understanding**

**Use when:** You want to understand the algorithm

**Features:**
- ✅ Heavily annotated original code
- ✅ Bug fixes applied
- ✅ Detailed comments

## 🔧 Configuration Quick Reference

### Basic Settings
```python
# In pipeline_test.py or stg_improved.py

# How many tampering operations per output image
max_tamperings_per_image = 10

# Size matching tolerance (±10%)
size_tolerance = 0.1

# Appearance matching tolerances
foreground_mean_tolerance = 20  # Text color similarity
background_mean_tolerance = 20  # Background color similarity
```

### OCR Settings (CRAFT)
```python
craft_text_threshold = 0.7      # Lower = more text detected
craft_link_threshold = 0.4      # Text grouping
craft_use_cuda = True           # Use GPU
```

### Mask Settings (doxapy)
```python
sauvola_window = 75             # Window size for adaptive threshold
sauvola_k = 0.2                 # Sauvola k parameter
```

## 📝 Output Structure

After running `pipeline_test.py`:

```
output/
└── run_20231014_153045/
    ├── 1_ocr/
    │   ├── ocr_results.pk          # OCR bounding boxes (pickle)
    │   └── ocr_results.json        # OCR bounding boxes (readable)
    │
    ├── 2_masks/
    │   ├── image1.png              # Binary masks
    │   └── image2.png
    │
    ├── 3_tampered/
    │   ├── tampered_000001.jpg     # Tampered images
    │   ├── mask_000001.png         # Ground truth masks
    │   └── ...
    │
    ├── 4_postprocessed/
    │   └── postprocessed_*.jpg     # After JPEG compression
    │
    └── debug/
        └── bbox_*.jpg              # Visualized bounding boxes
```

## 🐛 Common Issues & Solutions

### Issue: "No images found"
**Solution:** Make sure your input directory has image files (.jpg, .png, etc.)

### Issue: "CRAFT not found"
**Solution:** Install with `pip install craft-text-detector` OR the script will use fallback

### Issue: "doxapy not found"
**Solution:** Install with `pip install doxapy` OR the script will use Otsu thresholding

### Issue: "No tampering generated"
**Solution:** 
- Check if images have text (look at debug/bbox_*.jpg)
- Lower the tolerance values
- Check masks in 2_masks/ directory

### Issue: "Out of memory"
**Solution:** Use `--no_cuda` flag to run on CPU

## 🔍 Debugging Tips

### Check OCR Results
```bash
# View JSON file
cat output/run_*/1_ocr/ocr_results.json

# Or in Python
import json
with open('output/run_*/1_ocr/ocr_results.json') as f:
    data = json.load(f)
    print(f"Found {len(data)} images")
```

### Check Masks
```bash
# View masks to verify text segmentation
# Open images in 2_masks/ directory
```

### Check Bounding Boxes
```bash
# View bbox visualizations
# Open images in debug/ directory
```

### Enable Verbose Mode
```bash
# See detailed logs
python pipeline_test.py --input_dir test_images
# (verbose is on by default, use --quiet to disable)
```

## 📚 Next Steps

### For Research/Testing
1. Use `pipeline_test.py` with your document images
2. Check intermediate results in output directories
3. Adjust configuration based on results
4. Generate large dataset for training

### For Production
1. Implement pseudo-code functions in `stg_improved.py`
2. Add your custom OCR backend
3. Add your custom post-processing
4. Optimize for your use case

### For Contributing
1. Read `IMPROVEMENTS.md` for detailed changes
2. Fix any remaining TODOs
3. Add unit tests
4. Submit pull request to original repo

## 🎯 Testing Checklist

Before running on large dataset:

- [ ] Run `test_dependencies.py` - all checks pass
- [ ] Test with 1-2 images first
- [ ] Check OCR detects text regions (`debug/bbox_*.jpg`)
- [ ] Check masks look correct (`2_masks/`)
- [ ] Check tampering generated (`3_tampered/`)
- [ ] Adjust configuration if needed
- [ ] Run on full dataset

## 💡 Pro Tips

1. **Start Small**: Test with 1-2 images first
2. **Check Intermediates**: Always inspect intermediate results
3. **Use Visualization**: Enable bbox visualization for debugging
4. **Adjust Tolerances**: If no tampering generated, relax constraints
5. **Save Everything**: Keep intermediate results for analysis
6. **Use GPU**: Install PyTorch for CUDA support (much faster)

## 📞 Getting Help

1. **Check intermediate results** - Most issues are visible there
2. **Review PIPELINE_README.md** - Detailed troubleshooting guide
3. **Review IMPROVEMENTS.md** - Algorithm details
4. **Check original paper** - For methodology questions

## 🎉 You're Ready!

Run this to get started:
```bash
cd e:\signate\DocTamper\stg
python test_dependencies.py
python pipeline_test.py --input_dir test_images
```

Good luck with your tampering generation! 🚀
