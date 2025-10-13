"""
Selective Tampering Generation (STG) - IMPROVED Implementation

This is an enhanced version of the STG toy code with bug fixes and complete
pseudo-code for missing components. This implementation:

1. FIXES the critical bug where source region statistics were incorrectly computed
2. ADDS validation checks for insufficient foreground/background pixels
3. ADDS detailed pseudo-code for all missing post-processing functions
4. ADDS configuration options for reproducible generation
5. ADDS better error handling and logging

Key Improvements over original:
- Correctly computes source region statistics (not target)
- Validates regions have enough pixels before computing statistics
- Comprehensive pseudo-code for: OCR, binarization, blending, noise matching,
  JPEG compression, filtering, color transfer, and parallelization
- Better documentation of the algorithm and constraints
- Ready for extension to full implementation

Based on: Qu et al., "Towards Robust Tampered Text Detection in Document Image:
New Dataset and New Solution", CVPR 2023
"""

import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
import random

# ============================================================================
# CONFIGURATION
# ============================================================================

class STGConfig:
    """Configuration parameters for Selective Tampering Generation"""
    
    # Input/Output paths
    OCR_PICKLE_PATH = 'ocr.pk'
    OUTPUT_IMG_DIR = 'tamp_imgs'
    OUTPUT_MASK_DIR = 'tamp_masks'
    
    # Generation parameters
    MAX_TAMPERINGS_PER_IMAGE = 10  # Number of copy-move operations per output image
    
    # Compatibility constraints
    SIZE_TOLERANCE = 0.1          # ±10% for width/height matching
    FOREGROUND_MEAN_TOLERANCE = 20  # ±20 intensity units for text mean
    FOREGROUND_STD_TOLERANCE = 4    # ±4 intensity units for text std
    BACKGROUND_MEAN_TOLERANCE = 20  # ±20 intensity units for background mean
    BACKGROUND_STD_TOLERANCE = 4    # ±4 intensity units for background std
    
    # Validation thresholds
    MIN_FOREGROUND_PIXELS = 10    # Minimum text pixels required
    MIN_BACKGROUND_PIXELS = 10    # Minimum background pixels required
    
    # Post-processing (for full implementation)
    ENABLE_POISSON_BLENDING = False
    ENABLE_NOISE_MATCHING = False
    ENABLE_JPEG_COMPRESSION = False
    JPEG_QUALITY = 85
    ENABLE_FILTERING = False
    
    # Debug options
    VERBOSE = True
    SAVE_DEBUG_INFO = False

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_output_dirs(config):
    """Create output directories if they don't exist"""
    os.makedirs(config.OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_MASK_DIR, exist_ok=True)
    if config.VERBOSE:
        print(f"Output directories created:")
        print(f"  - Images: {config.OUTPUT_IMG_DIR}")
        print(f"  - Masks: {config.OUTPUT_MASK_DIR}")

def validate_region_statistics(msk, not_msk, config, region_name="region"):
    """
    Validate that a region has sufficient pixels for statistical computation.
    
    Args:
        msk: Boolean mask of foreground (text) pixels
        not_msk: Boolean mask of background pixels
        config: Configuration object
        region_name: Name for error messages
        
    Returns:
        True if valid, False otherwise
    """
    fg_pixels = msk.sum()
    bg_pixels = not_msk.sum()
    
    if fg_pixels < config.MIN_FOREGROUND_PIXELS:
        if config.VERBOSE:
            print(f"  [SKIP] {region_name}: Insufficient foreground pixels ({fg_pixels})")
        return False
    
    if bg_pixels < config.MIN_BACKGROUND_PIXELS:
        if config.VERBOSE:
            print(f"  [SKIP] {region_name}: Insufficient background pixels ({bg_pixels})")
        return False
    
    return True

def compute_region_statistics(region_img, region_mask):
    """
    Compute foreground and background statistics for a region.
    
    Args:
        region_img: Image patch
        region_mask: Boolean mask (True=foreground/text, False=background)
        
    Returns:
        Tuple of (fg_mean, fg_std, bg_mean, bg_std)
    """
    fg_mask = region_mask
    bg_mask = np.logical_not(region_mask)
    
    fg_mean = region_img[fg_mask].mean().astype(np.float32)
    fg_std = region_img[fg_mask].std().astype(np.float32)
    bg_mean = region_img[bg_mask].mean().astype(np.float32)
    bg_std = region_img[bg_mask].std().astype(np.float32)
    
    return fg_mean, fg_std, bg_mean, bg_std

def check_compatibility(target_stats, source_stats, target_size, source_size, config):
    """
    Check if source and target regions are compatible for copy-move tampering.
    
    Args:
        target_stats: (fg_mean, fg_std, bg_mean, bg_std) for target
        source_stats: (fg_mean, fg_std, bg_mean, bg_std) for source
        target_size: (width, height) of target region
        source_size: (width, height) of source region
        config: Configuration object
        
    Returns:
        True if compatible, False otherwise
    """
    tgt_fg_mean, tgt_fg_std, tgt_bg_mean, tgt_bg_std = target_stats
    src_fg_mean, src_fg_std, src_bg_mean, src_bg_std = source_stats
    tgt_w, tgt_h = target_size
    src_w, src_h = source_size
    
    # Size constraints: target size should be within tolerance of source size
    size_tolerance = config.SIZE_TOLERANCE
    w_min = src_w * (1 - size_tolerance)
    w_max = src_w * (1 + size_tolerance)
    h_min = src_h * (1 - size_tolerance)
    h_max = src_h * (1 + size_tolerance)
    
    if not (w_min <= tgt_w <= w_max):
        return False
    if not (h_min <= tgt_h <= h_max):
        return False
    
    # Statistical constraints: target and source should have similar appearance
    # Check foreground (text) mean
    fg_mean_diff = abs(tgt_fg_mean - src_fg_mean)
    if fg_mean_diff > config.FOREGROUND_MEAN_TOLERANCE:
        return False
    
    # Check foreground (text) std
    fg_std_diff = abs(tgt_fg_std - src_fg_std)
    if fg_std_diff > config.FOREGROUND_STD_TOLERANCE:
        return False
    
    # Check background mean
    bg_mean_diff = abs(tgt_bg_mean - src_bg_mean)
    if bg_mean_diff > config.BACKGROUND_MEAN_TOLERANCE:
        return False
    
    # Check background std
    bg_std_diff = abs(tgt_bg_std - src_bg_std)
    if bg_std_diff > config.BACKGROUND_STD_TOLERANCE:
        return False
    
    return True

# ============================================================================
# MAIN TAMPERING GENERATION FUNCTION
# ============================================================================

def generate_tampering(config):
    """
    Main function to generate tampered document images using Selective Tampering Generation.
    
    Algorithm:
    1. Load OCR bounding boxes and binary masks
    2. For each image:
        a. For each text region (target):
            i. Compute target appearance statistics
            ii. Find compatible source regions based on size and appearance
            iii. Copy-paste compatible source to target location
            iv. Update ground truth mask
        b. Save tampered image and ground truth mask
    
    Args:
        config: STGConfig object with parameters
    """
    # Create output directories
    create_output_dirs(config)
    
    # Load OCR bounding boxes
    if not os.path.exists(config.OCR_PICKLE_PATH):
        raise FileNotFoundError(
            f"OCR pickle file not found: {config.OCR_PICKLE_PATH}\n"
            f"Please run OCR detection and generate the pickle file first.\n"
            f"See pseudo-code in this file for OCR generation example."
        )
    
    with open(config.OCR_PICKLE_PATH, 'rb') as f:
        ocr_data = pickle.load(f)
    
    if config.VERBOSE:
        print(f"\nLoaded OCR data for {len(ocr_data)} images")
        print(f"Starting tampering generation with config:")
        print(f"  - Max tamperings per image: {config.MAX_TAMPERINGS_PER_IMAGE}")
        print(f"  - Size tolerance: ±{config.SIZE_TOLERANCE*100}%")
        print(f"  - Foreground mean tolerance: ±{config.FOREGROUND_MEAN_TOLERANCE}")
        print(f"  - Foreground std tolerance: ±{config.FOREGROUND_STD_TOLERANCE}")
        print(f"  - Background mean tolerance: ±{config.BACKGROUND_MEAN_TOLERANCE}")
        print(f"  - Background std tolerance: ±{config.BACKGROUND_STD_TOLERANCE}")
        print()
    
    img_cnt = 0  # Counter for output images
    total_tamperings = 0  # Total number of successful tampering operations
    
    # Process each image
    for img_path, bboxes in tqdm(ocr_data.items(), desc="Generating tamperings"):
        if config.VERBOSE:
            print(f"\nProcessing: {img_path}")
            print(f"  Found {len(bboxes)} text regions")
        
        # Load original image
        img_original = cv2.imread(img_path)
        if img_original is None:
            print(f"  [ERROR] Could not load image: {img_path}")
            continue
        
        h, w = img_original.shape[:2]
        
        # Load pristine copy (for copying regions)
        img_pristine = cv2.imread(img_path)
        
        # Load binary mask
        mask_path = img_path.replace('imgs', 'masks').replace('.jpg', '.png')
        if not os.path.exists(mask_path):
            print(f"  [ERROR] Mask not found: {mask_path}")
            continue
        
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            print(f"  [ERROR] Could not load mask: {mask_path}")
            continue
        
        # Convert to boolean: True=text (foreground), False=background
        mask_bool = (mask > 127)
        mask_bool_inv = np.logical_not(mask_bool)
        
        # Initialize working image and ground truth
        img_tampered = img_original.copy()
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        
        tampering_count = 0  # Count for current image
        
        # Try to apply tampering to each text region
        for target_idx, target_bbox in enumerate(bboxes):
            x1, y1, w1, h1 = target_bbox
            
            # Extract target region
            target_region = img_tampered[y1:y1+h1, x1:x1+w1]
            target_mask = mask_bool[y1:y1+h1, x1:x1+w1]
            target_mask_inv = mask_bool_inv[y1:y1+h1, x1:x1+w1]
            
            # Validate target region
            if not validate_region_statistics(target_mask, target_mask_inv, config, f"Target {target_idx}"):
                continue
            
            # Compute target statistics
            target_stats = compute_region_statistics(target_region, target_mask)
            
            # Find compatible source regions
            for source_idx, source_bbox in enumerate(bboxes):
                # Skip same region
                if target_idx == source_idx:
                    continue
                
                x2, y2, w2, h2 = source_bbox
                
                # Extract source region from pristine image
                source_region = img_pristine[y2:y2+h2, x2:x2+w2]
                source_mask = mask_bool[y2:y2+h2, x2:x2+w2]
                source_mask_inv = mask_bool_inv[y2:y2+h2, x2:x2+w2]
                
                # Validate source region
                if not validate_region_statistics(source_mask, source_mask_inv, config, f"Source {source_idx}"):
                    continue
                
                # Compute source statistics
                source_stats = compute_region_statistics(source_region, source_mask)
                
                # Check compatibility
                if check_compatibility(
                    target_stats, 
                    source_stats, 
                    (w1, h1), 
                    (w2, h2), 
                    config
                ):
                    # APPLY TAMPERING: Copy source region to target location
                    if config.VERBOSE:
                        print(f"  [TAMPER] Target {target_idx} <- Source {source_idx}")
                        print(f"    Target region: ({x1},{y1}) size ({w1},{h1})")
                        print(f"    Source region: ({x2},{y2}) size ({w2},{h2})")
                    
                    # Resize source to match target dimensions and paste
                    source_resized = cv2.resize(source_region, (w1, h1))
                    img_tampered[y1:y1+h1, x1:x1+w1] = source_resized
                    
                    # Update ground truth mask
                    gt_mask[y1:y1+h1, x1:x1+w1] = 255
                    
                    tampering_count += 1
                    total_tamperings += 1
                    
                    # Check if we've reached the maximum tamperings per image
                    if tampering_count >= config.MAX_TAMPERINGS_PER_IMAGE:
                        # Save current tampered image
                        output_img_path = os.path.join(config.OUTPUT_IMG_DIR, f'{img_cnt:06d}.jpg')
                        output_mask_path = os.path.join(config.OUTPUT_MASK_DIR, f'{img_cnt:06d}.png')
                        
                        cv2.imwrite(output_img_path, img_tampered)
                        cv2.imwrite(output_mask_path, gt_mask)
                        
                        if config.VERBOSE:
                            print(f"  [SAVE] Saved image {img_cnt} with {tampering_count} tamperings")
                        
                        img_cnt += 1
                        tampering_count = 0
                        
                        # Reset for next round
                        img_tampered = cv2.imread(img_path)
                        gt_mask = np.zeros((h, w), dtype=np.uint8)
                    
                    # Break after first compatible source (one tampering per target region)
                    break
        
        # Save final image even if it has fewer than max_cnt tamperings
        if tampering_count > 0:
            output_img_path = os.path.join(config.OUTPUT_IMG_DIR, f'{img_cnt:06d}.jpg')
            output_mask_path = os.path.join(config.OUTPUT_MASK_DIR, f'{img_cnt:06d}.png')
            
            cv2.imwrite(output_img_path, img_tampered)
            cv2.imwrite(output_mask_path, gt_mask)
            
            if config.VERBOSE:
                print(f"  [SAVE] Saved final image {img_cnt} with {tampering_count} tamperings")
            
            img_cnt += 1
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Tampering generation complete!")
    print(f"  - Generated {img_cnt} tampered images")
    print(f"  - Total tampering operations: {total_tamperings}")
    print(f"  - Average tamperings per image: {total_tamperings/max(1, img_cnt):.2f}")
    print(f"  - Output directory: {config.OUTPUT_IMG_DIR}")
    print(f"  - Ground truth directory: {config.OUTPUT_MASK_DIR}")
    print(f"{'='*70}\n")

# ============================================================================
# PSEUDO-CODE FOR MISSING COMPONENTS
# ============================================================================

# ----------------------------------------------------------------------------
# 1. OCR DETECTION AND BOUNDING BOX EXTRACTION
# ----------------------------------------------------------------------------

def generate_ocr_pickle_paddleocr(image_dir, output_path='ocr.pk'):
    """
    Generate OCR bounding boxes using PaddleOCR for all images in a directory.
    
    This function is NOT implemented in the toy version.
    To use PaddleOCR:
    1. Install: pip install paddlepaddle paddleocr
    2. Import: from paddleocr import PaddleOCR
    3. Initialize: ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    Args:
        image_dir: Directory containing images
        output_path: Path to save OCR pickle file
        
    Returns:
        Dictionary mapping image paths to bounding box lists
    """
    # from paddleocr import PaddleOCR
    # import glob
    # 
    # # Initialize OCR model
    # ocr_model = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
    # 
    # ocr_results = {}
    # 
    # # Process all images
    # image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    # for image_path in tqdm(image_paths, desc="Running OCR"):
    #     # Run OCR detection (det=True for detection, rec=False to skip recognition)
    #     result = ocr_model.ocr(image_path, det=True, rec=False)
    #     
    #     # Extract bounding boxes
    #     boxes = []
    #     if result and result[0]:  # Check if detection found anything
    #         for line in result:
    #             for detection in line:
    #                 # Detection format: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], confidence]
    #                 bbox_points = detection[0]
    #                 
    #                 # Convert polygon to axis-aligned bounding box
    #                 x_coords = [pt[0] for pt in bbox_points]
    #                 y_coords = [pt[1] for pt in bbox_points]
    #                 
    #                 x_min = int(min(x_coords))
    #                 y_min = int(min(y_coords))
    #                 x_max = int(max(x_coords))
    #                 y_max = int(max(y_coords))
    #                 
    #                 width = x_max - x_min
    #                 height = y_max - y_min
    #                 
    #                 # Store as [x, y, width, height]
    #                 boxes.append([x_min, y_min, width, height])
    #     
    #     ocr_results[image_path] = boxes
    # 
    # # Save to pickle
    # with open(output_path, 'wb') as f:
    #     pickle.dump(ocr_results, f)
    # 
    # print(f"OCR results saved to {output_path}")
    # print(f"Processed {len(ocr_results)} images")
    # 
    # return ocr_results
    pass

def generate_ocr_pickle_tesseract(image_dir, output_path='ocr.pk'):
    """
    Generate OCR bounding boxes using Tesseract for all images in a directory.
    
    This function is NOT implemented in the toy version.
    To use Tesseract:
    1. Install Tesseract: https://github.com/tesseract-ocr/tesseract
    2. Install Python wrapper: pip install pytesseract
    3. Import: import pytesseract
    
    Args:
        image_dir: Directory containing images
        output_path: Path to save OCR pickle file
        
    Returns:
        Dictionary mapping image paths to bounding box lists
    """
    # import pytesseract
    # import glob
    # from PIL import Image
    # 
    # ocr_results = {}
    # 
    # # Process all images
    # image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    # for image_path in tqdm(image_paths, desc="Running OCR"):
    #     # Load image
    #     img = Image.open(image_path)
    #     
    #     # Run Tesseract OCR with bounding box output
    #     # Output format: level, page_num, block_num, par_num, line_num, word_num,
    #     #                left, top, width, height, conf, text
    #     ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    #     
    #     # Extract bounding boxes (filter by confidence)
    #     boxes = []
    #     for i in range(len(ocr_data['text'])):
    #         # Skip empty detections and low confidence
    #         if int(ocr_data['conf'][i]) < 0:
    #             continue
    #         
    #         x = ocr_data['left'][i]
    #         y = ocr_data['top'][i]
    #         w = ocr_data['width'][i]
    #         h = ocr_data['height'][i]
    #         
    #         # Store as [x, y, width, height]
    #         boxes.append([x, y, w, h])
    #     
    #     ocr_results[image_path] = boxes
    # 
    # # Save to pickle
    # with open(output_path, 'wb') as f:
    #     pickle.dump(ocr_results, f)
    # 
    # print(f"OCR results saved to {output_path}")
    # print(f"Processed {len(ocr_results)} images")
    # 
    # return ocr_results
    pass

# ----------------------------------------------------------------------------
# 2. BINARY MASK GENERATION
# ----------------------------------------------------------------------------

def generate_binary_masks_sauvola(image_dir, mask_dir, window=75, k=0.2):
    """
    Generate binary masks using Sauvola adaptive thresholding.
    
    This function is NOT implemented in the toy version.
    To use Sauvola binarization:
    1. Install: pip install doxapy
    2. Import: import doxapy
    
    Args:
        image_dir: Directory containing images
        mask_dir: Directory to save binary masks
        window: Window size for adaptive thresholding (default: 75)
        k: Sauvola parameter (default: 0.2)
    """
    # import doxapy
    # import glob
    # 
    # os.makedirs(mask_dir, exist_ok=True)
    # 
    # # Initialize Sauvola binarization model
    # model = doxapy.Binarization(doxapy.Binarization.Algorithms.SAUVOLA)
    # 
    # # Process all images
    # image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    # for image_path in tqdm(image_paths, desc="Generating masks"):
    #     # Load image
    #     img = cv2.imread(image_path)
    #     
    #     # Initialize binarization
    #     model.initialize(img)
    #     
    #     # Create output mask
    #     mask = np.zeros_like(img)
    #     
    #     # Apply Sauvola binarization
    #     model.to_binary(mask, {"window": window, "k": k})
    #     
    #     # Save mask
    #     mask_filename = os.path.basename(image_path).replace('.jpg', '.png')
    #     mask_path = os.path.join(mask_dir, mask_filename)
    #     cv2.imwrite(mask_path, mask)
    # 
    # print(f"Generated {len(image_paths)} binary masks in {mask_dir}")
    pass

def generate_binary_masks_otsu(image_dir, mask_dir):
    """
    Generate binary masks using Otsu's thresholding.
    
    This is a simpler alternative to Sauvola that works well for documents
    with uniform lighting.
    
    Args:
        image_dir: Directory containing images
        mask_dir: Directory to save binary masks
    """
    # import glob
    # 
    # os.makedirs(mask_dir, exist_ok=True)
    # 
    # # Process all images
    # image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    # for image_path in tqdm(image_paths, desc="Generating masks"):
    #     # Load image as grayscale
    #     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #     
    #     # Apply Otsu's thresholding
    #     _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     
    #     # Invert if needed (text should be white)
    #     if mask.mean() > 127:  # More white than black
    #         mask = 255 - mask
    #     
    #     # Save mask
    #     mask_filename = os.path.basename(image_path).replace('.jpg', '.png')
    #     mask_path = os.path.join(mask_dir, mask_filename)
    #     cv2.imwrite(mask_path, mask)
    # 
    # print(f"Generated {len(image_paths)} binary masks in {mask_dir}")
    pass

# ----------------------------------------------------------------------------
# 3. POST-PROCESSING: POISSON BLENDING
# ----------------------------------------------------------------------------

def apply_poisson_blending(tampered_img, gt_mask, blend_method='normal'):
    """
    Apply Poisson blending to seamlessly blend tampered regions.
    
    Poisson blending performs gradient-domain editing to blend the copied
    region into the surrounding context without visible seams.
    
    This function is NOT implemented in the toy version.
    To use:
    1. OpenCV's seamlessClone function
    2. Requires: source image, destination image, mask, and center point
    
    Args:
        tampered_img: Image with copy-move tampering
        gt_mask: Ground truth mask (255=tampered, 0=authentic)
        blend_method: 'normal' or 'mixed' clone
        
    Returns:
        Blended image
    """
    # # Find connected components in ground truth mask
    # num_labels, labels = cv2.connectedComponents(gt_mask)
    # 
    # result = tampered_img.copy()
    # 
    # # Blend each tampered region separately
    # for label_id in range(1, num_labels):  # Skip 0 (background)
    #     # Extract region mask
    #     region_mask = (labels == label_id).astype(np.uint8) * 255
    #     
    #     # Find center of region
    #     moments = cv2.moments(region_mask)
    #     if moments['m00'] == 0:
    #         continue
    #     center_x = int(moments['m10'] / moments['m00'])
    #     center_y = int(moments['m01'] / moments['m00'])
    #     center = (center_x, center_y)
    #     
    #     # Apply Poisson blending
    #     blend_flag = cv2.NORMAL_CLONE if blend_method == 'normal' else cv2.MIXED_CLONE
    #     result = cv2.seamlessClone(result, result, region_mask, center, blend_flag)
    # 
    # return result
    pass

# ----------------------------------------------------------------------------
# 4. POST-PROCESSING: NOISE MATCHING
# ----------------------------------------------------------------------------

def match_noise_characteristics(tampered_img, gt_mask, window_size=50):
    """
    Match noise characteristics between tampered and surrounding regions.
    
    This analyzes the noise in surrounding areas and adds similar noise to
    the tampered regions to make them less detectable.
    
    This function is NOT implemented in the toy version.
    
    Args:
        tampered_img: Image with copy-move tampering
        gt_mask: Ground truth mask (255=tampered, 0=authentic)
        window_size: Size of window for noise estimation
        
    Returns:
        Image with matched noise
    """
    # result = tampered_img.copy()
    # 
    # # Find tampered regions
    # num_labels, labels = cv2.connectedComponents(gt_mask)
    # 
    # for label_id in range(1, num_labels):
    #     # Get tampered region mask
    #     region_mask = (labels == label_id)
    #     
    #     # Get bounding box
    #     coords = np.argwhere(region_mask)
    #     y_min, x_min = coords.min(axis=0)
    #     y_max, x_max = coords.max(axis=0)
    #     
    #     # Define surrounding area (excluding tampered region)
    #     y1 = max(0, y_min - window_size)
    #     y2 = min(tampered_img.shape[0], y_max + window_size)
    #     x1 = max(0, x_min - window_size)
    #     x2 = min(tampered_img.shape[1], x_max + window_size)
    #     
    #     surrounding_mask = np.zeros_like(gt_mask, dtype=bool)
    #     surrounding_mask[y1:y2, x1:x2] = True
    #     surrounding_mask = surrounding_mask & ~region_mask
    #     
    #     # Estimate noise in surrounding area
    #     for channel in range(3):
    #         surrounding_pixels = tampered_img[surrounding_mask, channel]
    #         noise_std = np.std(surrounding_pixels - cv2.GaussianBlur(
    #             tampered_img[:,:,channel], (5,5), 0)[surrounding_mask])
    #         
    #         # Add similar noise to tampered region
    #         noise = np.random.normal(0, noise_std, region_mask.sum())
    #         result[region_mask, channel] = np.clip(
    #             result[region_mask, channel] + noise, 0, 255
    #         )
    # 
    # return result.astype(np.uint8)
    pass

# ----------------------------------------------------------------------------
# 5. POST-PROCESSING: JPEG COMPRESSION
# ----------------------------------------------------------------------------

def apply_jpeg_compression(img, quality=85, multiple_rounds=False):
    """
    Apply JPEG compression to introduce realistic artifacts.
    
    Multiple compression rounds with different quality factors simulate
    real-world scenarios where images are compressed multiple times.
    
    This function is NOT implemented in the toy version.
    
    Args:
        img: Input image
        quality: JPEG quality factor (1-100)
        multiple_rounds: If True, apply 2-3 compression rounds
        
    Returns:
        Compressed image
    """
    # import tempfile
    # from PIL import Image
    # 
    # with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as tmp:
    #     if multiple_rounds:
    #         # First compression
    #         q1 = quality + random.randint(-5, 5)
    #         q1 = np.clip(q1, 1, 100)
    #         Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(
    #             tmp, 'JPEG', quality=q1
    #         )
    #         img = Image.open(tmp)
    #         
    #         # Second compression
    #         q2 = quality + random.randint(-10, 10)
    #         q2 = np.clip(q2, 1, 100)
    #         img.save(tmp, 'JPEG', quality=q2)
    #         img = Image.open(tmp)
    #         
    #         # Optional third compression
    #         if random.random() < 0.3:
    #             q3 = quality + random.randint(-5, 5)
    #             q3 = np.clip(q3, 1, 100)
    #             img.save(tmp, 'JPEG', quality=q3)
    #             img = Image.open(tmp)
    #     else:
    #         # Single compression
    #         Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(
    #             tmp, 'JPEG', quality=quality
    #         )
    #         img = Image.open(tmp)
    #     
    #     return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    pass

# ----------------------------------------------------------------------------
# 6. POST-PROCESSING: COLOR/INTENSITY TRANSFER
# ----------------------------------------------------------------------------

def apply_color_transfer(source_patch, target_context_patch):
    """
    Apply color/intensity transfer to match source patch to target context.
    
    This adjusts the color statistics of the source patch to match the
    surrounding context, making the tampering less detectable.
    
    This function is NOT implemented in the toy version.
    
    Args:
        source_patch: Source patch to adjust
        target_context_patch: Target context (surrounding area)
        
    Returns:
        Adjusted source patch
    """
    # result = source_patch.copy().astype(float)
    # 
    # # Method 1: Mean and standard deviation matching
    # for channel in range(3):
    #     src_mean = source_patch[:,:,channel].mean()
    #     src_std = source_patch[:,:,channel].std()
    #     
    #     tgt_mean = target_context_patch[:,:,channel].mean()
    #     tgt_std = target_context_patch[:,:,channel].std()
    #     
    #     # Avoid division by zero
    #     if src_std < 0.1:
    #         src_std = 0.1
    #     
    #     # Transfer color statistics
    #     result[:,:,channel] = (
    #         (result[:,:,channel] - src_mean) * (tgt_std / src_std) + tgt_mean
    #     )
    # 
    # # Clip to valid range
    # result = np.clip(result, 0, 255).astype(np.uint8)
    # 
    # # Method 2: Histogram matching (more sophisticated)
    # # from skimage.exposure import match_histograms
    # # result = match_histograms(source_patch, target_context_patch, multichannel=True)
    # 
    # return result
    pass

# ----------------------------------------------------------------------------
# 7. FILTERING: QUALITY CONTROL
# ----------------------------------------------------------------------------

def filter_low_quality_tampering(tampered_img, original_img, gt_mask, 
                                  max_tampered_ratio=0.5, min_tampered_pixels=100):
    """
    Filter out low-quality or unrealistic tampering examples.
    
    This ensures that only challenging and realistic forgeries are kept
    for training the detection model.
    
    This function is NOT implemented in the toy version.
    
    Args:
        tampered_img: Tampered image
        original_img: Original pristine image
        gt_mask: Ground truth mask
        max_tampered_ratio: Maximum ratio of tampered pixels
        min_tampered_pixels: Minimum number of tampered pixels
        
    Returns:
        True if tampering should be kept, False if discarded
    """
    # # Check 1: Tampered region size
    # tampered_pixels = (gt_mask > 0).sum()
    # total_pixels = gt_mask.size
    # tampered_ratio = tampered_pixels / total_pixels
    # 
    # if tampered_ratio > max_tampered_ratio:
    #     return False  # Too much tampering (unrealistic)
    # 
    # if tampered_pixels < min_tampered_pixels:
    #     return False  # Too little tampering (not meaningful)
    # 
    # # Check 2: Detect obvious artifacts
    # diff = np.abs(tampered_img.astype(float) - original_img.astype(float))
    # tampered_diff = diff[gt_mask > 0].mean()
    # 
    # if tampered_diff > 50:
    #     return False  # Too obvious (large color difference)
    # 
    # # Check 3: Check for copy-move detection using simple methods
    # # If detectable by simple methods, it's a good training example
    # # (This is inverted logic - we want examples that are detectable but not trivial)
    # 
    # return True
    pass

# ----------------------------------------------------------------------------
# 8. PARALLELIZATION
# ----------------------------------------------------------------------------

def generate_tampering_parallel(config, num_workers=4):
    """
    Parallelize tampering generation across multiple processes.
    
    This significantly speeds up generation for large datasets by processing
    multiple images simultaneously.
    
    This function is NOT implemented in the toy version.
    
    Args:
        config: STGConfig object
        num_workers: Number of parallel workers
    """
    # from multiprocessing import Pool, cpu_count
    # import functools
    # 
    # # Load OCR data
    # with open(config.OCR_PICKLE_PATH, 'rb') as f:
    #     ocr_data = pickle.load(f)
    # 
    # # Split data among workers
    # items = list(ocr_data.items())
    # 
    # # Create worker function
    # worker_func = functools.partial(process_single_image, config=config)
    # 
    # # Use multiprocessing pool
    # num_workers = min(num_workers, cpu_count())
    # with Pool(num_workers) as pool:
    #     results = list(tqdm(
    #         pool.imap(worker_func, items),
    #         total=len(items),
    #         desc="Generating tamperings (parallel)"
    #     ))
    # 
    # print(f"Processed {len(results)} images using {num_workers} workers")
    pass

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # Create configuration
    config = STGConfig()
    
    # Generate tampering
    generate_tampering(config)
    
    print("\nNOTE: This is the improved toy implementation.")
    print("For production use, implement the post-processing functions marked with pseudo-code.")
    print("\nKey improvements over original:")
    print("  ✓ Fixed bug in source region statistics computation")
    print("  ✓ Added validation for insufficient foreground/background pixels")
    print("  ✓ Comprehensive pseudo-code for all missing components")
    print("  ✓ Better code organization and documentation")
    print("  ✓ Configuration-based parameters for easy tuning")
