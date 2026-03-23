"""
Full STG Pipeline Testing Script

This script implements the complete Selective Tampering Generation pipeline
with step-by-step execution and intermediate result saving for debugging.

Pipeline Flow:
1. Read images from input folder
2. Run CRAFT OCR detection to get text bounding boxes
3. Generate binary masks using doxapy (Sauvola binarization)
4. Generate tampering using STG algorithm
5. Apply post-processing (optional)
6. Save all intermediate results

Usage:
    python pipeline_test.py --input_dir ./test_images --output_dir ./output

Requirements:
    - CRAFT (for OCR): https://github.com/clovaai/CRAFT-pytorch
    - doxapy: pip install doxapy
    - opencv-python: pip install opencv-python
    - numpy, tqdm, pillow
"""

import os
import sys
import cv2
import pickle
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

class PipelineConfig:
    """Configuration for the complete pipeline"""
    
    def __init__(self):
        # Input/Output directories
        self.input_dir = 'test_images'
        self.output_dir = 'pipeline_output'
        
        # Intermediate output directories
        self.ocr_output_dir = None  # Will be set in init_directories
        self.mask_output_dir = None
        self.tampered_output_dir = None
        self.postprocessed_output_dir = None
        self.debug_output_dir = None
        
        # OCR settings (CRAFT)
        self.craft_model_path = None  # Set to CRAFT model path if not using default
        self.craft_text_threshold = 0.7
        self.craft_link_threshold = 0.4
        self.craft_low_text = 0.4
        self.craft_use_cuda = True
        self.craft_canvas_size = 1280
        self.craft_mag_ratio = 1.5
        
        # Binary mask settings (doxapy)
        self.sauvola_window = 75
        self.sauvola_k = 0.2
        self.invert_mask = True  # Set True if text should be white
        
        # STG settings
        self.max_tamperings_per_image = 10
        self.size_tolerance = 0.1
        self.foreground_mean_tolerance = 20
        self.foreground_std_tolerance = 4
        self.background_mean_tolerance = 20
        self.background_std_tolerance = 4
        self.min_foreground_pixels = 10
        self.min_background_pixels = 10
        
        # Post-processing settings
        self.enable_poisson_blending = False
        self.enable_noise_matching = False
        self.enable_jpeg_compression = True
        self.jpeg_quality = 85
        
        # Debug settings
        self.save_intermediate_results = True
        self.visualize_bboxes = True
        self.verbose = True
        
    def init_directories(self):
        """Initialize all output directories"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output = os.path.join(self.output_dir, f'run_{timestamp}')
        
        self.ocr_output_dir = os.path.join(base_output, '1_ocr')
        self.mask_output_dir = os.path.join(base_output, '2_masks')
        self.tampered_output_dir = os.path.join(base_output, '3_tampered')
        self.postprocessed_output_dir = os.path.join(base_output, '4_postprocessed')
        self.debug_output_dir = os.path.join(base_output, 'debug')
        
        # Create all directories
        for dir_path in [self.ocr_output_dir, self.mask_output_dir, 
                         self.tampered_output_dir, self.postprocessed_output_dir,
                         self.debug_output_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        if self.verbose:
            print(f"Output directories created:")
            print(f"  Base: {base_output}")
            print(f"  OCR: {self.ocr_output_dir}")
            print(f"  Masks: {self.mask_output_dir}")
            print(f"  Tampered: {self.tampered_output_dir}")
            print(f"  Post-processed: {self.postprocessed_output_dir}")
            print(f"  Debug: {self.debug_output_dir}")
            print()

# ============================================================================
# STEP 1: READ IMAGES FROM INPUT FOLDER
# ============================================================================

def load_images(config):
    """
    Load all images from input directory.
    
    Args:
        config: PipelineConfig object
        
    Returns:
        List of image paths
    """
    print("="*70)
    print("STEP 1: Loading images from input directory")
    print("="*70)
    
    if not os.path.exists(config.input_dir):
        raise FileNotFoundError(f"Input directory not found: {config.input_dir}")
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # Find all images
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(config.input_dir).glob(f'*{ext}'))
        image_paths.extend(Path(config.input_dir).glob(f'*{ext.upper()}'))
    
    image_paths = [str(p) for p in image_paths]
    image_paths.sort()
    
    print(f"Found {len(image_paths)} images in {config.input_dir}")
    
    if len(image_paths) == 0:
        print("WARNING: No images found. Please check the input directory.")
        return []
    
    # Display first few images
    print("\nFirst few images:")
    for i, img_path in enumerate(image_paths[:5]):
        print(f"  {i+1}. {os.path.basename(img_path)}")
    
    if len(image_paths) > 5:
        print(f"  ... and {len(image_paths) - 5} more")
    
    print()
    return image_paths

# ============================================================================
# STEP 2: RUN CRAFT OCR DETECTION
# ============================================================================

def run_craft_ocr(image_paths, config):
    """
    Run CRAFT OCR detection on all images.
    
    Args:
        image_paths: List of image paths
        config: PipelineConfig object
        
    Returns:
        Dictionary mapping image paths to bounding box lists
    """
    print("="*70)
    print("STEP 2: Running CRAFT OCR detection")
    print("="*70)
    
    try:
        # Import CRAFT
        # Assuming CRAFT is installed and available
        # If not, provide installation instructions
        try:
            import craft_text_detector
            from craft_text_detector import Craft
        except ImportError:
            print("ERROR: CRAFT not found. Please install it:")
            print("  pip install craft-text-detector")
            print("OR clone from: https://github.com/fcakyon/craft-text-detector")
            print("\nUsing fallback: simple text detection")
            return run_fallback_ocr(image_paths, config)
        
        # Initialize CRAFT
        print("Initializing CRAFT model...")
        craft = Craft(
            output_dir=config.ocr_output_dir,
            crop_type="poly",
            cuda=config.craft_use_cuda,
            text_threshold=config.craft_text_threshold,
            link_threshold=config.craft_link_threshold,
            low_text=config.craft_low_text,
            canvas_size=config.craft_canvas_size,
            mag_ratio=config.craft_mag_ratio
        )
        
        ocr_results = {}
        
        # Process each image
        for img_path in tqdm(image_paths, desc="Running CRAFT OCR"):
            # Run CRAFT detection
            prediction_result = craft.detect_text(img_path)
            
            # Extract bounding boxes
            boxes = []
            if prediction_result and 'boxes' in prediction_result:
                for box in prediction_result['boxes']:
                    # box format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    x_coords = [pt[0] for pt in box]
                    y_coords = [pt[1] for pt in box]
                    
                    x_min = int(min(x_coords))
                    y_min = int(min(y_coords))
                    x_max = int(max(x_coords))
                    y_max = int(max(y_coords))
                    
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    # Store as [x, y, width, height]
                    boxes.append([x_min, y_min, width, height])
            
            ocr_results[img_path] = boxes
            
            if config.verbose:
                print(f"  {os.path.basename(img_path)}: {len(boxes)} text regions detected")
            
            # Save visualization if enabled
            if config.visualize_bboxes:
                visualize_bboxes(img_path, boxes, config.debug_output_dir)
        
        # Save OCR results
        ocr_pickle_path = os.path.join(config.ocr_output_dir, 'ocr_results.pk')
        with open(ocr_pickle_path, 'wb') as f:
            pickle.dump(ocr_results, f)
        
        # Save as JSON for human readability
        ocr_json_path = os.path.join(config.ocr_output_dir, 'ocr_results.json')
        ocr_results_serializable = {
            os.path.basename(k): v for k, v in ocr_results.items()
        }
        with open(ocr_json_path, 'w') as f:
            json.dump(ocr_results_serializable, f, indent=2)
        
        print(f"\nOCR results saved:")
        print(f"  Pickle: {ocr_pickle_path}")
        print(f"  JSON: {ocr_json_path}")
        print(f"  Total images processed: {len(ocr_results)}")
        print(f"  Total text regions: {sum(len(boxes) for boxes in ocr_results.values())}")
        print()
        
        return ocr_results
        
    except Exception as e:
        print(f"ERROR in CRAFT OCR: {e}")
        print("Using fallback OCR detection...")
        return run_fallback_ocr(image_paths, config)

def run_fallback_ocr(image_paths, config):
    """
    Fallback OCR using simple contour detection.
    Use when CRAFT is not available.
    
    Args:
        image_paths: List of image paths
        config: PipelineConfig object
        
    Returns:
        Dictionary mapping image paths to bounding box lists
    """
    print("Using fallback: Simple contour-based text detection")
    
    ocr_results = {}
    
    for img_path in tqdm(image_paths, desc="Fallback OCR"):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Extract bounding boxes
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter small regions
            if w > 10 and h > 10:
                boxes.append([x, y, w, h])
        
        ocr_results[img_path] = boxes
        
        if config.visualize_bboxes:
            visualize_bboxes(img_path, boxes, config.debug_output_dir)
    
    # Save results
    ocr_pickle_path = os.path.join(config.ocr_output_dir, 'ocr_results.pk')
    with open(ocr_pickle_path, 'wb') as f:
        pickle.dump(ocr_results, f)
    
    print(f"Fallback OCR results saved: {ocr_pickle_path}")
    print()
    
    return ocr_results

def visualize_bboxes(img_path, boxes, output_dir):
    """Visualize bounding boxes on image"""
    img = cv2.imread(img_path)
    
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    output_path = os.path.join(
        output_dir, 
        f"bbox_{os.path.basename(img_path)}"
    )
    cv2.imwrite(output_path, img)

# ============================================================================
# STEP 3: GENERATE BINARY MASKS USING DOXAPY
# ============================================================================

def generate_binary_masks(image_paths, config):
    """
    Generate binary masks using doxapy Sauvola binarization.
    
    Args:
        image_paths: List of image paths
        config: PipelineConfig object
        
    Returns:
        Dictionary mapping image paths to mask paths
    """
    print("="*70)
    print("STEP 3: Generating binary masks using doxapy")
    print("="*70)
    
    try:
        import doxapy
    except ImportError:
        print("ERROR: doxapy not found. Please install it:")
        print("  pip install doxapy")
        print("\nUsing fallback: Otsu thresholding")
        return generate_masks_fallback(image_paths, config)
    
    # Initialize Sauvola binarization
    model = doxapy.Binarization(doxapy.Binarization.Algorithms.SAUVOLA)
    
    mask_paths = {}
    
    for img_path in tqdm(image_paths, desc="Generating masks"):
        # Load image
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"  ERROR: Could not load {img_path}")
            continue
        
        # Convert to grayscale before initialize (doxapy requires grayscale uint8)
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
        
        # Initialize binarization
        model.initialize(img_gray)
        
        # Create output mask (grayscale to match input grayscale)
        mask = np.zeros((img_gray.shape[0], img_gray.shape[1]), dtype=np.uint8)
        
        # Apply Sauvola binarization
        model.to_binary(mask, {
            "window": config.sauvola_window, 
            "k": config.sauvola_k
        })
        
        # Invert if needed (text should be white)
        if config.invert_mask:
            if mask.mean() < 127:  # More black than white
                mask = 255 - mask
        
        # Save mask
        mask_filename = os.path.basename(img_path).replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(config.mask_output_dir, mask_filename)
        cv2.imwrite(mask_path, mask)
        
        mask_paths[img_path] = mask_path
        
        if config.verbose:
            print(f"  {os.path.basename(img_path)} -> {mask_filename}")
    
    print(f"\nGenerated {len(mask_paths)} binary masks")
    print(f"Masks saved to: {config.mask_output_dir}")
    print()
    
    return mask_paths

def generate_masks_fallback(image_paths, config):
    """Fallback mask generation using Otsu thresholding"""
    print("Using fallback: Otsu thresholding")
    
    mask_paths = {}
    
    for img_path in tqdm(image_paths, desc="Generating masks (Otsu)"):
        # Load image as grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue
        
        # Apply Otsu's thresholding
        _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if needed
        if mask.mean() < 127:
            mask = 255 - mask
        
        # Save mask
        mask_filename = os.path.basename(img_path).replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(config.mask_output_dir, mask_filename)
        cv2.imwrite(mask_path, mask)
        
        mask_paths[img_path] = mask_path
    
    print(f"Generated {len(mask_paths)} masks using Otsu")
    print()
    
    return mask_paths

# ============================================================================
# STEP 4: GENERATE TAMPERING
# ============================================================================

def generate_tampering(ocr_results, mask_paths, config):
    """
    Generate tampered images using STG algorithm.
    
    Args:
        ocr_results: Dictionary of OCR results
        mask_paths: Dictionary of mask paths
        config: PipelineConfig object
        
    Returns:
        List of generated tampered image paths
    """
    print("="*70)
    print("STEP 4: Generating tampered images using STG")
    print("="*70)
    
    tampered_images = []
    img_cnt = 0
    total_tamperings = 0
    source_bboxes = {}  # Track source boxes for each image
    source_info = {}  # Track detailed source information for label generation
    
    for img_path, bboxes in tqdm(ocr_results.items(), desc="Generating tampering"):
        if config.verbose:
            print(f"\nProcessing: {os.path.basename(img_path)}")
            print(f"  Found {len(bboxes)} text regions")
        
        # Load original image
        img_original = cv2.imread(img_path)
        if img_original is None:
            print(f"  ERROR: Could not load image: {img_path}")
            continue
        
        h, w = img_original.shape[:2]
        
        # Load pristine copy
        img_pristine = cv2.imread(img_path)
        
        # Load mask
        if img_path not in mask_paths:
            print(f"  ERROR: Mask not found for {img_path}")
            continue
        
        mask = cv2.imread(mask_paths[img_path], 0)
        if mask is None:
            print(f"  ERROR: Could not load mask: {mask_paths[img_path]}")
            continue
        
        # Convert to boolean
        mask_bool = (mask > 127)
        mask_bool_inv = np.logical_not(mask_bool)
        
        # Initialize working image and ground truth
        img_tampered = img_original.copy()
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        
        tampering_count = 0
        
        # Try to apply tampering to each text region
        for target_idx, target_bbox in enumerate(bboxes):
            x1, y1, w1, h1 = target_bbox
            
            # Bounds checking
            if x1 < 0 or y1 < 0 or x1+w1 > w or y1+h1 > h:
                continue
            
            # Extract target region
            target_region = img_tampered[y1:y1+h1, x1:x1+w1]
            target_mask = mask_bool[y1:y1+h1, x1:x1+w1]
            target_mask_inv = mask_bool_inv[y1:y1+h1, x1:x1+w1]
            
            # Validate target region
            if target_mask.sum() < config.min_foreground_pixels:
                continue
            if target_mask_inv.sum() < config.min_background_pixels:
                continue
            
            # Compute target statistics
            target_stats = compute_region_statistics(target_region, target_mask)
            
            # Find compatible source regions
            for source_idx, source_bbox in enumerate(bboxes):
                if target_idx == source_idx:
                    continue
                
                x2, y2, w2, h2 = source_bbox
                
                # Bounds checking
                if x2 < 0 or y2 < 0 or x2+w2 > w or y2+h2 > h:
                    continue
                
                # Extract source region
                source_region = img_pristine[y2:y2+h2, x2:x2+w2]
                source_mask = mask_bool[y2:y2+h2, x2:x2+w2]
                source_mask_inv = mask_bool_inv[y2:y2+h2, x2:x2+w2]
                
                # Validate source region
                if source_mask.sum() < config.min_foreground_pixels:
                    continue
                if source_mask_inv.sum() < config.min_background_pixels:
                    continue
                
                # Compute source statistics
                source_stats = compute_region_statistics(source_region, source_mask)

                # Check compatibility
                if check_compatibility(target_stats, source_stats, (w1, h1), (w2, h2), config):
                    # Apply tampering
                    if config.verbose:
                        print(f"  [TAMPER] Target {target_idx} <- Source {source_idx}")

                    source_resized = cv2.resize(source_region, (w1, h1))
                    img_tampered[y1:y1+h1, x1:x1+w1] = source_resized
                    gt_mask[y1:y1+h1, x1:x1+w1] = 255

                    # Track source information for label generation
                    # Save: source bbox coordinates and source statistics
                    if img_path not in source_info:
                        source_info[img_path] = {
                            'source_bboxes': [source_bbox],  # Store coordinates
                            'source_stats': [source_stats],   # Store statistics
                            'source_idx': source_idx
                        }
                    else:
                        source_info[img_path]['source_bboxes'].append(source_bbox)
                        source_info[img_path]['source_stats'].append(source_stats)
                        source_info[img_path]['source_idx'] = source_idx

                    tampering_count += 1
                    total_tamperings += 1
                    
                    # Check if max tamperings reached
                    if tampering_count >= config.max_tamperings_per_image:
                        # Save tampered image
                        output_img_path = os.path.join(
                            config.tampered_output_dir,
                            f'tampered_{img_cnt:06d}.jpg'
                        )
                        output_mask_path = os.path.join(
                            config.tampered_output_dir,
                            f'mask_{img_cnt:06d}.png'
                        )

                        # Generate label file
                        label_filename = f'tampered_{img_cnt:06d}.json'
                        label_path = os.path.join(config.tampered_output_dir, label_filename)

                        tampering_records = []
                        source_records = []

                        # Get source information for this image
                        source_data = source_info.get(img_path, None)
                        if source_data is None:
                            raise ValueError(f"No source data found for image {img_path}. Tampering failed.")

                        # Convert source_info structure to simpler format
                        source_bboxes_list = source_data.get('source_bboxes', [])
                        source_stats_list = source_data.get('source_stats', [])
                        source_idx = source_data.get('source_idx', 0)

                        for i in range(tampering_count):
                            # Get source bbox and stats for this tampering operation
                            # source_bbox is stored as list: [x, y, width, height]
                            source_bbox = source_bboxes_list[i] if i < len(source_bboxes_list) else [0, 0, 0, 0]
                            source_bbox_stats = source_stats_list[i] if i < len(source_stats_list) else source_stats_list[0] if source_stats_list else target_stats

                            record = {
                                'target': {
                                    'x': int(target_bbox[0]),
                                    'y': int(target_bbox[1]),
                                    'width': int(target_bbox[2]),
                                    'height': int(target_bbox[3])
                                },
                                'source': {
                                    'x': int(source_bbox[0]) if len(source_bbox) > 0 else 0,
                                    'y': int(source_bbox[1]) if len(source_bbox) > 1 else 0,
                                    'width': int(source_bbox[2]) if len(source_bbox) > 2 else 0,
                                    'height': int(source_bbox[3]) if len(source_bbox) > 3 else 0
                                },
                                'statistics': {
                                    'target': {
                                        'fg_mean': float(target_stats[0]),
                                        'fg_std': float(target_stats[1]),
                                        'bg_mean': float(target_stats[2]),
                                        'bg_std': float(target_stats[3])
                                    },
                                    'source': {
                                        'fg_mean': float(source_bbox_stats[0]) if source_bbox_stats and len(source_bbox_stats) > 0 else float(target_stats[0]),
                                        'fg_std': float(source_bbox_stats[1]) if source_bbox_stats and len(source_bbox_stats) > 1 else float(target_stats[1]),
                                        'bg_mean': float(source_bbox_stats[2]) if source_bbox_stats and len(source_bbox_stats) > 2 else float(target_stats[2]),
                                        'bg_std': float(source_bbox_stats[3]) if source_bbox_stats and len(source_bbox_stats) > 3 else float(target_stats[3])
                                    }
                                }
                            }
                            tampering_records.append(record)
                            source_records.append({
                                'index': i,
                                'source_idx': int(source_idx) if isinstance(source_idx, int) else 0,
                                'source_bbox': [int(x) for x in source_bbox] if source_bbox and all(isinstance(x, (int, float)) for x in source_bbox) else [0, 0, 0, 0],
                                'source_stats': [float(x) for x in source_bbox_stats] if source_bbox_stats and all(isinstance(x, (int, float)) for x in source_bbox_stats) else [float(x) for x in target_stats]
                            })

                        label_data = {
                            'image_id': f'tampered_{img_cnt:06d}',
                            'source_image_path': img_path,
                            'source_records': source_records,
                            'tampering_count': tampering_count,
                            'total_tamperings': total_tamperings,
                            'output_image_path': output_img_path,
                            'ground_truth_mask_path': output_mask_path,
                            'label_file_path': label_path,
                            'tampering_operations': tampering_records,
                            'metadata': {
                                'pipeline_run': datetime.now().isoformat(),
                                'tool': 'STG Pipeline',
                                'original_source_bboxes_count': len(source_bboxes_list)
                            },
                            'generation_config': {
                                'max_tamperings_per_image': config.max_tamperings_per_image,
                                'size_tolerance': config.size_tolerance,
                                'foreground_mean_tolerance': config.foreground_mean_tolerance,
                                'foreground_std_tolerance': config.foreground_std_tolerance,
                                'background_mean_tolerance': config.background_mean_tolerance,
                                'background_std_tolerance': config.background_std_tolerance
                            }
                        }

                        # Save label file
                        with open(label_path, 'w') as f:
                            json.dump(label_data, f, indent=2)

                        cv2.imwrite(output_img_path, img_tampered)
                        cv2.imwrite(output_mask_path, gt_mask)

                        tampered_images.append({
                            'image': output_img_path,
                            'mask': output_mask_path,
                            'label': label_path,
                            'source_image': img_path,
                            'tampering_count': tampering_count
                        })
                        
                        if config.verbose:
                            print(f"  [SAVE] Saved image {img_cnt}")
                        
                        img_cnt += 1
                        tampering_count = 0
                        
                        # Reset
                        img_tampered = cv2.imread(img_path)
                        gt_mask = np.zeros((h, w), dtype=np.uint8)
                    
                    break
        
        # Save final image
        if tampering_count > 0:
            output_img_path = os.path.join(
                config.tampered_output_dir, 
                f'tampered_{img_cnt:06d}.jpg'
            )
            output_mask_path = os.path.join(
                config.tampered_output_dir, 
                f'mask_{img_cnt:06d}.png'
            )
            
            cv2.imwrite(output_img_path, img_tampered)
            cv2.imwrite(output_mask_path, gt_mask)
            
            tampered_images.append({
                'image': output_img_path,
                'mask': output_mask_path,
                'source': img_path,
                'tampering_count': tampering_count
            })
            
            img_cnt += 1
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Tampering generation complete!")
    print(f"  Generated {len(tampered_images)} tampered images")
    print(f"  Total tampering operations: {total_tamperings}")
    if len(tampered_images) > 0:
        print(f"  Average tamperings per image: {total_tamperings/len(tampered_images):.2f}")
    print(f"  Output directory: {config.tampered_output_dir}")
    print(f"{'='*70}\n")
    
    return tampered_images

def compute_region_statistics(region_img, region_mask):
    """Compute foreground and background statistics"""
    fg_mask = region_mask
    bg_mask = np.logical_not(region_mask)
    
    fg_mean = region_img[fg_mask].mean().astype(np.float32)
    fg_std = region_img[fg_mask].std().astype(np.float32)
    bg_mean = region_img[bg_mask].mean().astype(np.float32)
    bg_std = region_img[bg_mask].std().astype(np.float32)
    
    return fg_mean, fg_std, bg_mean, bg_std

def check_compatibility(target_stats, source_stats, target_size, source_size, config):
    """Check if source and target are compatible"""
    tgt_fg_mean, tgt_fg_std, tgt_bg_mean, tgt_bg_std = target_stats
    src_fg_mean, src_fg_std, src_bg_mean, src_bg_std = source_stats
    tgt_w, tgt_h = target_size
    src_w, src_h = source_size
    
    # Size constraints
    w_min = src_w * (1 - config.size_tolerance)
    w_max = src_w * (1 + config.size_tolerance)
    h_min = src_h * (1 - config.size_tolerance)
    h_max = src_h * (1 + config.size_tolerance)
    
    if not (w_min <= tgt_w <= w_max and h_min <= tgt_h <= h_max):
        return False
    
    # Statistical constraints
    if abs(tgt_fg_mean - src_fg_mean) > config.foreground_mean_tolerance:
        return False
    if abs(tgt_fg_std - src_fg_std) > config.foreground_std_tolerance:
        return False
    if abs(tgt_bg_mean - src_bg_mean) > config.background_mean_tolerance:
        return False
    if abs(tgt_bg_std - src_bg_std) > config.background_std_tolerance:
        return False
    
    return True

# ============================================================================
# STEP 5: POST-PROCESSING
# ============================================================================

def apply_postprocessing(tampered_images, config):
    """
    Apply post-processing to tampered images.
    
    Args:
        tampered_images: List of tampered image info dicts
        config: PipelineConfig object
    """
    print("="*70)
    print("STEP 5: Applying post-processing")
    print("="*70)
    
    if not (config.enable_poisson_blending or config.enable_noise_matching or 
            config.enable_jpeg_compression):
        print("No post-processing enabled. Skipping.")
        print()
        return
    
    for item in tqdm(tampered_images, desc="Post-processing"):
        img_path = item['image']
        mask_path = item['mask']
        
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)
        
        # Apply JPEG compression
        if config.enable_jpeg_compression:
            img = apply_jpeg_compression(img, config.jpeg_quality)
        
        # Save post-processed image
        output_filename = os.path.basename(img_path).replace('tampered', 'postprocessed')
        output_path = os.path.join(config.postprocessed_output_dir, output_filename)
        cv2.imwrite(output_path, img)
    
    print(f"Post-processed images saved to: {config.postprocessed_output_dir}")
    print()

def apply_jpeg_compression(img, quality=85):
    """Apply JPEG compression"""
    import tempfile
    import os
    from PIL import Image
    
    # Use delete=False to avoid permission issues on Windows
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    try:
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(
            tmp.name, 'JPEG', quality=quality
        )
        img_compressed = Image.open(tmp.name)
        compressed_img = cv2.cvtColor(np.array(img_compressed), cv2.COLOR_RGB2BGR)
    finally:
        tmp.close()
        # Try to delete the temp file after use
        try:
            os.unlink(tmp.name)
        except:
            pass
    
    return compressed_img

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(config):
    """Run the complete STG pipeline"""
    print("\n" + "="*70)
    print("SELECTIVE TAMPERING GENERATION (STG) - FULL PIPELINE")
    print("="*70)
    print(f"Input directory: {config.input_dir}")
    print(f"Output directory: {config.output_dir}")
    print("="*70 + "\n")
    
    # Initialize directories
    config.init_directories()
    
    # Step 1: Load images
    image_paths = load_images(config)
    if len(image_paths) == 0:
        print("ERROR: No images found. Exiting.")
        return
    
    # Step 2: Run OCR
    ocr_results = run_craft_ocr(image_paths, config)
    
    # Step 3: Generate binary masks
    mask_paths = generate_binary_masks(image_paths, config)
    
    # Step 4: Generate tampering
    tampered_images = generate_tampering(ocr_results, mask_paths, config)
    
    # Step 5: Post-processing
    apply_postprocessing(tampered_images, config)
    
    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"Input images: {len(image_paths)}")
    print(f"OCR results: {len(ocr_results)}")
    print(f"Binary masks: {len(mask_paths)}")
    print(f"Tampered images: {len(tampered_images)}")
    print(f"\nAll outputs saved to: {config.output_dir}")
    print("="*70 + "\n")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='STG Pipeline Testing Script')
    parser.add_argument('--input_dir', type=str, default='test_images',
                       help='Input directory containing images')
    parser.add_argument('--output_dir', type=str, default='pipeline_output',
                       help='Output directory for results')
    parser.add_argument('--max_tamperings', type=int, default=10,
                       help='Maximum tamperings per image')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA for CRAFT')
    parser.add_argument('--jpeg_quality', type=int, default=85,
                       help='JPEG compression quality (1-100)')
    parser.add_argument('--quiet', action='store_true',
                       help='Disable verbose output')
    
    args = parser.parse_args()
    
    # Create configuration
    config = PipelineConfig()
    config.input_dir = args.input_dir
    config.output_dir = args.output_dir
    config.max_tamperings_per_image = args.max_tamperings
    config.craft_use_cuda = not args.no_cuda
    config.jpeg_quality = args.jpeg_quality
    config.verbose = not args.quiet
    
    # Run pipeline
    try:
        run_pipeline(config)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
