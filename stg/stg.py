"""
Selective Tampering Generation (STG) - Toy Implementation

This script generates synthetic document tampering by copying text regions between
locations in the same document, creating realistic forgery examples for training
document tampering detection models.

Algorithm Overview:
1. Load OCR bounding boxes for all text regions in images
2. Load binary masks separating text (foreground) from background
3. For each text region (target), find compatible source regions based on:
   - Size similarity (±10% width/height tolerance)
   - Statistical similarity (foreground/background color distribution)
4. Copy-paste compatible source regions to target locations
5. Generate ground truth masks marking tampered regions

Key Constraints for Realistic Tampering:
- Source and target must have similar dimensions
- Foreground (text) color statistics must match (mean ±20, std ±4)
- Background color statistics must match (mean ±20, std ±4)

Note: This is a simplified toy version (5% of original code).
Missing components: OCR integration, post-processing, filtering, acceleration.
"""

import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm

# ============================================================================
# PREPARATION REQUIREMENTS
# ============================================================================

# REQUIREMENT 1: OCR pickle file (ocr.pk)
# Format: {'img_path1': [[x1,y1,w1,h1], [x2,y2,w2,h2], ...], 
#          'img_path2': [[x1,y1,w1,h1], ...]}
# - Keys: Image file paths (e.g., 'imgs/0.jpg')
# - Values: List of bounding boxes [x, y, width, height] for each text region
# - Example OCR tools: PaddleOCR v3, OCR.space API, Tesseract

### PSEUDO-CODE FOR MISSING OCR GENERATION:
"""
def generate_ocr_pickle(image_dir):
    '''
    Generate OCR bounding boxes for all images in a directory.
    This function is NOT implemented in the toy version.
    '''
    ocr_results = {}
    
    for image_path in glob(os.path.join(image_dir, '*.jpg')):
        # Initialize OCR model (e.g., PaddleOCR)
        # ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
        
        # Run OCR detection
        # result = ocr_model.ocr(image_path, det=True, rec=False)
        
        # Extract bounding boxes
        boxes = []
        # for line in result:
        #     for word_info in line:
        #         bbox = word_info[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        #         x_coords = [pt[0] for pt in bbox]
        #         y_coords = [pt[1] for pt in bbox]
        #         x, y = min(x_coords), min(y_coords)
        #         w = max(x_coords) - x
        #         h = max(y_coords) - y
        #         boxes.append([x, y, w, h])
        
        ocr_results[image_path] = boxes
    
    # Save to pickle
    with open('ocr.pk', 'wb') as f:
        pickle.dump(ocr_results, f)
    
    return ocr_results
"""

# REQUIREMENT 2: Binary masks for text/background separation
# Format: Binary images where text=white (255), background=black (0)
# - Image: 'imgs/0.jpg' -> Mask: 'masks/0.png'
# - Example binarization method: Sauvola adaptive thresholding

### OCR Resources:
# - OCR.space API: http://ocr.space/
# - OCRSpace Python wrapper: https://github.com/ErikBoesen/ocrspace
# - PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR

### Image Binarization Example (using doxapy):
"""
import doxapy  # pip install doxapy
import numpy as np
import cv2

def generate_binary_mask(image_path, mask_path):
    '''
    Generate binary mask separating text from background.
    Uses Sauvola adaptive binarization.
    '''
    # Initialize binarization model with Sauvola algorithm
    model = doxapy.Binarization(doxapy.Binarization.Algorithms.SAUVOLA)
    
    # Load image
    img = cv2.imread(image_path)
    
    # Initialize model with image
    model.initialize(img)
    
    # Create output mask
    msk = np.zeros_like(img)
    
    # Apply binarization (window=75, k=0.2 are common Sauvola parameters)
    model.to_binary(msk, {"window": 75, "k": 0.2})
    
    # Save binary mask
    cv2.imwrite(mask_path, msk)
"""

# ============================================================================
# MAIN SCRIPT
# ============================================================================

# ============================================================================
# LOAD OCR DATA
# ============================================================================

# Load OCR bounding boxes from pickle file
# fpk structure: {'imgs/0.jpg': [[x1,y1,w1,h1], ...], 'imgs/1.jpg': [[x1,y1,w1,h1], ...]}
with open('ocr.pk', 'rb') as f:
    fpk = pickle.load(f)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

img_cnt = 0        # Counter for output tampered images
max_cnt = 10       # Maximum number of tampering operations per output image
                   # After max_cnt tamperings, save the image and start fresh

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def getdir(path):
    """
    Create directory if it doesn't exist.
    
    Args:
        path (str): Directory path to create
    """
    if not os.path.exists(path):
        os.makedirs(path)

# Create output directories for tampered images and ground truth masks
getdir('tamp_imgs')   # Stores generated tampered images
getdir('tamp_masks')  # Stores ground truth masks (255=tampered, 0=authentic)

# ============================================================================
# MAIN TAMPERING GENERATION LOOP
# ============================================================================

# Iterate through each image and its OCR bounding boxes
for k, box in tqdm(fpk.items()):
    # ========================================================================
    # STEP 1: Load Image and Mask
    # ========================================================================
    
    # Load original image (will be modified with tampering)
    img1 = cv2.imread(k)
    h, w = img1.shape[:2]  # Get image dimensions
    
    # Load pristine copy (source for copying regions)
    img2 = cv2.imread(k)
    
    # Load binary mask: True=text (foreground), False=background
    # Converts mask to boolean: pixels > 127 become True (text)
    mask = (cv2.imread(k.replace('imgs', 'masks')[:-4] + '.png', 0) > 127)
    not_mask = np.logical_not(mask)  # Inverted mask for background
    
    # Initialize ground truth mask (all zeros = authentic)
    gt = np.zeros((h, w), dtype=np.uint8)
    
    # Counter for tampering operations on current image
    cnt = 0
    
    # ========================================================================
    # STEP 2: Iterate Through Target Regions
    # ========================================================================
    
    # bi1: box index 1 (target region index)
    # b1: bounding box 1 [x, y, width, height] - TARGET region to be replaced
    for bi1, b1 in enumerate(box):
        
        # --------------------------------------------------------------------
        # Extract target region and compute its statistics
        # --------------------------------------------------------------------
        
        # Extract target region pixels from image
        # Slice: rows from y to y+h, columns from x to x+w
        tgt = img1[b1[1]:b1[1]+b1[3], b1[0]:b1[0]+b1[2]]
        
        # Extract corresponding mask regions
        msk = mask[b1[1]:b1[1]+b1[3], b1[0]:b1[0]+b1[2]]      # Text pixels
        not_msk = not_mask[b1[1]:b1[1]+b1[3], b1[0]:b1[0]+b1[2]]  # Background pixels
        
        # Compute color statistics for TARGET region
        # Foreground (text) statistics
        tgtfm = tgt[msk].mean().astype(np.float32)   # Mean intensity of text
        tgtfs = tgt[msk].std().astype(np.float32)    # Std deviation of text
        
        # Background statistics
        tgtbm = tgt[not_msk].mean().astype(np.float32)  # Mean intensity of background
        tgtbs = tgt[not_msk].std().astype(np.float32)   # Std deviation of background
        
        # ====================================================================
        # STEP 3: Find Compatible Source Regions
        # ====================================================================
        
        # bi2: box index 2 (source region index)
        # b2: bounding box 2 [x, y, width, height] - SOURCE region to copy from
        for bi2, b2 in enumerate(box):
            
            # Skip if comparing same region
            if bi1 != bi2:
                
                # ------------------------------------------------------------
                # Define compatibility constraints
                # ------------------------------------------------------------
                
                # SIZE CONSTRAINTS: ±10% tolerance for width and height
                # Ensures visual similarity in region dimensions
                w11 = (0.9 * b2[2])  # Minimum acceptable width (90% of source)
                w12 = (1.1 * b2[2])  # Maximum acceptable width (110% of source)
                h11 = (0.9 * b2[3])  # Minimum acceptable height (90% of source)
                h12 = (1.1 * b2[3])  # Maximum acceptable height (110% of source)
                
                # CRITICAL FIX: Extract SOURCE region statistics (from b2, not b1)
                # This computes the appearance statistics of the SOURCE region we want to copy
                src = img2[b2[1]:b2[1]+b2[3], b2[0]:b2[0]+b2[2]]
                src_msk = mask[b2[1]:b2[1]+b2[3], b2[0]:b2[0]+b2[2]]
                src_not_msk = not_mask[b2[1]:b2[1]+b2[3], b2[0]:b2[0]+b2[2]]
                
                # Check if regions have sufficient foreground/background pixels
                if src_msk.sum() < 10 or src_not_msk.sum() < 10:
                    continue  # Skip if insufficient pixels for statistics
                if msk.sum() < 10 or not_msk.sum() < 10:
                    continue
                
                # Compute SOURCE region statistics
                tgtfm2 = src[src_msk].mean().astype(np.float32)       # Source foreground mean
                tgtfs2 = src[src_msk].std().astype(np.float32)        # Source foreground std
                tgtbm2 = src[src_not_msk].mean().astype(np.float32)   # Source background mean
                tgtbs2 = src[src_not_msk].std().astype(np.float32)    # Source background std
                
                # STATISTICAL CONSTRAINTS: Define acceptable ranges
                # Foreground (text) mean: ±20 intensity units
                tgtfm21 = tgtfm2 - 20
                tgtfm22 = tgtfm2 + 20
                
                # Foreground (text) std: ±4 intensity units
                tgtfs21 = tgtfs2 - 4
                tgtfs22 = tgtfs2 + 4
                
                # Background mean: ±20 intensity units
                tgtbm21 = tgtbm2 - 20
                tgtbm22 = tgtbm2 + 20
                
                # Background std: ±4 intensity units
                tgtbs21 = tgtbs2 - 4
                tgtbs22 = tgtbs2 + 4
                
                # ------------------------------------------------------------
                # Check all compatibility constraints
                # ------------------------------------------------------------
                
                # Combined constraint check:
                # 1. Size compatibility: target width/height within ±10% of source
                # 2. Foreground mean compatibility: within ±20 range
                # 3. Background mean compatibility: within ±20 range
                # 4. Foreground std compatibility: within ±4 range
                # 5. Background std compatibility: within ±4 range
                if ((w11 <= b1[2] <= w12) and           # Width check
                    (h11 <= b1[3] <= h12) and           # Height check
                    (tgtfm21 <= tgtfm <= tgtfm22) and   # Foreground mean check
                    (tgtbm21 <= tgtbm <= tgtbm22) and   # Background mean check
                    (tgtfs21 <= tgtfs <= tgtfs22) and   # Foreground std check
                    (tgtbs21 <= tgtbs <= tgtbs22)):     # Background std check
                    
                    # --------------------------------------------------------
                    # APPLY TAMPERING: Copy-paste operation
                    # --------------------------------------------------------
                    
                    # Debug print: show shape of target and source regions
                    print(img1[b1[1]:b1[1]+b1[3], b1[0]:b1[0]+b1[2]].shape, 
                          img2[b2[1]:b2[1]+b2[3], b2[0]:b2[0]+b2[2]].shape)
                    
                    # Copy source region (b2) from pristine image (img2)
                    # Resize to match target dimensions
                    # Paste into target location (b1) in modified image (img1)
                    img1[b1[1]:b1[1]+b1[3], b1[0]:b1[0]+b1[2]] = cv2.resize(
                        img2[b2[1]:b2[1]+b2[3], b2[0]:b2[0]+b2[2]], 
                        (int(b1[2]), int(b1[3]))
                    )
                    
                    # Mark target region as tampered in ground truth mask
                    gt[b1[1]:b1[1]+b1[3], b1[0]:b1[0]+b1[2]] = 255
                    
                    # Increment tampering counter
                    cnt = (cnt + 1)
                    
                    # --------------------------------------------------------
                    # Save image if max tampering operations reached
                    # --------------------------------------------------------
                    if cnt > max_cnt:
                        # Save tampered image
                        cv2.imwrite('tamp_imgs/%d.jpg' % img_cnt, img1)
                        # Save ground truth mask
                        cv2.imwrite('tamp_masks/%d.png' % img_cnt, gt)
                        
                        # Reset for next image
                        cnt = 0
                        img_cnt = (img_cnt + 1)
                        
                        # Reload pristine image to start fresh
                        img1 = cv2.imread(k)
                        gt = np.zeros((h, w), dtype=np.uint8)
    
    # ========================================================================
    # STEP 4: Save final image (with remaining tamperings)
    # ========================================================================
    
    # Save the last version of the tampered image (may have < max_cnt tamperings)
    cv2.imwrite('tamp_imgs/%d.jpg' % img_cnt, img1)
    cv2.imwrite('tamp_masks/%d.png' % img_cnt, gt)
    
    # Reset counters for next source image
    cnt = 0
    img_cnt = (img_cnt + 1)

# ============================================================================
# END OF TAMPERING GENERATION
# ============================================================================

### PSEUDO-CODE FOR MISSING POST-PROCESSING (not in toy version):
"""
def post_process_tampering(tampered_img, ground_truth_mask):
    '''
    Apply post-processing to make tampering more realistic.
    These functions are NOT implemented in the toy version.
    '''
    
    # 1. BLENDING: Smooth boundaries between copied region and surroundings
    #    - Apply Poisson blending or alpha blending
    #    - Reduces visible seams at region boundaries
    # tampered_img = poisson_blend(tampered_img, ground_truth_mask)
    
    # 2. NOISE MATCHING: Match noise characteristics
    #    - Analyze noise in surrounding areas
    #    - Add similar noise to copied region
    # tampered_img = match_noise(tampered_img, ground_truth_mask)
    
    # 3. JPEG COMPRESSION: Apply realistic compression artifacts
    #    - Compress and decompress to introduce JPEG artifacts
    #    - Makes detection harder by masking copy-paste traces
    # tampered_img = jpeg_compress(tampered_img, quality=85)
    
    # 4. FILTERING: Remove low-quality tampering examples
    #    - Check if tampering is detectable by simple methods
    #    - Remove obvious forgeries that don't challenge the detector
    # if is_too_obvious(tampered_img, ground_truth_mask):
    #     return None
    
    return tampered_img

def accelerate_generation(fpk, num_processes=8):
    '''
    Parallelize tampering generation across multiple processes.
    NOT implemented in toy version.
    '''
    # Use multiprocessing to process multiple images simultaneously
    # from multiprocessing import Pool
    # with Pool(num_processes) as p:
    #     p.map(generate_tampering_for_image, fpk.items())
    pass
"""

print(f"\nTampering generation complete!")
print(f"Generated {img_cnt} tampered images in 'tamp_imgs/' directory")
print(f"Ground truth masks saved in 'tamp_masks/' directory")
