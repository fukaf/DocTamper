"""
Quick test script to verify pipeline dependencies and setup.

Run this before using pipeline_test.py to check if everything is installed correctly.
"""

import sys

def check_imports():
    """Check if all required packages are installed"""
    print("="*70)
    print("CHECKING DEPENDENCIES")
    print("="*70)
    
    results = {}
    
    # Essential dependencies
    essentials = [
        ('cv2', 'opencv-python', 'pip install opencv-python'),
        ('numpy', 'numpy', 'pip install numpy'),
        ('PIL', 'pillow', 'pip install pillow'),
        ('tqdm', 'tqdm', 'pip install tqdm'),
    ]
    
    # Optional dependencies
    optionals = [
        ('craft_text_detector', 'craft-text-detector', 'pip install craft-text-detector'),
        ('doxapy', 'doxapy', 'pip install doxapy'),
    ]
    
    print("\nEssential Dependencies:")
    print("-" * 70)
    for module, package, install_cmd in essentials:
        try:
            __import__(module)
            print(f"✓ {package:30s} - INSTALLED")
            results[package] = True
        except ImportError:
            print(f"✗ {package:30s} - MISSING")
            print(f"  Install with: {install_cmd}")
            results[package] = False
    
    print("\nOptional Dependencies (fallback available if missing):")
    print("-" * 70)
    for module, package, install_cmd in optionals:
        try:
            __import__(module)
            print(f"✓ {package:30s} - INSTALLED")
            results[package] = True
        except ImportError:
            print(f"○ {package:30s} - NOT INSTALLED (will use fallback)")
            print(f"  Install with: {install_cmd}")
            results[package] = False
    
    return results

def check_cuda():
    """Check CUDA availability"""
    print("\n" + "="*70)
    print("CHECKING CUDA/GPU SUPPORT")
    print("="*70)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("○ CUDA not available (will use CPU)")
            return False
    except ImportError:
        print("○ PyTorch not installed (needed for CRAFT GPU support)")
        print("  Install with: pip install torch torchvision")
        return False

def test_opencv():
    """Test OpenCV functionality"""
    print("\n" + "="*70)
    print("TESTING OPENCV")
    print("="*70)
    
    try:
        import cv2
        import numpy as np
        
        # Create test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[25:75, 25:75] = [255, 255, 255]
        
        # Test basic operations
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"✓ OpenCV working correctly")
        print(f"  Version: {cv2.__version__}")
        print(f"  Test: Created image, converted to gray, found {len(contours)} contour(s)")
        return True
    except Exception as e:
        print(f"✗ OpenCV test failed: {e}")
        return False

def test_doxapy():
    """Test doxapy functionality"""
    print("\n" + "="*70)
    print("TESTING DOXAPY")
    print("="*70)
    
    try:
        import doxapy
        import numpy as np
        
        # Create test image
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test Sauvola
        model = doxapy.Binarization(doxapy.Binarization.Algorithms.SAUVOLA)
        model.initialize(test_img)
        mask = np.zeros_like(test_img)
        model.to_binary(mask, {"window": 75, "k": 0.2})
        
        print(f"✓ doxapy working correctly")
        print(f"  Test: Applied Sauvola binarization to 100x100 image")
        return True
    except ImportError:
        print(f"○ doxapy not installed (will use Otsu fallback)")
        return False
    except Exception as e:
        print(f"✗ doxapy test failed: {e}")
        return False

def test_craft():
    """Test CRAFT text detector"""
    print("\n" + "="*70)
    print("TESTING CRAFT")
    print("="*70)
    
    try:
        import craft_text_detector
        print(f"✓ craft-text-detector installed")
        print(f"  Note: First run will download CRAFT model (~80MB)")
        return True
    except ImportError:
        print(f"○ craft-text-detector not installed (will use contour fallback)")
        print(f"  Install with: pip install craft-text-detector")
        return False

def check_directories():
    """Check if test directories exist"""
    print("\n" + "="*70)
    print("CHECKING DIRECTORIES")
    print("="*70)
    
    import os
    
    # Check if test_images exists
    if os.path.exists('test_images'):
        num_images = len([f for f in os.listdir('test_images') 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        print(f"✓ test_images/ directory exists")
        print(f"  Found {num_images} image(s)")
        if num_images == 0:
            print(f"  ⚠ No images found. Add some images to test the pipeline.")
    else:
        print(f"○ test_images/ directory not found")
        print(f"  Create it and add test images:")
        print(f"    mkdir test_images")
        print(f"    # Copy some document images to test_images/")
    
    # Check if output directory exists
    if os.path.exists('pipeline_output'):
        print(f"✓ pipeline_output/ directory exists")
    else:
        print(f"○ pipeline_output/ directory will be created on first run")

def print_summary(results):
    """Print summary of checks"""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    essentials_ok = all([
        results.get('opencv-python', False),
        results.get('numpy', False),
        results.get('pillow', False),
        results.get('tqdm', False),
    ])
    
    if essentials_ok:
        print("✓ All essential dependencies installed")
        print("  You can run the pipeline!")
    else:
        print("✗ Some essential dependencies missing")
        print("  Please install missing packages before running pipeline")
    
    print("\nOptional components:")
    if results.get('craft-text-detector', False):
        print("  ✓ CRAFT OCR available (recommended)")
    else:
        print("  ○ CRAFT OCR not available (will use fallback)")
    
    if results.get('doxapy', False):
        print("  ✓ doxapy available (recommended)")
    else:
        print("  ○ doxapy not available (will use Otsu fallback)")
    
    print("\nNext steps:")
    if essentials_ok:
        print("  1. Create test_images/ directory")
        print("  2. Add some document images to test_images/")
        print("  3. Run: python pipeline_test.py")
    else:
        print("  1. Install missing essential packages")
        print("  2. Run this test script again")
        print("  3. Follow next steps above")
    
    print()

def main():
    """Main test function"""
    print("\n")
    print("="*70)
    print("STG PIPELINE - DEPENDENCY CHECK")
    print("="*70)
    print()
    
    # Run checks
    results = check_imports()
    check_cuda()
    test_opencv()
    test_doxapy()
    test_craft()
    check_directories()
    
    # Print summary
    print_summary(results)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
