"""
Image Processing Core Module
Handles image enhancement operations using OpenCV
"""

from qtpy.QtGui import QPixmap, QImage
from qtpy.QtCore import Qt
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from io import BytesIO


class ImageProcessor:
    """Handles image processing and enhancement operations using OpenCV."""
    
    def __init__(self):
        self.use_opencv = True  # Flag to switch between OpenCV and PIL
        self.original_image = None  # Store original OpenCV image
    
    def enhance_image(self, pixmap, brightness=0, contrast=0, saturation=0):
        """
        Enhance image with brightness, contrast, and saturation adjustments using OpenCV.
        
        Args:
            pixmap (QPixmap): Original image pixmap
            brightness (int): Brightness adjustment (-100 to 100)
            contrast (int): Contrast adjustment (-100 to 100)
            saturation (int): Saturation adjustment (-100 to 100)
            
        Returns:
            QPixmap: Enhanced image pixmap
        """
        try:
            if self.use_opencv:
                return self.enhance_with_opencv(pixmap, brightness, contrast, saturation)
            else:
                return self.enhance_with_pil(pixmap, brightness, contrast, saturation)
            
        except Exception as e:
            print(f"Error enhancing image: {e}")
            return pixmap  # Return original on error
    
    def qpixmap_to_pil(self, pixmap):
        """Convert QPixmap to PIL Image."""
        # Convert QPixmap to QImage
        qimage = pixmap.toImage()
        
        # Convert QImage to bytes
        buffer = BytesIO()
        qimage.save(buffer, "PNG")
        buffer.seek(0)
        
        # Create PIL Image from bytes
        return Image.open(buffer).convert("RGB")
    
    def pil_to_qpixmap(self, pil_image):
        """Convert PIL Image to QPixmap."""
        # Convert PIL Image to bytes
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Create QPixmap from bytes
        pixmap = QPixmap()
        pixmap.loadFromData(buffer.getvalue())
        
        return pixmap
    
    def enhance_with_opencv(self, pixmap, brightness=0, contrast=0, saturation=0):
        """Enhance image using OpenCV for better performance and quality."""
        try:
            # Convert QPixmap to OpenCV image
            cv_image = self.qpixmap_to_cv2(pixmap)
            
            # Apply brightness and contrast
            if brightness != 0 or contrast != 0:
                # Convert brightness (-100,100) to OpenCV range
                alpha = 1.0 + (contrast / 100.0)  # Contrast (1.0 = no change)
                beta = brightness * 2.55  # Brightness (0 = no change)
                
                cv_image = cv2.convertScaleAbs(cv_image, alpha=alpha, beta=beta)
            
            # Apply saturation adjustment
            if saturation != 0:
                # Convert BGR to HSV
                hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
                hsv = hsv.astype(np.float64)
                
                # Adjust saturation
                saturation_factor = 1.0 + (saturation / 100.0)
                hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor
                
                # Clip values to valid range and convert back
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
                hsv = hsv.astype(np.uint8)
                cv_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Convert back to QPixmap
            return self.cv2_to_qpixmap(cv_image)
            
        except Exception as e:
            print(f"OpenCV enhancement failed: {e}, falling back to PIL")
            # Fallback to PIL enhancement
            return self.enhance_with_pil(pixmap, brightness, contrast, saturation)
    
    def enhance_with_pil(self, pixmap, brightness=0, contrast=0, saturation=0):
        """Fallback PIL-based enhancement method."""
        # Convert QPixmap to PIL Image
        pil_image = self.qpixmap_to_pil(pixmap)
        
        # Apply brightness adjustment
        if brightness != 0:
            enhancer = ImageEnhance.Brightness(pil_image)
            factor = 1.0 + (brightness / 100.0)
            pil_image = enhancer.enhance(factor)
        
        # Apply contrast adjustment
        if contrast != 0:
            enhancer = ImageEnhance.Contrast(pil_image)
            factor = 1.0 + (contrast / 100.0)
            pil_image = enhancer.enhance(factor)
        
        # Apply saturation adjustment
        if saturation != 0:
            enhancer = ImageEnhance.Color(pil_image)
            factor = 1.0 + (saturation / 100.0)
            pil_image = enhancer.enhance(factor)
        
        # Convert back to QPixmap
        return self.pil_to_qpixmap(pil_image)
    
    def qpixmap_to_cv2(self, pixmap):
        """Convert QPixmap to OpenCV image (BGR format)."""
        try:
            # Method 1: Direct conversion using ARGB32 format
            qimage = pixmap.toImage().convertToFormat(QImage.Format.Format_ARGB32)
            width = qimage.width()
            height = qimage.height()
            
            # Get the raw data
            ptr = qimage.bits()
            if ptr is None:
                raise ValueError("Could not get image data")
            
            # Convert to numpy array
            arr = np.frombuffer(ptr, dtype=np.uint8)
            expected_size = height * width * 4  # 4 bytes per pixel for ARGB32
            
            if len(arr) != expected_size:
                raise ValueError(f"Size mismatch: got {len(arr)}, expected {expected_size}")
                
            arr = arr.reshape((height, width, 4))
            
            # Extract BGR channels (ARGB format)
            cv_image = np.zeros((height, width, 3), dtype=np.uint8)
            cv_image[:, :, 0] = arr[:, :, 0]  # Blue
            cv_image[:, :, 1] = arr[:, :, 1]  # Green  
            cv_image[:, :, 2] = arr[:, :, 2]  # Red
            
            return cv_image
            
        except Exception as e:
            # Fallback: Use PIL as intermediate
            try:
                pil_image = self.qpixmap_to_pil(pixmap)
                # Convert PIL to numpy array
                rgb_array = np.array(pil_image)
                # Convert RGB to BGR
                cv_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                return cv_image
            except Exception as e2:
                print(f"Both conversion methods failed: {e}, {e2}")
                raise e2
    
    def cv2_to_qpixmap(self, cv_image):
        """Convert OpenCV image (BGR) to QPixmap."""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        
        # Create QImage
        qimage = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Convert to QPixmap
        return QPixmap.fromImage(qimage)
    
    def apply_fastNlMeansDenoising(self, pixmap, h=10, h_color=10, template_window_size=7, search_window_size=21):
        """
        Apply Non-local Means Denoising for noise reduction.
        
        Args:
            pixmap (QPixmap): Original image
            h (float): Filter strength for luminance component (higher = more denoising, less detail)
            h_color (float): Filter strength for color components
            template_window_size (int): Size of template patch (should be odd, recommended 7)
            search_window_size (int): Size of search window (should be odd, recommended 21)
            
        Returns:
            QPixmap: Denoised image
        """
        try:
            cv_image = self.qpixmap_to_cv2(pixmap)
            
            # Ensure window sizes are odd and valid
            if template_window_size % 2 == 0:
                template_window_size += 1
            if search_window_size % 2 == 0:
                search_window_size += 1
            
            template_window_size = max(3, template_window_size)
            search_window_size = max(template_window_size + 2, search_window_size)
            
            # Apply Non-local Means Denoising
            denoised_image = cv2.fastNlMeansDenoisingColored(
                cv_image, None, h, h_color, template_window_size, search_window_size
            )
            
            return self.cv2_to_qpixmap(denoised_image)
            
        except Exception as e:
            print(f"Error applying noise reduction: {e}")
            return pixmap
    
    def apply_gaussian_blur(self, pixmap, kernel_size=15, sigma_x=0, sigma_y=0):
        """
        Apply Gaussian blur with detailed parameter control.
        
        Args:
            pixmap (QPixmap): Original image
            kernel_size (int): Size of the Gaussian kernel (must be odd)
            sigma_x (float): Standard deviation in X direction (0 = auto-calculate)
            sigma_y (float): Standard deviation in Y direction (0 = use sigma_x)
            
        Returns:
            QPixmap: Blurred image
        """
        try:
            cv_image = self.qpixmap_to_cv2(pixmap)
            
            # Ensure kernel size is odd and positive
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel_size = max(3, kernel_size)
            
            # Apply Gaussian blur with specified parameters
            cv_image = cv2.GaussianBlur(cv_image, (kernel_size, kernel_size), sigma_x, sigmaY=sigma_y)
            
            return self.cv2_to_qpixmap(cv_image)
            
        except Exception as e:
            print(f"Error applying Gaussian blur: {e}")
            return pixmap
    
    def apply_filter(self, pixmap, filter_type):
        """
        Apply various OpenCV filters to the image.
        
        Args:
            pixmap (QPixmap): Original image
            filter_type (str): Type of filter ('sharpen', 'edge', etc.)
            
        Returns:
            QPixmap: Filtered image
        """
        try:
            cv_image = self.qpixmap_to_cv2(pixmap)
            
            if filter_type == 'sharpen':
                kernel = np.array([[-1,-1,-1],
                                 [-1, 9,-1],
                                 [-1,-1,-1]])
                cv_image = cv2.filter2D(cv_image, -1, kernel)
            elif filter_type == 'edge':
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                cv_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            elif filter_type == 'emboss':
                kernel = np.array([[-2, -1, 0],
                                 [-1,  1, 1],
                                 [ 0,  1, 2]])
                cv_image = cv2.filter2D(cv_image, -1, kernel)
            else:
                print(f"Unknown filter type: {filter_type}")
                return pixmap
            
            return self.cv2_to_qpixmap(cv_image)
            
        except Exception as e:
            print(f"Error applying {filter_type} filter: {e}")
            return pixmap  # Return original image on error
    
    def apply_histogram_equalization(self, pixmap):
        """Apply histogram equalization for better contrast."""
        try:
            cv_image = self.qpixmap_to_cv2(pixmap)
            
            # Convert to YUV color space
            yuv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2YUV)
            
            # Apply histogram equalization to Y channel
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            
            # Convert back to BGR
            cv_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            
            return self.cv2_to_qpixmap(cv_image)
            
        except Exception as e:
            print(f"Error applying histogram equalization: {e}")
            return pixmap