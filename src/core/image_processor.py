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
import os
import logging
from typing import Dict, List, Tuple, Optional, Union


class ImageAnalyzer:
    """
    Analyzes images and identifies quality issues (noise, lighting, backlight, blur).
    Does not modify images; returns analysis and recommendations.
    """

    def __init__(
        self,
        noise_threshold: float = 200.0,
        brightness_min: int = 60,
        brightness_max: int = 200,
        contrast_threshold: float = 25.0,
    ):
        self.noise_threshold = noise_threshold
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.contrast_threshold = contrast_threshold

        logging.basicConfig(level=logging.INFO)
        self._logger = logging.getLogger(__name__)

    def detect_noise(self, image: np.ndarray) -> Dict:
        """Detect noise levels in image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_intensity = float(np.mean(gray))
        std_intensity = float(np.std(gray))

        try:
            denoised = cv2.bilateralFilter(gray, 5, 20, 20)
            mse = float(np.mean((gray.astype(float) - denoised.astype(float)) ** 2))
            if mse <= 1e-8:
                psnr = 100.0
            else:
                psnr = float(20.0 * np.log10(255.0 / np.sqrt(mse)))
        except Exception:
            psnr = 0.0

        noise_score = laplacian_var / 1000.0
        if psnr < 20:
            noise_level = "high"
            severity = "severe"
        elif psnr < 30:
            noise_level = "medium" if laplacian_var < 25 else "low"
            severity = "moderate" if laplacian_var < 20 else "mild"
        else:
            noise_level = "low"
            severity = "mild"

        is_noisy = laplacian_var > self.noise_threshold or psnr < 25

        return {
            "laplacian_variance": laplacian_var,
            "mean_intensity": mean_intensity,
            "std_intensity": std_intensity,
            "noise_std": std_intensity,
            "psnr": psnr,
            "noise_score": noise_score,
            "is_noisy": is_noisy,
            "noise_level": noise_level,
            "severity": severity,
        }

    def detect_lighting(self, image: np.ndarray) -> Dict:
        """Detect lighting issues (brightness, contrast, exposure)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        mean_brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        min_val = float(np.min(gray))
        max_val = float(np.max(gray))
        dynamic_range = max_val - min_val

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        overexposed_pixels = float(np.sum(hist[240:]) / gray.size)
        underexposed_pixels = float(np.sum(hist[:15]) / gray.size)

        return {
            "mean_brightness": mean_brightness,
            "contrast": contrast,
            "dynamic_range": dynamic_range,
            "overexposed_ratio": overexposed_pixels,
            "underexposed_ratio": underexposed_pixels,
            "is_too_bright": mean_brightness > self.brightness_max,
            "is_too_dark": mean_brightness < self.brightness_min,
            "is_low_contrast": contrast < self.contrast_threshold,
            "is_low_light": mean_brightness < self.brightness_min,
            "severity": self._assess_lighting_severity(mean_brightness, contrast),
        }

    def detect_backlight(self, image: np.ndarray) -> Dict:
        """Detect backlight conditions where background is much brighter than subject."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape

        chs, che = h // 4, 3 * h // 4
        cws, cwe = w // 4, 3 * w // 4
        center_region = gray[chs:che, cws:cwe]

        top_edge = gray[: h // 6, :]
        bottom_edge = gray[5 * h // 6 :, :]
        left_edge = gray[:, : w // 6]
        right_edge = gray[:, 5 * w // 6 :]

        center_brightness = float(np.mean(center_region))
        edge_brightness = float(
            np.mean(
                np.concatenate(
                    [
                        top_edge.flatten(),
                        bottom_edge.flatten(),
                        left_edge.flatten(),
                        right_edge.flatten(),
                    ]
                )
            )
        )

        brightness_diff = edge_brightness - center_brightness
        brightness_ratio = edge_brightness / max(center_brightness, 1.0)

        edge_bright_ratio = float(
            np.sum(
                np.concatenate(
                    [
                        top_edge.flatten(),
                        bottom_edge.flatten(),
                        left_edge.flatten(),
                        right_edge.flatten(),
                    ]
                )
                > 200
            )
            / (top_edge.size + bottom_edge.size + left_edge.size + right_edge.size)
        )
        center_dark_ratio = float(np.sum(center_region < 80) / center_region.size)

        has_backlight = False
        severity = "none"

        if brightness_diff > 30 and brightness_ratio > 1.5:
            has_backlight = True
            if brightness_diff > 80 or brightness_ratio > 2.2:
                severity = "severe"
            elif brightness_diff > 50 or brightness_ratio > 1.8:
                severity = "moderate"
            else:
                severity = "mild"
        elif brightness_diff > 20 and edge_bright_ratio > 0.2 and center_dark_ratio > 0.3:
            has_backlight = True
            severity = "mild"

        if center_brightness < 60 and edge_brightness > 150:
            has_backlight = True
            if center_brightness < 40:
                severity = "severe"
            elif center_brightness < 50:
                severity = "moderate"
            else:
                severity = "mild"

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        dark_pixels = float(np.sum(hist[:60]) / gray.size)
        bright_pixels = float(np.sum(hist[200:]) / gray.size)
        if dark_pixels > 0.15 and bright_pixels > 0.1 and not has_backlight:
            has_backlight = True
            severity = "moderate"

        gray_std = float(np.std(gray))
        if gray_std > 60 and not has_backlight:
            very_dark = float(np.sum(gray < 50) / gray.size)
            very_bright = float(np.sum(gray > 180) / gray.size)
            if very_dark > 0.1 and very_bright > 0.1:
                has_backlight = True
                severity = "moderate"

        return {
            "has_backlight": has_backlight,
            "severity": severity,
            "center_brightness": center_brightness,
            "edge_brightness": edge_brightness,
            "brightness_difference": brightness_diff,
            "brightness_ratio": brightness_ratio,
            "edge_bright_ratio": edge_bright_ratio,
            "center_dark_ratio": center_dark_ratio,
            "confidence": self._calculate_backlight_confidence(
                brightness_diff, brightness_ratio, edge_bright_ratio, center_dark_ratio
            ),
        }

    def detect_blur(self, image: np.ndarray) -> Dict:
        """Detect blur/focus issues using Laplacian variance and Sobel gradients."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mean = float(np.mean(np.sqrt(sobelx ** 2 + sobely ** 2)))

        is_blurry = laplacian_var < 100.0
        severity = "severe" if laplacian_var < 50 else "moderate" if laplacian_var < 100 else "mild"

        return {
            "laplacian_variance": laplacian_var,
            "sobel_mean": sobel_mean,
            "is_blurry": is_blurry,
            "sharpness_score": float(min(100.0, laplacian_var / 5.0)),
            "severity": severity,
        }

    def detect_all_issues(self, image: Union[str, np.ndarray]) -> Dict:
        """
        Comprehensive analysis of image quality issues.
        Accepts an image path or a numpy array (BGR or grayscale).
        """
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image path does not exist: {image}")
            img = cv2.imread(image)
            if img is None:
                raise ValueError("Failed to read image with OpenCV")
        else:
            img = image

        noise = self.detect_noise(img)
        lighting = self.detect_lighting(img)
        backlight = self.detect_backlight(img)
        blur = self.detect_blur(img)

        quality_score = self._calculate_overall_quality(noise, lighting, backlight, blur)
        recommendations = self._generate_recommendations(noise, lighting, backlight, blur)

        return {
            "image_info": {
                "dimensions": (int(img.shape[1]), int(img.shape[0])),
                "channels": int(img.shape[2]) if len(img.shape) == 3 else 1,
            },
            "noise": noise,
            "lighting": lighting,
            "backlight": backlight,
            "blur": blur,
            "overall_quality_score": quality_score,
            "needs_correction": quality_score < 0.7,
            "recommendations": recommendations,
            "priority_issues": self._get_priority_issues(noise, lighting, backlight, blur),
        }

    def _assess_lighting_severity(self, brightness: float, contrast: float) -> str:
        if brightness < 40 or contrast < 20:
            return "severe"
        elif brightness < 60 or contrast < 30:
            return "moderate"
        elif brightness < 80 or contrast < 40:
            return "mild"
        else:
            return "none"

    def _calculate_backlight_confidence(
        self,
        brightness_diff: float,
        brightness_ratio: float,
        edge_bright_ratio: float,
        center_dark_ratio: float,
    ) -> float:
        confidence = 0.0
        if brightness_diff > 100:
            confidence += 0.4
        elif brightness_diff > 50:
            confidence += 0.2
        if brightness_ratio > 2.5:
            confidence += 0.3
        elif brightness_ratio > 1.8:
            confidence += 0.2
        if edge_bright_ratio > 0.3 and center_dark_ratio > 0.4:
            confidence += 0.3
        return float(min(1.0, confidence))

    def _calculate_overall_quality(self, noise: Dict, lighting: Dict, backlight: Dict, blur: Dict) -> float:
        score = 1.0
        if noise.get("is_noisy"):
            level = noise.get("noise_level")
            if level == "high":
                score -= 0.3
            elif level == "medium":
                score -= 0.2
            else:
                score -= 0.1
        if lighting.get("is_low_light"):
            score -= 0.3
        if lighting.get("is_low_contrast"):
            score -= 0.2
        if backlight.get("has_backlight"):
            sev = backlight.get("severity")
            if sev == "severe":
                score -= 0.3
            elif sev == "moderate":
                score -= 0.2
            else:
                score -= 0.1
        if blur.get("is_blurry"):
            sev = blur.get("severity")
            if sev == "severe":
                score -= 0.4
            elif sev == "moderate":
                score -= 0.25
            else:
                score -= 0.1
        return float(max(0.0, score))

    def _generate_recommendations(self, noise: Dict, lighting: Dict, backlight: Dict, blur: Dict) -> List[str]:
        recs: List[str] = []
        if blur.get("is_blurry"):
            recs.append(f"Apply sharpening filter ({blur.get('severity')} blur detected)")
        if backlight.get("has_backlight"):
            recs.append(f"Apply backlight correction ({backlight.get('severity')} backlight)")
        if lighting.get("is_low_light"):
            recs.append("Apply low-light enhancement")
        if lighting.get("is_low_contrast"):
            recs.append("Apply contrast enhancement (CLAHE)")
        if noise.get("is_noisy"):
            recs.append(f"Apply noise reduction ({noise.get('noise_level')} noise)")
        return recs

    def _get_priority_issues(self, noise: Dict, lighting: Dict, backlight: Dict, blur: Dict) -> List[str]:
        issues: List[str] = []
        if blur.get("severity") == "severe":
            issues.append("severe_blur")
        if backlight.get("severity") == "severe":
            issues.append("severe_backlight")
        if lighting.get("mean_brightness", 0.0) < 40:
            issues.append("very_low_light")
        if blur.get("severity") == "moderate":
            issues.append("moderate_blur")
        if backlight.get("severity") == "moderate":
            issues.append("moderate_backlight")
        if lighting.get("is_low_light"):
            issues.append("low_light")
        if lighting.get("is_low_contrast"):
            issues.append("low_contrast")
        if noise.get("is_noisy"):
            issues.append("noise")
        if backlight.get("severity") == "mild":
            issues.append("mild_backlight")
        if blur.get("severity") == "mild":
            issues.append("mild_blur")
        return issues


class ImageProcessor:
    """Handles image processing and enhancement operations using OpenCV."""
    
    def __init__(self):
        self.use_opencv = True  # Flag to switch between OpenCV and PIL
        self.original_image = None  # Store original OpenCV image
        self.analyzer = ImageAnalyzer()  # Initialize image analyzer
    
    def analyze_image(self, pixmap):
        """
        Analyze a QPixmap using ImageAnalyzer.
        
        Args:
            pixmap (QPixmap): Image pixmap to analyze
            
        Returns:
            Dict: Analysis results from ImageAnalyzer
        """
        if not pixmap:
            return None
            
        try:
            # Convert QPixmap to OpenCV image
            cv_image = self.qpixmap_to_cv2(pixmap)
            
            # Use ImageAnalyzer to detect all issues
            analysis = self.analyzer.detect_all_issues(cv_image)
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None

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
    
    def apply_bilateral_filter(self, pixmap, d=9, sigma_color=75, sigma_space=75):
        """
        Apply bilateral filter to reduce noise while keeping edges sharp.
        
        Args:
            pixmap (QPixmap): Input image pixmap
            d (int): Diameter of each pixel neighborhood (5-15 recommended)
            sigma_color (float): Filter sigma in the color space (10-150)
            sigma_space (float): Filter sigma in the coordinate space (10-150)
            
        Returns:
            QPixmap: Bilateral filtered image
        """
        try:
            cv_image = self.qpixmap_to_cv2(pixmap)
            
            # Apply bilateral filter
            # d: Diameter of each pixel neighborhood. If negative, computed from sigma_space
            # sigma_color: Filter sigma in the color space. Larger value means farther colors mix more
            # sigma_space: Filter sigma in the coordinate space. Larger value means farther pixels mix more
            cv_image = cv2.bilateralFilter(cv_image, d, sigma_color, sigma_space)
            
            return self.cv2_to_qpixmap(cv_image)
            
        except Exception as e:
            print(f"Error applying bilateral filter: {e}")
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
    
    def analyze_image(self, pixmap):
        """Analyze image quality using ImageAnalyzer."""
        try:
            cv_image = self.qpixmap_to_cv2(pixmap)
            analysis = self.analyzer.detect_all_issues(cv_image)
            return analysis
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None