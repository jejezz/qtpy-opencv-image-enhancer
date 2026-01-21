"""
Main Window for Qt Image Enhancer
"""

import os
from qtpy.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QMessageBox,
    QSlider, QGroupBox, QGridLayout, QStatusBar,
    QFrame, QScrollArea, QTextEdit
)
from qtpy.QtCore import Qt, Signal, QSettings
from qtpy.QtGui import QPixmap, QIcon, QPalette, QColor, QImage
from src.core.image_processor import ImageProcessor
from src.core.face_recognition_api import FaceRecognitionAPI
import cv2
import numpy as np
import tempfile
import os
import configparser
from pathlib import Path


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.image_processor = ImageProcessor()
        self.face_api = FaceRecognitionAPI()
        self.current_image_path = None
        self.original_pixmap = None
        self.current_working_pixmap = None  # Current state with all enhancements and filters applied
        self.truly_original_pixmap = None  # Store the unmodified original image
        self.current_analysis = None  # Store current image analysis
        self.extracted_face_pixmap = None  # Store extracted face
        self.original_extracted_face_pixmap = None  # Store original unenhanced extracted face
        self.extracted_face_quality = None  # Store face quality info
        self.recognition_result = None  # Store recognition result
        
        # Initialize config
        self.config_file = Path("config.ini")
        self.init_config()
        
        self.init_ui()
        self.connect_signals()
        self.load_config_values()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Qt Image Enhancer")
        self.setGeometry(100, 100, 1000, 700)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel for controls
        self.create_control_panel(main_layout)
        
        # Center panel for image display
        self.create_image_panel(main_layout)
        
        # Right panel for face recognition
        self.create_face_recognition_panel(main_layout)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def create_control_panel(self, parent_layout):
        """Create the left control panel."""
        control_widget = QWidget()
        control_widget.setFixedWidth(300)
        control_layout = QVBoxLayout(control_widget)
        
        # File operations group
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout(file_group)
        
        self.load_btn = QPushButton("Load Image")
        self.save_btn = QPushButton("Save Image")
        self.save_btn.setEnabled(False)
        
        file_layout.addWidget(self.load_btn)
        file_layout.addWidget(self.save_btn)
        
        # Enhancement controls group
        enhance_group = QGroupBox("Image Enhancement")
        enhance_layout = QGridLayout(enhance_group)
        
        # Grayscale button
        self.grayscale_btn = QPushButton("Convert to Grayscale")
        self.grayscale_btn.setEnabled(False)
        self.grayscale_btn.setStyleSheet("""
            QPushButton {
                background-color: #9E9E9E;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
                margin-bottom: 8px;
            }
            QPushButton:hover {
                background-color: #757575;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        enhance_layout.addWidget(self.grayscale_btn, 0, 0, 1, 3)
        
        # Brightness control
        enhance_layout.addWidget(QLabel("Brightness:"), 1, 0)
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_label = QLabel("0")
        enhance_layout.addWidget(self.brightness_slider, 1, 1)
        enhance_layout.addWidget(self.brightness_label, 1, 2)
        
        # Contrast control
        enhance_layout.addWidget(QLabel("Contrast:"), 2, 0)
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(0)
        self.contrast_label = QLabel("0")
        enhance_layout.addWidget(self.contrast_slider, 2, 1)
        enhance_layout.addWidget(self.contrast_label, 2, 2)
        
        # Saturation control
        enhance_layout.addWidget(QLabel("Saturation:"), 3, 0)
        self.saturation_slider = QSlider(Qt.Orientation.Horizontal)
        self.saturation_slider.setRange(-100, 100)
        self.saturation_slider.setValue(0)
        self.saturation_label = QLabel("0")
        enhance_layout.addWidget(self.saturation_slider, 3, 1)
        enhance_layout.addWidget(self.saturation_label, 3, 2)
        
        # OpenCV Filters group
        filters_group = QGroupBox("OpenCV Filters")
        filters_layout = QVBoxLayout(filters_group)
        
        # Gaussian Blur controls
        blur_group = QGroupBox("Gaussian Blur")
        blur_layout = QGridLayout(blur_group)
        
        # Kernel Size control
        blur_layout.addWidget(QLabel("Kernel Size:"), 0, 0)
        self.blur_kernel_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_kernel_slider.setRange(1, 15)  # Will be converted to odd numbers (1,3,5...31)
        self.blur_kernel_slider.setValue(7)  # Default: 15x15 kernel
        self.blur_kernel_label = QLabel("15")
        blur_layout.addWidget(self.blur_kernel_slider, 0, 1)
        blur_layout.addWidget(self.blur_kernel_label, 0, 2)
        
        # Sigma X control
        blur_layout.addWidget(QLabel("Sigma X:"), 1, 0)
        self.blur_sigmax_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_sigmax_slider.setRange(0, 100)  # 0.0 to 10.0 (divided by 10)
        self.blur_sigmax_slider.setValue(0)  # 0 = auto-calculate from kernel size
        self.blur_sigmax_label = QLabel("Auto")
        blur_layout.addWidget(self.blur_sigmax_slider, 1, 1)
        blur_layout.addWidget(self.blur_sigmax_label, 1, 2)
        
        # Sigma Y control
        blur_layout.addWidget(QLabel("Sigma Y:"), 2, 0)
        self.blur_sigmay_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_sigmay_slider.setRange(0, 100)  # 0.0 to 10.0 (divided by 10)
        self.blur_sigmay_slider.setValue(0)  # 0 = use sigmaX value
        self.blur_sigmay_label = QLabel("Auto")
        blur_layout.addWidget(self.blur_sigmay_slider, 2, 1)
        blur_layout.addWidget(self.blur_sigmay_label, 2, 2)
        
        # Apply Blur button
        self.apply_blur_btn = QPushButton("Apply Gaussian Blur")
        self.apply_blur_btn.setEnabled(False)
        blur_layout.addWidget(self.apply_blur_btn, 3, 0, 1, 3)
        
        # Bilateral Filter controls
        bilateral_group = QGroupBox("Bilateral Filter (Edge-Preserving)")
        bilateral_layout = QGridLayout(bilateral_group)
        
        # d (diameter of pixel neighborhood)
        bilateral_layout.addWidget(QLabel("Diameter (d):"), 0, 0)
        self.bilateral_d_slider = QSlider(Qt.Orientation.Horizontal)
        self.bilateral_d_slider.setRange(5, 15)  # Recommended range 5-15
        self.bilateral_d_slider.setValue(9)  # Default
        self.bilateral_d_label = QLabel("9")
        bilateral_layout.addWidget(self.bilateral_d_slider, 0, 1)
        bilateral_layout.addWidget(self.bilateral_d_label, 0, 2)
        
        # sigma_color (filter sigma in color space)
        bilateral_layout.addWidget(QLabel("Color Sigma:"), 1, 0)
        self.bilateral_color_slider = QSlider(Qt.Orientation.Horizontal)
        self.bilateral_color_slider.setRange(10, 150)  # Recommended range 10-150
        self.bilateral_color_slider.setValue(75)  # Default
        self.bilateral_color_label = QLabel("75")
        bilateral_layout.addWidget(self.bilateral_color_slider, 1, 1)
        bilateral_layout.addWidget(self.bilateral_color_label, 1, 2)
        
        # sigma_space (filter sigma in coordinate space)
        bilateral_layout.addWidget(QLabel("Space Sigma:"), 2, 0)
        self.bilateral_space_slider = QSlider(Qt.Orientation.Horizontal)
        self.bilateral_space_slider.setRange(10, 150)  # Recommended range 10-150
        self.bilateral_space_slider.setValue(75)  # Default
        self.bilateral_space_label = QLabel("75")
        bilateral_layout.addWidget(self.bilateral_space_slider, 2, 1)
        bilateral_layout.addWidget(self.bilateral_space_label, 2, 2)
        
        # Apply Bilateral Filter button
        self.apply_bilateral_btn = QPushButton("Apply Bilateral Filter")
        self.apply_bilateral_btn.setEnabled(False)
        bilateral_layout.addWidget(self.apply_bilateral_btn, 3, 0, 1, 3)
        
        # Noise Reduction controls
        denoise_group = QGroupBox("Noise Reduction (Non-local Means)")
        denoise_layout = QGridLayout(denoise_group)
        
        # h (luminance filter strength)
        denoise_layout.addWidget(QLabel("h (Luminance):"), 0, 0)
        self.denoise_h_slider = QSlider(Qt.Orientation.Horizontal)
        self.denoise_h_slider.setRange(1, 30)  # 1 to 30 (typical range)
        self.denoise_h_slider.setValue(10)  # Default: 10
        self.denoise_h_label = QLabel("10")
        denoise_layout.addWidget(self.denoise_h_slider, 0, 1)
        denoise_layout.addWidget(self.denoise_h_label, 0, 2)
        
        # hColor (color filter strength)
        denoise_layout.addWidget(QLabel("hColor:"), 1, 0)
        self.denoise_hcolor_slider = QSlider(Qt.Orientation.Horizontal)
        self.denoise_hcolor_slider.setRange(1, 30)  # 1 to 30
        self.denoise_hcolor_slider.setValue(10)  # Default: 10
        self.denoise_hcolor_label = QLabel("10")
        denoise_layout.addWidget(self.denoise_hcolor_slider, 1, 1)
        denoise_layout.addWidget(self.denoise_hcolor_label, 1, 2)
        
        # Template window size
        denoise_layout.addWidget(QLabel("Template Size:"), 2, 0)
        self.denoise_template_slider = QSlider(Qt.Orientation.Horizontal)
        self.denoise_template_slider.setRange(1, 4)  # Will convert to 3,5,7,9
        self.denoise_template_slider.setValue(2)  # Default: 7
        self.denoise_template_label = QLabel("7")
        denoise_layout.addWidget(self.denoise_template_slider, 2, 1)
        denoise_layout.addWidget(self.denoise_template_label, 2, 2)
        
        # Search window size
        denoise_layout.addWidget(QLabel("Search Size:"), 3, 0)
        self.denoise_search_slider = QSlider(Qt.Orientation.Horizontal)
        self.denoise_search_slider.setRange(1, 6)  # Will convert to 15,17,19,21,23,25
        self.denoise_search_slider.setValue(4)  # Default: 21
        self.denoise_search_label = QLabel("21")
        denoise_layout.addWidget(self.denoise_search_slider, 3, 1)
        denoise_layout.addWidget(self.denoise_search_label, 3, 2)
        
        # Apply Denoise button
        self.apply_denoise_btn = QPushButton("Apply Noise Reduction")
        self.apply_denoise_btn.setEnabled(False)
        denoise_layout.addWidget(self.apply_denoise_btn, 4, 0, 1, 3)
        
        # Other filter buttons
        self.sharpen_btn = QPushButton("Sharpen")
        self.edge_btn = QPushButton("Edge Detection")
        self.emboss_btn = QPushButton("Emboss")
        self.histogram_btn = QPushButton("Auto Contrast (Histogram)")
        
        # Disable filter buttons initially
        self.sharpen_btn.setEnabled(False)
        self.edge_btn.setEnabled(False)
        self.emboss_btn.setEnabled(False)
        self.histogram_btn.setEnabled(False)
        
        filters_layout.addWidget(blur_group)
        filters_layout.addWidget(bilateral_group)
        filters_layout.addWidget(denoise_group)
        filters_layout.addWidget(self.sharpen_btn)
        filters_layout.addWidget(self.edge_btn)
        filters_layout.addWidget(self.emboss_btn)
        filters_layout.addWidget(self.histogram_btn)
        
        # Reset button
        self.reset_btn = QPushButton("Reset All")
        self.reset_btn.setEnabled(False)
        
        # Add to control layout
        control_layout.addWidget(file_group)
        control_layout.addWidget(enhance_group)
        control_layout.addWidget(filters_group)
        control_layout.addWidget(self.reset_btn)
        control_layout.addStretch()
        
        parent_layout.addWidget(control_widget)
    
    def create_image_panel(self, parent_layout):
        """Create the right image display panel."""
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        
        # Add Analyze button at the top
        self.analyze_btn = QPushButton("Analyze Current Image")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
                margin-bottom: 5px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        image_layout.addWidget(self.analyze_btn)
        
        # Create a container for the image and overlay
        self.image_container = QWidget()
        self.image_container.setMinimumSize(600, 500)
        
        # Image display label
        self.image_label = QLabel(self.image_container)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #ccc;
                border-radius: 10px;
                background-color: #f9f9f9;
                color: #666;
                font-size: 16px;
            }
        """)
        self.image_label.setText("Click 'Load Image' to start")
        self.image_label.setGeometry(0, 0, 600, 500)
        
        # Create analysis overlay
        self.create_analysis_overlay()
        
        image_layout.addWidget(self.image_container)
        parent_layout.addWidget(image_widget, 2)  # Give more space to image panel
    
    def create_analysis_overlay(self):
        """Create transparent overlay for image analysis display."""
        # Analysis overlay frame
        self.analysis_overlay = QFrame(self.image_container)
        self.analysis_overlay.setFixedSize(320, 300)
        self.analysis_overlay.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 0, 0, 180);
                border: 1px solid rgba(255, 255, 255, 100);
                border-radius: 8px;
            }
        """)
        self.analysis_overlay.hide()  # Initially hidden
        
        # Analysis text display
        overlay_layout = QVBoxLayout(self.analysis_overlay)
        overlay_layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title_label = QLabel("Image Analysis")
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 14px;
                font-weight: bold;
                margin-bottom: 5px;
            }
        """)
        overlay_layout.addWidget(title_label)
        
        # Scrollable analysis text
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setStyleSheet("""
            QTextEdit {
                background-color: transparent;
                border: none;
                color: white;
                font-size: 11px;
                font-family: 'Consolas', 'Monaco', monospace;
            }
            QScrollBar:vertical {
                background-color: rgba(255, 255, 255, 30);
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: rgba(255, 255, 255, 100);
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: rgba(255, 255, 255, 150);
            }
        """)
        overlay_layout.addWidget(self.analysis_text)
    
    def create_face_recognition_panel(self, parent_layout):
        """Create the right face recognition panel."""
        face_widget = QWidget()
        face_widget.setFixedWidth(320)
        face_layout = QVBoxLayout(face_widget)
        
        # Face extraction group
        extraction_group = QGroupBox("Face Extraction")
        extraction_layout = QVBoxLayout(extraction_group)
        
        self.extract_face_btn = QPushButton("Extract Face")
        self.extract_face_btn.setEnabled(False)
        self.extract_face_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        extraction_layout.addWidget(self.extract_face_btn)
        
        # Face image display
        self.face_image_label = QLabel()
        self.face_image_label.setFixedSize(280, 200)
        self.face_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.face_image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #ccc;
                border-radius: 8px;
                background-color: #f9f9f9;
                color: #666;
                font-size: 12px;
            }
        """)
        self.face_image_label.setText("No face extracted")
        extraction_layout.addWidget(self.face_image_label)
        
        # Face quality display
        quality_group = QGroupBox("Face Quality")
        quality_layout = QVBoxLayout(quality_group)
        
        self.face_quality_text = QTextEdit()
        self.face_quality_text.setReadOnly(True)
        self.face_quality_text.setMaximumHeight(120)
        self.face_quality_text.setStyleSheet("""
            QTextEdit {
                background-color: #011324;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 5px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
            }
        """)
        self.face_quality_text.setPlainText("No face extracted yet")
        quality_layout.addWidget(self.face_quality_text)
        
        # Face recognition group
        recognition_group = QGroupBox("Face Recognition")
        recognition_layout = QVBoxLayout(recognition_group)
        
        self.recognize_face_btn = QPushButton("Recognize Face")
        self.recognize_face_btn.setEnabled(False)
        self.recognize_face_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        recognition_layout.addWidget(self.recognize_face_btn)
        
        # Recognition result display
        result_group = QGroupBox("Recognition Result")
        result_layout = QVBoxLayout(result_group)
        
        self.recognition_result_text = QTextEdit()
        self.recognition_result_text.setReadOnly(True)
        self.recognition_result_text.setMaximumHeight(100)
        self.recognition_result_text.setStyleSheet("""
            QTextEdit {
                background-color: #011324;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 5px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
            }
        """)
        self.recognition_result_text.setPlainText("No recognition attempted yet")
        result_layout.addWidget(self.recognition_result_text)
        
        # Add all groups to main layout
        face_layout.addWidget(extraction_group)
        face_layout.addWidget(quality_group)
        face_layout.addWidget(recognition_group)
        face_layout.addWidget(result_group)
        face_layout.addStretch()
        
        parent_layout.addWidget(face_widget)

    def update_analysis_display(self, analysis):
        """Update the analysis overlay with new analysis data."""
        if not analysis:
            self.analysis_overlay.hide()
            return
            
        # Format analysis data
        text_parts = []
        
        # Quality score
        score = analysis.get('overall_quality_score', 0.0)
        quality_status = "GOOD" if score >= 0.7 else "POOR" if score < 0.4 else "FAIR"
        text_parts.append(f"🎯 Quality: {quality_status} ({score:.2f})")
        
        # Image info
        info = analysis.get('image_info', {})
        dims = info.get('dimensions', (0, 0))
        text_parts.append(f"📐 Size: {dims[0]}x{dims[1]}")
        
        # Lighting analysis
        lighting = analysis.get('lighting', {})
        brightness = lighting.get('mean_brightness', 0)
        contrast = lighting.get('contrast', 0)
        text_parts.append(f"💡 Brightness: {brightness:.1f}")
        text_parts.append(f"🔳 Contrast: {contrast:.1f}")
        
        # Issues detection
        issues = []
        if analysis.get('noise', {}).get('is_noisy'):
            noise_level = analysis['noise'].get('noise_level', 'unknown')
            issues.append(f"Noise ({noise_level})")
        if analysis.get('blur', {}).get('is_blurry'):
            blur_severity = analysis['blur'].get('severity', 'unknown')
            issues.append(f"Blur ({blur_severity})")
        if analysis.get('backlight', {}).get('has_backlight'):
            backlight_severity = analysis['backlight'].get('severity', 'unknown')
            issues.append(f"Backlight ({backlight_severity})")
        if analysis.get('lighting', {}).get('is_low_light'):
            issues.append("Low light")
        if analysis.get('lighting', {}).get('is_low_contrast'):
            issues.append("Low contrast")
            
        if issues:
            text_parts.append("\n⚠️ Issues:")
            for issue in issues:
                text_parts.append(f"  • {issue}")
        else:
            text_parts.append("\n✅ No major issues")
            
        # Recommendations
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            text_parts.append("\n💡 Recommendations:")
            for rec in recommendations[:3]:  # Limit to top 3
                text_parts.append(f"  • {rec}")
                
        # Technical details
        text_parts.append("\n🔧 Technical:")
        if 'noise' in analysis:
            psnr = analysis['noise'].get('psnr', 0)
            laplacian = analysis['noise'].get('laplacian_variance', 0)
            text_parts.append(f"  PSNR: {psnr:.1f}dB")
            text_parts.append(f"  Sharpness: {laplacian:.1f}")
        
        self.analysis_text.setPlainText('\n'.join(text_parts))
        self.analysis_overlay.show()
    
    def position_analysis_overlay(self):
        """Position the analysis overlay in the bottom-right of the image container."""
        if hasattr(self, 'analysis_overlay'):
            container_rect = self.image_container.geometry()
            overlay_x = container_rect.width() - self.analysis_overlay.width() - 10
            overlay_y = container_rect.height() - self.analysis_overlay.height() - 10
            self.analysis_overlay.move(max(10, overlay_x), max(10, overlay_y))
    
    def connect_signals(self):
        """Connect UI signals to their handlers."""
        self.load_btn.clicked.connect(self.load_image)
        self.save_btn.clicked.connect(self.save_image)
        self.reset_btn.clicked.connect(self.reset_adjustments)
        
        # Connect sliders
        self.brightness_slider.valueChanged.connect(self.on_brightness_changed)
        self.contrast_slider.valueChanged.connect(self.on_contrast_changed)
        self.saturation_slider.valueChanged.connect(self.on_saturation_changed)
        
        # Connect enhancement buttons
        self.grayscale_btn.clicked.connect(self.apply_grayscale)
        
        # Connect OpenCV filter buttons
        self.apply_blur_btn.clicked.connect(self.apply_gaussian_blur)
        self.apply_bilateral_btn.clicked.connect(self.apply_bilateral_filter)
        self.apply_denoise_btn.clicked.connect(self.apply_noise_reduction)
        self.sharpen_btn.clicked.connect(lambda: self.apply_filter('sharpen'))
        self.edge_btn.clicked.connect(lambda: self.apply_filter('edge'))
        self.emboss_btn.clicked.connect(lambda: self.apply_filter('emboss'))
        self.histogram_btn.clicked.connect(self.apply_histogram_equalization)
        
        # Connect Gaussian Blur sliders
        self.blur_kernel_slider.valueChanged.connect(self.on_blur_kernel_changed)
        self.blur_sigmax_slider.valueChanged.connect(self.on_blur_sigmax_changed)
        self.blur_sigmay_slider.valueChanged.connect(self.on_blur_sigmay_changed)
        
        # Connect Bilateral Filter sliders
        self.bilateral_d_slider.valueChanged.connect(self.on_bilateral_d_changed)
        self.bilateral_color_slider.valueChanged.connect(self.on_bilateral_color_changed)
        self.bilateral_space_slider.valueChanged.connect(self.on_bilateral_space_changed)
        
        # Connect Noise Reduction sliders
        self.denoise_h_slider.valueChanged.connect(self.on_denoise_h_changed)
        self.denoise_hcolor_slider.valueChanged.connect(self.on_denoise_hcolor_changed)
        self.denoise_template_slider.valueChanged.connect(self.on_denoise_template_changed)
        self.denoise_search_slider.valueChanged.connect(self.on_denoise_search_changed)
        
        # Connect face recognition buttons
        self.extract_face_btn.clicked.connect(self.extract_face)
        self.recognize_face_btn.clicked.connect(self.recognize_face)
        
        # Connect analyze button
        self.analyze_btn.clicked.connect(self.analyze_current_image)
    
    def init_config(self):
        """Initialize configuration file."""
        self.config = configparser.ConfigParser()
        if self.config_file.exists():
            self.config.read(self.config_file)
        else:
            # Create default config
            self.config['DEFAULT'] = {
                'last_directory': str(Path.home()),
                'last_brightness': '0',
                'last_contrast': '0', 
                'last_saturation': '0',
                'last_bilateral_d': '9',
                'last_bilateral_sigma_color': '75',
                'last_bilateral_sigma_space': '75',
                'last_denoise_h': '3',
                'last_denoise_template_window_size': '7',
                'last_denoise_search_window_size': '21'
            }
            self.save_config()
    
    def save_config(self):
        """Save current values to config file."""
        try:
            with open(self.config_file, 'w') as f:
                self.config.write(f)
        except Exception as e:
            print(f"Failed to save config: {e}")
    
    def load_config_values(self):
        """Load saved values from config to UI."""
        try:
            # Load enhancement values
            self.brightness_slider.setValue(int(self.config.get('DEFAULT', 'last_brightness', fallback='0')))
            self.contrast_slider.setValue(int(self.config.get('DEFAULT', 'last_contrast', fallback='0')))
            self.saturation_slider.setValue(int(self.config.get('DEFAULT', 'last_saturation', fallback='0')))
            
            # Load filter values
            self.bilateral_d_slider.setValue(int(self.config.get('DEFAULT', 'last_bilateral_d', fallback='9')))
            self.bilateral_color_slider.setValue(int(self.config.get('DEFAULT', 'last_bilateral_sigma_color', fallback='75')))
            self.bilateral_space_slider.setValue(int(self.config.get('DEFAULT', 'last_bilateral_sigma_space', fallback='75')))
            self.denoise_h_slider.setValue(int(self.config.get('DEFAULT', 'last_denoise_h', fallback='3')))
            self.denoise_template_slider.setValue(int(self.config.get('DEFAULT', 'last_denoise_template_window_size', fallback='7')))
            self.denoise_search_slider.setValue(int(self.config.get('DEFAULT', 'last_denoise_search_window_size', fallback='21')))
        except Exception as e:
            print(f"Failed to load config values: {e}")
    
    def save_current_values_to_config(self):
        """Save current slider values and directory to config."""
        try:
            if self.current_image_path:
                self.config['DEFAULT']['last_directory'] = str(Path(self.current_image_path).parent)
            
            # Save enhancement values
            self.config['DEFAULT']['last_brightness'] = str(self.brightness_slider.value())
            self.config['DEFAULT']['last_contrast'] = str(self.contrast_slider.value())
            self.config['DEFAULT']['last_saturation'] = str(self.saturation_slider.value())
            
            # Save filter values
            self.config['DEFAULT']['last_bilateral_d'] = str(self.bilateral_d_slider.value())
            self.config['DEFAULT']['last_bilateral_sigma_color'] = str(self.bilateral_color_slider.value())
            self.config['DEFAULT']['last_bilateral_sigma_space'] = str(self.bilateral_space_slider.value())
            self.config['DEFAULT']['last_denoise_h'] = str(self.denoise_h_slider.value())
            self.config['DEFAULT']['last_denoise_template_window_size'] = str(self.denoise_template_slider.value())
            self.config['DEFAULT']['last_denoise_search_window_size'] = str(self.denoise_search_slider.value())
            
            self.save_config()
        except Exception as e:
            print(f"Failed to save current values to config: {e}")
    
    def load_image(self):
        """Load an image file."""
        # Get last directory from config
        last_dir = self.config.get('DEFAULT', 'last_directory', fallback=str(Path.home()))
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image File",
            last_dir,
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)"
        )
        
        if file_path:
            try:
                self.current_image_path = file_path
                self.original_pixmap = QPixmap(file_path)
                self.current_working_pixmap = QPixmap(file_path)  # Initialize working copy
                self.truly_original_pixmap = QPixmap(file_path)  # Store the true original
                self.display_image(self.original_pixmap)
                
                # Save directory to config
                self.save_current_values_to_config()
                
                # Enable controls
                self.save_btn.setEnabled(True)
                self.reset_btn.setEnabled(True)
                self.grayscale_btn.setEnabled(True)
                
                # Enable filter buttons
                self.apply_blur_btn.setEnabled(True)
                self.apply_bilateral_btn.setEnabled(True)
                self.apply_denoise_btn.setEnabled(True)
                self.sharpen_btn.setEnabled(True)
                self.edge_btn.setEnabled(True)
                self.emboss_btn.setEnabled(True)
                self.histogram_btn.setEnabled(True)
                
                # Enable face extraction
                self.extract_face_btn.setEnabled(True)
                
                # Enable analyze button
                self.analyze_btn.setEnabled(True)
                
                # Reset sliders to config values
                self.reset_adjustments()
                
                # Auto-analyze the image
                self.analyze_current_image()
                
                # Auto-extract face with anti-spoofing
                self.auto_extract_face()
                
                self.status_bar.showMessage(f"Loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")
    
    def save_image(self):
        """Save the current image."""
        if not self.current_image_path:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        
        if file_path:
            try:
                # Get current displayed pixmap and save it
                current_pixmap = self.image_label.pixmap()
                if current_pixmap:
                    current_pixmap.save(file_path)
                    self.status_bar.showMessage(f"Saved: {os.path.basename(file_path)}")
                    QMessageBox.information(self, "Success", "Image saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image:\n{str(e)}")
    
    def display_image(self, pixmap):
        """Display pixmap in the image label with proper scaling."""
        if pixmap:
            # Scale pixmap to fit label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            # Position the analysis overlay
            self.position_analysis_overlay()
    
    def display_face_image(self, pixmap):
        """Display face pixmap in the face image label with proper scaling."""
        if pixmap and hasattr(self, 'face_image_label'):
            # Scale pixmap to fit face label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.face_image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.face_image_label.setPixmap(scaled_pixmap)
    
    def reset_adjustments(self):
        """Reset all adjustment sliders to config values."""
        try:
            # Load values from config
            self.brightness_slider.setValue(int(self.config.get('DEFAULT', 'last_brightness', fallback='0')))
            self.contrast_slider.setValue(int(self.config.get('DEFAULT', 'last_contrast', fallback='0')))
            self.saturation_slider.setValue(int(self.config.get('DEFAULT', 'last_saturation', fallback='0')))
            
            # Reset filter sliders to config values
            self.bilateral_d_slider.setValue(int(self.config.get('DEFAULT', 'last_bilateral_d', fallback='9')))
            self.bilateral_color_slider.setValue(int(self.config.get('DEFAULT', 'last_bilateral_sigma_color', fallback='75')))
            self.bilateral_space_slider.setValue(int(self.config.get('DEFAULT', 'last_bilateral_sigma_space', fallback='75')))
            self.denoise_h_slider.setValue(int(self.config.get('DEFAULT', 'last_denoise_h', fallback='3')))
            self.denoise_template_slider.setValue(int(self.config.get('DEFAULT', 'last_denoise_template_window_size', fallback='7')))
            self.denoise_search_slider.setValue(int(self.config.get('DEFAULT', 'last_denoise_search_window_size', fallback='21')))
        except Exception as e:
            print(f"Error loading config values, using defaults: {e}")
            # Fallback to default values
            self.brightness_slider.setValue(0)
            self.contrast_slider.setValue(0)
            self.saturation_slider.setValue(0)
        
        # Reload the original image from file to refresh in-memory image
        if self.current_image_path:
            try:
                # Reload fresh from file
                self.original_pixmap = QPixmap(self.current_image_path)
                self.current_working_pixmap = QPixmap(self.current_image_path)  # Reset working copy
                self.truly_original_pixmap = QPixmap(self.current_image_path)
                self.display_image(self.original_pixmap)
                
                # Clear any extracted face data since we're starting fresh
                self.extracted_face_pixmap = None
                self.original_extracted_face_pixmap = None
                self.extracted_face_quality = None
                self.recognition_result = None
                
                # Clear face display
                if hasattr(self, 'face_image_label'):
                    self.face_image_label.clear()
                    self.face_image_label.setText("No face extracted")
                
                # Disable face recognition button since face is cleared
                if hasattr(self, 'recognize_face_btn'):
                    self.recognize_face_btn.setEnabled(False)
                
                # Re-analyze the fresh image
                self.analyze_current_image()
                
                self.status_bar.showMessage("Reset to original image (reloaded from file)")
                
            except Exception as e:
                self.status_bar.showMessage(f"Error reloading image: {str(e)}")
        else:
            # Fallback to in-memory reset if no file path
            if self.truly_original_pixmap:
                self.original_pixmap = QPixmap(self.truly_original_pixmap)  # Make a copy
                self.current_working_pixmap = QPixmap(self.truly_original_pixmap)  # Reset working copy
                self.display_image(self.original_pixmap)
                self.status_bar.showMessage("Reset to original image (all filters removed)")
                
            # Also reset extracted face to original if it exists
            if self.original_extracted_face_pixmap:
                self.extracted_face_pixmap = QPixmap(self.original_extracted_face_pixmap)  # Reset to original extracted face
                self.display_face_image(self.extracted_face_pixmap)  # Update face display
            else:
                # Clear face display if no extracted face
                if hasattr(self, 'face_image_label'):
                    self.face_image_label.clear()
                    self.face_image_label.setText("No face extracted")
    
    def auto_extract_face(self):
        """Automatically extract face from loaded image with anti-spoofing."""
        if not self.current_image_path:
            return
            
        try:
            self.status_bar.showMessage("Extracting faces with anti-spoofing...")
            
            # Extract faces directly from the original image file with anti-spoofing enabled
            faces_data = self.face_api.extract_faces_from_server(self.current_image_path, liveness_check=True)
            
            if not faces_data:
                # No faces detected - reset everything to prevent useless enhancements
                self.status_bar.showMessage("No faces detected - resetting to original image")
                self.extracted_face_pixmap = None
                self.original_extracted_face_pixmap = None
                self.extracted_face_quality = None
                self.recognition_result = None
                
                # Clear face display
                if hasattr(self, 'face_image_label'):
                    self.face_image_label.clear()
                    self.face_image_label.setText("No face extracted")
                
                # Clear face quality display
                if hasattr(self, 'face_quality_text'):
                    self.face_quality_text.setPlainText("No face extracted yet")
                
                # Disable face recognition button
                if hasattr(self, 'recognize_face_btn'):
                    self.recognize_face_btn.setEnabled(False)
                    
                # Show message to user
                QMessageBox.information(self, "No Faces Detected", 
                    "No faces were detected with anti-spoofing. "
                    "Face enhancement features will be disabled for this image.")
                return
                
            # Process faces and get the largest one
            face_image, quality_info = self.face_api.get_largest_face_with_quality(faces_data)
            
            if face_image is not None:
                # Convert OpenCV image to QPixmap
                height, width, channel = face_image.shape
                bytes_per_line = 3 * width
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                
                # Store original extracted face (without enhancements)
                self.original_extracted_face_pixmap = QPixmap.fromImage(q_image)
                self.extracted_face_pixmap = QPixmap.fromImage(q_image)  # Current state
                self.extracted_face_quality = quality_info
                
                # Display the extracted face
                self.display_face_image(self.extracted_face_pixmap)
                
                # Update face quality display
                self.update_face_quality_display(quality_info)
                
                # Enable face recognition button
                if hasattr(self, 'recognize_face_btn'):
                    self.recognize_face_btn.setEnabled(True)
                
                # Show quality info in status
                if quality_info:
                    quality_score = quality_info.get('quality_score', 'Unknown')
                    self.status_bar.showMessage(f"Face extracted successfully - Quality: {quality_score}")
                else:
                    self.status_bar.showMessage("Face extracted successfully")
                    
            else:
                self.status_bar.showMessage("Face extraction failed")
                QMessageBox.warning(self, "Face Extraction Failed", 
                    "Could not extract face from the image.")
                    
        except Exception as e:
            self.status_bar.showMessage(f"Face extraction error: {str(e)}")
            QMessageBox.critical(self, "Face Extraction Error", f"Error during face extraction:\n{str(e)}")
    
    def on_brightness_changed(self, value):
        """Handle brightness slider change."""
        self.brightness_label.setText(str(value))
        self.apply_enhancements()
        self.save_current_values_to_config()
    
    def on_contrast_changed(self, value):
        """Handle contrast slider change."""
        self.contrast_label.setText(str(value))
        self.apply_enhancements()
        self.save_current_values_to_config()
    
    def on_saturation_changed(self, value):
        """Handle saturation slider change."""
        self.saturation_label.setText(str(value))
        self.apply_enhancements()
        self.save_current_values_to_config()
    
    def apply_grayscale(self):
        """Convert the current working image to grayscale."""
        if not self.current_working_pixmap:
            return
            
        try:
            # Convert main image to grayscale
            grayscale_pixmap = self.convert_to_enhanced_grayscale(self.current_working_pixmap)
            if not grayscale_pixmap:
                raise Exception("Failed to convert main image to grayscale")
            
            # Update the current working image
            self.current_working_pixmap = grayscale_pixmap
            
            # Also apply grayscale to extracted face if it exists
            if self.original_extracted_face_pixmap:
                face_grayscale_pixmap = self.convert_to_enhanced_grayscale(self.original_extracted_face_pixmap)
                if face_grayscale_pixmap:
                    # Update both original and current extracted face
                    self.original_extracted_face_pixmap = face_grayscale_pixmap
                    self.extracted_face_pixmap = face_grayscale_pixmap
                    # Update face display
                    self.display_face_image(self.extracted_face_pixmap)
            
            # Reset enhancement sliders since filter creates new base
            self.brightness_slider.setValue(0)
            self.contrast_slider.setValue(0)
            self.saturation_slider.setValue(0)
            
            self.display_image(grayscale_pixmap)
            
            self.status_bar.showMessage("Converted image to grayscale")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to convert to grayscale:\n{str(e)}")
    
    def apply_enhancements(self):
        """Apply current enhancement values to the image and extracted face."""
        if not self.original_pixmap:
            return
        
        try:
            # Get current slider values
            brightness = self.brightness_slider.value()
            contrast = self.contrast_slider.value()
            saturation = self.saturation_slider.value()
            
            # Apply enhancements to current working image
            enhanced_pixmap = self.image_processor.enhance_image(
                self.current_working_pixmap,
                brightness=brightness,
                contrast=contrast,
                saturation=saturation
            )
            
            self.display_image(enhanced_pixmap)
            
            # Also apply enhancements to extracted face if it exists
            if self.original_extracted_face_pixmap:
                self.extracted_face_pixmap = self.image_processor.enhance_image(
                    self.original_extracted_face_pixmap,
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation
                )
                # Update face display with enhanced version
                self.display_face_image(self.extracted_face_pixmap)
            
        except Exception as e:
            self.status_bar.showMessage(f"Enhancement error: {str(e)}")
    
    def convert_to_enhanced_grayscale(self, pixmap):
        """Convert a QPixmap to enhanced grayscale using LAB luminance and noise reduction."""
        if not pixmap:
            return None
            
        try:
            # Convert QPixmap to OpenCV format
            q_image = pixmap.toImage()
            q_image = q_image.convertToFormat(QImage.Format.Format_RGB888)
            
            width = q_image.width()
            height = q_image.height()
            bytes_per_line = q_image.bytesPerLine()
            
            # Convert QImage to numpy array with proper dimensions
            ptr = q_image.bits()
            total_bytes = height * bytes_per_line
            arr = np.array(ptr, dtype=np.uint8, copy=True)[:total_bytes]
            arr = arr.reshape(height, bytes_per_line)
            
            # Extract only the RGB channels (ignore padding if any)
            rgb_arr = arr[:, :width*3].reshape(height, width, 3)
            
            # Convert RGB to BGR for OpenCV
            bgr_image = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
            
            # Enhanced natural conversion with noise reduction
            # First apply gentle noise reduction to color image
            denoised = cv2.bilateralFilter(bgr_image, 5, 40, 40)
            
            # Use LAB luminance for most natural look
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            # Apply gentle contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(4, 4))
            l_enhanced = clahe.apply(l_channel)

            # Apply grayscale conversion
            gray = cv2.cvtColor(l_enhanced, cv2.COLOR_BGR2GRAY) if len(l_enhanced.shape) == 3 else l_enhanced
            
            # Convert back to RGB for Qt
            rgb_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            
            # Convert back to QPixmap
            height, width, channel = rgb_image.shape
            bytes_per_line_new = 3 * width
            q_image_new = QImage(rgb_image.data, width, height, bytes_per_line_new, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(q_image_new)
            
        except Exception as e:
            print(f"Error in grayscale conversion: {e}")
            return None
    
    def on_blur_kernel_changed(self, value):
        """Handle blur kernel size slider change."""
        # Convert slider value (1-15) to odd kernel size (3-31)
        kernel_size = value * 2 + 1
        self.blur_kernel_label.setText(str(kernel_size))
    
    def on_blur_sigmax_changed(self, value):
        """Handle blur sigma X slider change."""
        if value == 0:
            self.blur_sigmax_label.setText("Auto")
        else:
            sigma_x = value / 10.0  # Convert to 0.1-10.0 range
            self.blur_sigmax_label.setText(f"{sigma_x:.1f}")
    
    def on_blur_sigmay_changed(self, value):
        """Handle blur sigma Y slider change."""
        if value == 0:
            self.blur_sigmay_label.setText("Auto")
        else:
            sigma_y = value / 10.0  # Convert to 0.1-10.0 range
            self.blur_sigmay_label.setText(f"{sigma_y:.1f}")
    
    def on_bilateral_d_changed(self, value):
        """Handle bilateral filter diameter slider change."""
        self.bilateral_d_label.setText(str(value))
        self.save_current_values_to_config()
    
    def on_bilateral_color_changed(self, value):
        """Handle bilateral filter color sigma slider change."""
        self.bilateral_color_label.setText(str(value))
        self.save_current_values_to_config()
    
    def on_bilateral_space_changed(self, value):
        """Handle bilateral filter space sigma slider change."""
        self.bilateral_space_label.setText(str(value))
        self.save_current_values_to_config()
    
    def on_denoise_h_changed(self, value):
        """Handle denoise h (luminance) slider change."""
        self.denoise_h_label.setText(str(value))
        self.save_current_values_to_config()
    
    def on_denoise_hcolor_changed(self, value):
        """Handle denoise hColor slider change."""
        self.denoise_hcolor_label.setText(str(value))
        self.save_current_values_to_config()
    
    def on_denoise_template_changed(self, value):
        """Handle denoise template window size slider change."""
        # Convert slider value (1-4) to template window size (3,5,7,9)
        template_size = value * 2 + 1
        self.denoise_template_label.setText(str(template_size))
        self.save_current_values_to_config()
    
    def on_denoise_search_changed(self, value):
        """Handle denoise search window size slider change."""
        # Convert slider value (1-6) to search window size (15,17,19,21,23,25)
        search_size = value * 2 + 13
        self.denoise_search_label.setText(str(search_size))
        self.save_current_values_to_config()
    
    def apply_noise_reduction(self):
        """Apply Non-local Means Denoising with current slider settings."""
        if not self.current_working_pixmap:
            return
        
        try:
            # Get slider values
            h = self.denoise_h_slider.value()
            h_color = self.denoise_hcolor_slider.value()
            template_size = self.denoise_template_slider.value() * 2 + 1
            search_size = self.denoise_search_slider.value() * 2 + 13
            
            # Apply noise reduction to current working image
            denoised_pixmap = self.image_processor.apply_fastNlMeansDenoising(
                self.current_working_pixmap, h, h_color, template_size, search_size
            )
            
            # Update the current working image
            self.current_working_pixmap = denoised_pixmap
            
            # Also apply noise reduction to extracted face if it exists
            if self.original_extracted_face_pixmap:
                face_denoised_pixmap = self.image_processor.apply_fastNlMeansDenoising(
                    self.original_extracted_face_pixmap, h, h_color, template_size, search_size
                )
                # Update the original extracted face to the filtered version
                self.original_extracted_face_pixmap = face_denoised_pixmap
                self.extracted_face_pixmap = face_denoised_pixmap
                # Update face display
                self.display_face_image(self.extracted_face_pixmap)
            
            # Reset enhancement sliders since filter creates new base
            self.brightness_slider.setValue(0)
            self.contrast_slider.setValue(0)
            self.saturation_slider.setValue(0)
            
            self.display_image(denoised_pixmap)
            
            self.status_bar.showMessage(f"Applied noise reduction (h: {h}, hColor: {h_color}, template: {template_size}, search: {search_size})")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply noise reduction:\n{str(e)}")
    
    def apply_gaussian_blur(self):
        """Apply Gaussian blur with current slider settings."""
        if not self.current_working_pixmap:
            return
        
        try:
            # Get slider values
            kernel_size = self.blur_kernel_slider.value() * 2 + 1
            sigma_x = 0 if self.blur_sigmax_slider.value() == 0 else self.blur_sigmax_slider.value() / 10.0
            sigma_y = 0 if self.blur_sigmay_slider.value() == 0 else self.blur_sigmay_slider.value() / 10.0
            
            # Apply Gaussian blur to current working image
            blurred_pixmap = self.image_processor.apply_gaussian_blur(
                self.current_working_pixmap, kernel_size, sigma_x, sigma_y
            )
            
            # Update the current working image
            self.current_working_pixmap = blurred_pixmap
            
            # Also apply gaussian blur to extracted face if it exists
            if self.original_extracted_face_pixmap:
                face_blurred_pixmap = self.image_processor.apply_gaussian_blur(
                    self.original_extracted_face_pixmap, kernel_size, sigma_x, sigma_y
                )
                # Update the original extracted face to the filtered version
                self.original_extracted_face_pixmap = face_blurred_pixmap
                self.extracted_face_pixmap = face_blurred_pixmap
                # Update face display
                self.display_face_image(self.extracted_face_pixmap)
            
            # Reset enhancement sliders since filter creates new base
            self.brightness_slider.setValue(0)
            self.contrast_slider.setValue(0)
            self.saturation_slider.setValue(0)
            
            self.display_image(blurred_pixmap)
            
            self.status_bar.showMessage(f"Applied Gaussian blur (kernel: {kernel_size}, σx: {sigma_x:.1f}, σy: {sigma_y:.1f})")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply Gaussian blur:\n{str(e)}")
    
    def apply_bilateral_filter(self):
        """Apply bilateral filter with current slider settings."""
        if not self.current_working_pixmap:
            return
        
        try:
            # Get slider values
            d = self.bilateral_d_slider.value()
            sigma_color = self.bilateral_color_slider.value()
            sigma_space = self.bilateral_space_slider.value()
            
            # Apply bilateral filter to current working image
            filtered_pixmap = self.image_processor.apply_bilateral_filter(
                self.current_working_pixmap, d, sigma_color, sigma_space
            )
            
            # Update the current working image
            self.current_working_pixmap = filtered_pixmap
            
            # Also apply bilateral filter to extracted face if it exists
            if self.original_extracted_face_pixmap:
                face_filtered_pixmap = self.image_processor.apply_bilateral_filter(
                    self.original_extracted_face_pixmap, d, sigma_color, sigma_space
                )
                # Update the original extracted face to the filtered version
                self.original_extracted_face_pixmap = face_filtered_pixmap
                self.extracted_face_pixmap = face_filtered_pixmap
                # Update face display
                self.display_face_image(self.extracted_face_pixmap)
            
            # Reset enhancement sliders since filter creates new base
            self.brightness_slider.setValue(0)
            self.contrast_slider.setValue(0)
            self.saturation_slider.setValue(0)
            
            self.display_image(filtered_pixmap)
            
            self.status_bar.showMessage(f"Applied bilateral filter (d: {d}, σColor: {sigma_color}, σSpace: {sigma_space})")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply bilateral filter:\n{str(e)}")
    
    def apply_filter(self, filter_type):
        """Apply OpenCV filter to the current image."""
        if not self.current_working_pixmap:
            return
        
        try:
            # Apply filter to current working image
            filtered_pixmap = self.image_processor.apply_filter(self.current_working_pixmap, filter_type)
            
            # Update the current working image
            self.current_working_pixmap = filtered_pixmap
            
            # Also apply filter to extracted face if it exists
            if self.original_extracted_face_pixmap:
                face_filtered_pixmap = self.image_processor.apply_filter(
                    self.original_extracted_face_pixmap, filter_type
                )
                # Update the original extracted face to the filtered version
                self.original_extracted_face_pixmap = face_filtered_pixmap
                self.extracted_face_pixmap = face_filtered_pixmap
                # Update face display
                self.display_face_image(self.extracted_face_pixmap)
            
            # Reset enhancement sliders since filter creates new base
            self.brightness_slider.setValue(0)
            self.contrast_slider.setValue(0)
            self.saturation_slider.setValue(0)
            
            self.display_image(filtered_pixmap)
            
            self.status_bar.showMessage(f"Applied {filter_type} filter")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply {filter_type} filter:\n{str(e)}")
    
    def apply_histogram_equalization(self):
        """Apply histogram equalization for auto contrast."""
        if not self.current_working_pixmap:
            return
        
        try:
            # Apply histogram equalization to current working image
            enhanced_pixmap = self.image_processor.apply_histogram_equalization(self.current_working_pixmap)
            
            # Update the current working image
            self.current_working_pixmap = enhanced_pixmap
            
            # Also apply histogram equalization to extracted face if it exists
            if self.original_extracted_face_pixmap:
                face_enhanced_pixmap = self.image_processor.apply_histogram_equalization(
                    self.original_extracted_face_pixmap
                )
                # Update the original extracted face to the enhanced version
                self.original_extracted_face_pixmap = face_enhanced_pixmap
                self.extracted_face_pixmap = face_enhanced_pixmap
                # Update face display
                self.display_face_image(self.extracted_face_pixmap)
            
            # Reset enhancement sliders since filter creates new base
            self.brightness_slider.setValue(0)
            self.contrast_slider.setValue(0)
            self.saturation_slider.setValue(0)
            
            self.display_image(enhanced_pixmap)
            
            self.status_bar.showMessage("Applied automatic contrast enhancement")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply histogram equalization:\n{str(e)}")
    
    def analyze_current_image(self):
        """Analyze the currently displayed image and update the display."""
        if not self.current_working_pixmap:
            return
            
        try:
            self.status_bar.showMessage("Analyzing current image...")
            
            # Get the full-resolution current image by applying current enhancements
            # to the current working pixmap (which includes all filters)
            brightness = self.brightness_slider.value()
            contrast = self.contrast_slider.value()
            saturation = self.saturation_slider.value()
            
            # Create full-resolution enhanced image
            if brightness == 0 and contrast == 0 and saturation == 0:
                # No enhancements, use current working image as-is
                current_full_res_pixmap = self.current_working_pixmap
            else:
                # Apply enhancements to current working image (with filters)
                current_full_res_pixmap = self.image_processor.enhance_image(
                    self.current_working_pixmap,
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation
                )
            
            # Analyze the full-resolution current image using ImageProcessor
            analysis = self.image_processor.analyze_image(current_full_res_pixmap)
            self.current_analysis = analysis
            
            # Update the analysis display
            self.update_analysis_display(analysis)
            
            if analysis:
                score = analysis.get('overall_quality_score', 0.0)
                info = analysis.get('image_info', {})
                dims = info.get('dimensions', (0, 0))
                self.status_bar.showMessage(f"Analysis complete - Quality: {score:.2f} (Full resolution: {dims[0]}x{dims[1]})")
            else:
                self.status_bar.showMessage("Analysis completed - No data available")
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            self.status_bar.showMessage("Analysis failed")
    
    def resizeEvent(self, event):
        """Handle window resize to update image display."""
        super().resizeEvent(event)
        if self.image_label.pixmap():
            # Re-display current image with new size
            current_pixmap = self.original_pixmap if self.original_pixmap else self.image_label.pixmap()
            self.display_image(current_pixmap)
        # Reposition overlay after resize
        if hasattr(self, 'analysis_overlay'):
            self.position_analysis_overlay()
    
    def extract_face(self):
        """Extract face from current enhanced image using face recognition API."""
        # Check if we have an image displayed
        current_pixmap = self.image_label.pixmap()
        if not current_pixmap:
            QMessageBox.warning(self, "Warning", "Please load an image first")
            return
            
        try:
            self.status_bar.showMessage("Extracting face from enhanced image...")
            
            # Save current enhanced image to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
                current_pixmap.save(temp_path, 'JPEG', quality=95)
            
            # Extract faces from the temporary enhanced image
            faces_data = self.face_api.extract_faces_from_server(temp_path)
            
            # Clean up temporary file
            import os
            try:
                os.unlink(temp_path)
            except:
                pass  # Ignore cleanup errors
            
            if not faces_data:
                QMessageBox.information(self, "No Faces", "No faces detected in the enhanced image")
                self.status_bar.showMessage("No faces detected")
                return
                
            # Process faces and get the largest one
            face_image, quality_info = self.face_api.get_largest_face_with_quality(faces_data)
            
            if face_image is not None:
                # Convert OpenCV image to QPixmap
                height, width, channel = face_image.shape
                bytes_per_line = 3 * width
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                
                # Store original extracted face (without enhancements)
                self.original_extracted_face_pixmap = QPixmap.fromImage(q_image)
                
                # Apply current enhancement settings to the extracted face
                brightness = self.brightness_slider.value()
                contrast = self.contrast_slider.value()
                saturation = self.saturation_slider.value()
                
                self.extracted_face_pixmap = self.image_processor.enhance_image(
                    self.original_extracted_face_pixmap,
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation
                )
                
                # Store quality info
                self.extracted_face_quality = quality_info
                
                # Display extracted face in the face panel (not main image area)
                self.display_face_image(self.extracted_face_pixmap)
                self.update_face_quality_display(quality_info)
                
                # Enable recognition button
                self.recognize_face_btn.setEnabled(True)
                
                self.status_bar.showMessage(f"Face extracted successfully")
            else:
                QMessageBox.warning(self, "Error", "Failed to process extracted face")
                self.status_bar.showMessage("Face extraction failed")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Face extraction failed:\n{str(e)}")
            self.status_bar.showMessage("Face extraction error")
    
    def recognize_face(self):
        """Recognize the current face image."""
        if not self.extracted_face_pixmap:
            QMessageBox.warning(self, "Warning", "Please extract a face first")
            return
            
        try:
            self.status_bar.showMessage("Recognizing face...")
            
            # Save current face to temporary file for recognition
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
                self.extracted_face_pixmap.save(temp_path, 'JPEG')
                
            # Perform recognition
            result = self.face_api.face_recognition(temp_path, "GUI Recognition")
            
            # Clean up temp file
            try:
                os.remove(temp_path)
            except:
                pass
                
            # Store and display result
            self.recognition_result = result
            self.update_recognition_result_display(result)
            
            if result.get('success'):
                person_name = result.get('person_name', 'Unknown')
                confidence = result.get('confidence', 0)
                self.status_bar.showMessage(f"Recognized: {person_name} ({confidence:.3f})")
            else:
                self.status_bar.showMessage("Face not recognized")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Face recognition failed:\n{str(e)}")
            self.status_bar.showMessage("Recognition error")
    
    def update_face_quality_display(self, quality_info):
        """Update face quality display with quality information."""
        if not quality_info:
            self.face_quality_text.setPlainText("No quality information available")
            return
            
        text_parts = []
        
        # Anti-spoofing information (display first as it's critical)
        if 'is_real' in quality_info:
            is_real = quality_info.get('is_real', True)
            antispoof_score = quality_info.get('antispoof_score', 1.0)
            anti_spoofing_status = quality_info.get('anti_spoofing_status', 'UNKNOWN')
            
            status_icon = "✅" if is_real else "❌"
            text_parts.append(f"{status_icon} Anti-Spoofing: {anti_spoofing_status}")
            text_parts.append(f"Spoof Score: {antispoof_score:.3f}")
            
            # Face detection confidence if available
            if 'face_confidence' in quality_info:
                face_confidence = quality_info.get('face_confidence', 0.0)
                text_parts.append(f"Face Confidence: {face_confidence:.3f}")
        
        # Quality score and status
        score = quality_info.get('quality_score', 0.0)
        quality_status = "GOOD" if score >= 0.7 else "POOR" if score < 0.4 else "FAIR"
        text_parts.append(f"Quality: {quality_status} ({score:.2f})")
        
        # Lighting quality
        lighting_quality = quality_info.get('lighting_quality', 'unknown')
        text_parts.append(f"Lighting: {lighting_quality.upper()}")
        
        # Metrics
        brightness = quality_info.get('brightness', 0)
        contrast = quality_info.get('contrast', 0)
        text_parts.append(f"Brightness: {brightness:.1f}")
        text_parts.append(f"Contrast: {contrast:.1f}")
        
        # Issues
        issues = quality_info.get('detected_issues', [])
        if issues:
            text_parts.append(f"Issues: {', '.join(issues)}")
        else:
            text_parts.append("No major issues detected")
            
        # Enhancement recommendation
        needs_enhancement = quality_info.get('needs_enhancement', False)
        text_parts.append(f"Enhancement needed: {'Yes' if needs_enhancement else 'No'}")
        
        self.face_quality_text.setPlainText('\n'.join(text_parts))
    
    def update_recognition_result_display(self, result):
        """Update recognition result display."""
        if not result:
            self.recognition_result_text.setPlainText("No recognition result available")
            return
            
        text_parts = []
        
        if result.get('success'):
            person_name = result.get('person_name', 'Unknown')
            confidence = result.get('confidence', 0)
            distance = result.get('distance', 0)
            processing_time = result.get('processing_time_ms', 0)
            
            text_parts.append(f"✅ RECOGNIZED")
            text_parts.append(f"Person: {person_name}")
            text_parts.append(f"Confidence: {confidence:.3f}")
            text_parts.append(f"Distance: {distance:.3f}")
            text_parts.append(f"Time: {processing_time}ms")
        else:
            error = result.get('error', '')
            message = result.get('message', 'Recognition failed')
            processing_time = result.get('processing_time_ms', 0)
            
            text_parts.append(f"❌ NOT RECOGNIZED")
            text_parts.append(f"Message: {message}")
            if error:
                text_parts.append(f"Error: {error}")
            text_parts.append(f"Time: {processing_time}ms")
            
        self.recognition_result_text.setPlainText('\n'.join(text_parts))