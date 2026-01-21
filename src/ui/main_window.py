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
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QPixmap, QIcon, QPalette, QColor
from src.core.image_processor import ImageProcessor


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.image_processor = ImageProcessor()
        self.current_image_path = None
        self.original_pixmap = None
        self.truly_original_pixmap = None  # Store the unmodified original image
        self.current_analysis = None  # Store current image analysis
        
        self.init_ui()
        self.connect_signals()
    
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
        
        # Right panel for image display
        self.create_image_panel(main_layout)
        
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
        
        # Brightness control
        enhance_layout.addWidget(QLabel("Brightness:"), 0, 0)
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_label = QLabel("0")
        enhance_layout.addWidget(self.brightness_slider, 0, 1)
        enhance_layout.addWidget(self.brightness_label, 0, 2)
        
        # Contrast control
        enhance_layout.addWidget(QLabel("Contrast:"), 1, 0)
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(0)
        self.contrast_label = QLabel("0")
        enhance_layout.addWidget(self.contrast_slider, 1, 1)
        enhance_layout.addWidget(self.contrast_label, 1, 2)
        
        # Saturation control
        enhance_layout.addWidget(QLabel("Saturation:"), 2, 0)
        self.saturation_slider = QSlider(Qt.Orientation.Horizontal)
        self.saturation_slider.setRange(-100, 100)
        self.saturation_slider.setValue(0)
        self.saturation_label = QLabel("0")
        enhance_layout.addWidget(self.saturation_slider, 2, 1)
        enhance_layout.addWidget(self.saturation_label, 2, 2)
        
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
        parent_layout.addWidget(image_widget)
    
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
        
        # Connect OpenCV filter buttons
        self.apply_blur_btn.clicked.connect(self.apply_gaussian_blur)
        self.apply_denoise_btn.clicked.connect(self.apply_noise_reduction)
        self.sharpen_btn.clicked.connect(lambda: self.apply_filter('sharpen'))
        self.edge_btn.clicked.connect(lambda: self.apply_filter('edge'))
        self.emboss_btn.clicked.connect(lambda: self.apply_filter('emboss'))
        self.histogram_btn.clicked.connect(self.apply_histogram_equalization)
        
        # Connect Gaussian Blur sliders
        self.blur_kernel_slider.valueChanged.connect(self.on_blur_kernel_changed)
        self.blur_sigmax_slider.valueChanged.connect(self.on_blur_sigmax_changed)
        self.blur_sigmay_slider.valueChanged.connect(self.on_blur_sigmay_changed)
        
        # Connect Noise Reduction sliders
        self.denoise_h_slider.valueChanged.connect(self.on_denoise_h_changed)
        self.denoise_hcolor_slider.valueChanged.connect(self.on_denoise_hcolor_changed)
        self.denoise_template_slider.valueChanged.connect(self.on_denoise_template_changed)
        self.denoise_search_slider.valueChanged.connect(self.on_denoise_search_changed)
    
    def load_image(self):
        """Load an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image File",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)"
        )
        
        if file_path:
            try:
                self.current_image_path = file_path
                self.original_pixmap = QPixmap(file_path)
                self.truly_original_pixmap = QPixmap(file_path)  # Store the true original
                self.display_image(self.original_pixmap)
                
                # Enable controls
                self.save_btn.setEnabled(True)
                self.reset_btn.setEnabled(True)
                
                # Enable filter buttons
                self.apply_blur_btn.setEnabled(True)
                self.apply_denoise_btn.setEnabled(True)
                self.sharpen_btn.setEnabled(True)
                self.edge_btn.setEnabled(True)
                self.emboss_btn.setEnabled(True)
                self.histogram_btn.setEnabled(True)
                
                # Reset sliders
                self.reset_adjustments()
                
                # Auto-analyze the image
                self.analyze_current_image()
                
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
    
    def reset_adjustments(self):
        """Reset all adjustment sliders and restore original image (remove all filters)."""
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(0)
        self.saturation_slider.setValue(0)
        
        # Reset Gaussian Blur sliders
        self.blur_kernel_slider.setValue(7)  # Default: 15x15 kernel
        self.blur_sigmax_slider.setValue(0)  # Auto
        self.blur_sigmay_slider.setValue(0)  # Auto
        
        # Reset Noise Reduction sliders
        self.denoise_h_slider.setValue(10)  # Default: 10
        self.denoise_hcolor_slider.setValue(10)  # Default: 10
        self.denoise_template_slider.setValue(2)  # Default: 7
        self.denoise_search_slider.setValue(4)  # Default: 21
        
        # Restore the truly original image (before any filters)
        if self.truly_original_pixmap:
            self.original_pixmap = QPixmap(self.truly_original_pixmap)  # Make a copy
            self.display_image(self.original_pixmap)
            self.status_bar.showMessage("Reset to original image (all filters removed)")
    
    def on_brightness_changed(self, value):
        """Handle brightness slider change."""
        self.brightness_label.setText(str(value))
        self.apply_enhancements()
    
    def on_contrast_changed(self, value):
        """Handle contrast slider change."""
        self.contrast_label.setText(str(value))
        self.apply_enhancements()
    
    def on_saturation_changed(self, value):
        """Handle saturation slider change."""
        self.saturation_label.setText(str(value))
        self.apply_enhancements()
    
    def apply_enhancements(self):
        """Apply current enhancement values to the image."""
        if not self.original_pixmap:
            return
        
        try:
            # Get current slider values
            brightness = self.brightness_slider.value()
            contrast = self.contrast_slider.value()
            saturation = self.saturation_slider.value()
            
            # Apply enhancements using the image processor
            enhanced_pixmap = self.image_processor.enhance_image(
                self.original_pixmap,
                brightness=brightness,
                contrast=contrast,
                saturation=saturation
            )
            
            self.display_image(enhanced_pixmap)
            
        except Exception as e:
            self.status_bar.showMessage(f"Enhancement error: {str(e)}")
    
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
    
    def on_denoise_h_changed(self, value):
        """Handle denoise h (luminance) slider change."""
        self.denoise_h_label.setText(str(value))
    
    def on_denoise_hcolor_changed(self, value):
        """Handle denoise hColor slider change."""
        self.denoise_hcolor_label.setText(str(value))
    
    def on_denoise_template_changed(self, value):
        """Handle denoise template window size slider change."""
        # Convert slider value (1-4) to template window size (3,5,7,9)
        template_size = value * 2 + 1
        self.denoise_template_label.setText(str(template_size))
    
    def on_denoise_search_changed(self, value):
        """Handle denoise search window size slider change."""
        # Convert slider value (1-6) to search window size (15,17,19,21,23,25)
        search_size = value * 2 + 13
        self.denoise_search_label.setText(str(search_size))
    
    def apply_noise_reduction(self):
        """Apply Non-local Means Denoising with current slider settings."""
        if not self.original_pixmap:
            return
        
        try:
            # Get slider values
            h = self.denoise_h_slider.value()
            h_color = self.denoise_hcolor_slider.value()
            template_size = self.denoise_template_slider.value() * 2 + 1
            search_size = self.denoise_search_slider.value() * 2 + 13
            
            # Apply noise reduction
            denoised_pixmap = self.image_processor.apply_fastNlMeansDenoising(
                self.original_pixmap, h, h_color, template_size, search_size
            )
            
            # Update the working image
            self.original_pixmap = denoised_pixmap
            self.display_image(denoised_pixmap)
            
            # Reset enhancement sliders since we're working with a new base image
            self.brightness_slider.setValue(0)
            self.contrast_slider.setValue(0)
            self.saturation_slider.setValue(0)
            
            self.status_bar.showMessage(f"Applied noise reduction (h: {h}, hColor: {h_color}, template: {template_size}, search: {search_size})")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply noise reduction:\n{str(e)}")
    
    def apply_gaussian_blur(self):
        """Apply Gaussian blur with current slider settings."""
        if not self.original_pixmap:
            return
        
        try:
            # Get slider values
            kernel_size = self.blur_kernel_slider.value() * 2 + 1
            sigma_x = 0 if self.blur_sigmax_slider.value() == 0 else self.blur_sigmax_slider.value() / 10.0
            sigma_y = 0 if self.blur_sigmay_slider.value() == 0 else self.blur_sigmay_slider.value() / 10.0
            
            # Apply Gaussian blur with custom parameters
            blurred_pixmap = self.image_processor.apply_gaussian_blur(
                self.original_pixmap, kernel_size, sigma_x, sigma_y
            )
            
            # Update the working image (not the truly original)
            self.original_pixmap = blurred_pixmap
            self.display_image(blurred_pixmap)
            
            # Reset enhancement sliders since we're working with a new base image
            self.brightness_slider.setValue(0)
            self.contrast_slider.setValue(0)
            self.saturation_slider.setValue(0)
            
            self.status_bar.showMessage(f"Applied Gaussian blur (kernel: {kernel_size}, σx: {sigma_x:.1f}, σy: {sigma_y:.1f})")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply Gaussian blur:\n{str(e)}")
    
    def apply_filter(self, filter_type):
        """Apply OpenCV filter to the current image."""
        if not self.original_pixmap:
            return
        
        try:
            # Apply filter to current working image
            filtered_pixmap = self.image_processor.apply_filter(self.original_pixmap, filter_type)
            
            # Update the working image
            self.original_pixmap = filtered_pixmap
            self.display_image(filtered_pixmap)
            
            # Reset enhancement sliders since we're working with a new base image
            self.brightness_slider.setValue(0)
            self.contrast_slider.setValue(0)
            self.saturation_slider.setValue(0)
            
            self.status_bar.showMessage(f"Applied {filter_type} filter")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply {filter_type} filter:\n{str(e)}")
    
    def apply_histogram_equalization(self):
        """Apply histogram equalization for auto contrast."""
        if not self.original_pixmap:
            return
        
        try:
            # Apply histogram equalization to current working image
            enhanced_pixmap = self.image_processor.apply_histogram_equalization(self.original_pixmap)
            
            # Update the working image
            self.original_pixmap = enhanced_pixmap
            self.display_image(enhanced_pixmap)
            
            # Reset enhancement sliders since we're working with a new base image
            self.brightness_slider.setValue(0)
            self.contrast_slider.setValue(0)
            self.saturation_slider.setValue(0)
            
            self.status_bar.showMessage("Applied automatic contrast enhancement")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply histogram equalization:\n{str(e)}")
    
    def analyze_current_image(self):
        """Analyze the currently loaded image and update the display."""
        if not self.original_pixmap:
            return
            
        try:
            # Analyze the image using ImageProcessor
            analysis = self.image_processor.analyze_image(self.original_pixmap)
            self.current_analysis = analysis
            
            # Update the analysis display
            self.update_analysis_display(analysis)
            
            if analysis:
                score = analysis.get('overall_quality_score', 0.0)
                self.status_bar.showMessage(f"Analysis complete - Quality score: {score:.2f}")
            
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