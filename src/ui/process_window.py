"""
Process Window for batch face detection processing
"""

import os
from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
    QFileDialog, QLineEdit, QLabel, QTableWidget, 
    QTableWidgetItem, QMessageBox, QProgressBar,
    QHeaderView, QAbstractItemView
)
from qtpy.QtCore import Qt, QThread, Signal, QDir, QSize
from qtpy.QtGui import QDoubleValidator, QPixmap, QIcon
from src.core.face_recognition_api import FaceRecognitionAPI
import cv2
import numpy as np
from pathlib import Path


class ProcessWorker(QThread):
    """Worker thread for processing images in the background."""
    
    progress_updated = Signal(int)
    result_added = Signal(str, str, str, str, float)  # path, classification, antispoof_info, match_status, confidence
    finished = Signal()
    error_occurred = Signal(str)
    
    def __init__(self, directory_path, threshold):
        super().__init__()
        self.directory_path = directory_path
        self.threshold = threshold
        self.face_api = FaceRecognitionAPI()
        self.should_stop = False
    
    def stop(self):
        """Stop the processing."""
        self.should_stop = True
    
    def run(self):
        """Run the face detection process."""
        try:
            # Get all image files from directory
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
            image_files = []
            
            directory = Path(self.directory_path)
            for file_path in directory.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    image_files.append(file_path)
            
            total_files = len(image_files)
            if total_files == 0:
                self.error_occurred.emit("No image files found in the selected directory.")
                return
            
            for i, image_path in enumerate(image_files):
                if self.should_stop:
                    break
                
                try:
                    # Extract faces using server API
                    faces = self.face_api.extract_faces_from_server(str(image_path))
                    
                    if faces is None:
                        # If extraction failed, set defaults
                        valid_faces = 0
                        max_confidence = 0.0
                        antispoof_info = "Failed"
                    else:
                        # Count faces with confidence above threshold
                        valid_faces = 0
                        max_confidence = 0.0
                        antispoof_info = "No Data"
                        
                        for face in faces:
                            confidence = face.get('confidence', 0.0)
                            if confidence >= self.threshold:
                                valid_faces += 1
                            if confidence > max_confidence:
                                max_confidence = confidence
                        
                        # Extract anti-spoofing information from first face (highest priority)
                        if faces:
                            first_face = faces[0]
                            antispoof_status = first_face.get('is_real', 'FAKE') and 'REAL' or 'FAKE'
                            antispoof_score = first_face.get('antispoof_score', 0.0)
                            is_real = first_face.get('is_real', False)
                            
                            # Format anti-spoofing info: status + score
                            status_icon = "✅" if is_real else "❌"
                            antispoof_info = f"{status_icon} {antispoof_status} ({antispoof_score:.3f})"
                    
                    # Classify image based on filename
                    filename = image_path.name.lower()
                    if filename.startswith('fake'):
                        classification = 'FAKE'
                    else:
                        classification = 'REAL'
                    
                    # Calculate match status between filename classification and anti-spoofing
                    if faces is None or not faces:
                        match_status = "No Face"
                    else:
                        first_face = faces[0]
                        is_real = first_face.get('is_real', False)
                        antispoof_result = 'REAL' if is_real else 'FAKE'
                        
                        if classification == antispoof_result:
                            match_status = "✅ Match"
                        else:
                            match_status = f"❌ {classification} vs {antispoof_result}"
                    
                    # Emit result
                    self.result_added.emit(str(image_path), classification, antispoof_info, match_status, max_confidence)
                    
                    # Update progress
                    progress = int((i + 1) * 100 / total_files)
                    self.progress_updated.emit(progress)
                    
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue
            
            self.finished.emit()
            
        except Exception as e:
            self.error_occurred.emit(f"Processing error: {str(e)}")


class ProcessWindow(QDialog):
    """Window for batch processing face detection on image directories."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Face Detection Processing")
        self.setMinimumSize(800, 600)
        self.resize(1000, 700)
        
        # Initialize variables
        self.selected_directory = ""
        self.worker_thread = None
        
        # Statistics tracking
        self.total_processed = 0
        self.correct_matches = 0
        self.incorrect_matches = 0
        self.no_face_detected = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Top controls layout
        top_layout = QHBoxLayout()
        
        # Directory selection button
        self.dir_button = QPushButton("Select Directory")
        self.dir_button.clicked.connect(self.select_directory)
        top_layout.addWidget(self.dir_button)
        
        # Threshold input
        threshold_label = QLabel("Confidence Threshold:")
        top_layout.addWidget(threshold_label)
        
        self.threshold_input = QLineEdit()
        self.threshold_input.setText("0.5")
        # Set validator to accept decimal numbers between 0 and 1
        validator = QDoubleValidator(0.0, 1.0, 2)
        validator.setNotation(QDoubleValidator.StandardNotation)
        self.threshold_input.setValidator(validator)
        self.threshold_input.setMaximumWidth(100)
        top_layout.addWidget(self.threshold_input)
        
        # Process button
        self.process_button = QPushButton("Process Face Detection")
        self.process_button.clicked.connect(self.start_processing)
        self.process_button.setEnabled(False)
        top_layout.addWidget(self.process_button)
        
        # Stop button
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        top_layout.addWidget(self.stop_button)
        
        # Add stretch to push controls to left
        top_layout.addStretch()
        
        layout.addLayout(top_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Ready - %p%")
        layout.addWidget(self.progress_bar)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(["Thumbnail", "Classification", "Anti-Spoofing", "Match Status", "Max Confidence"])
        
        # Configure table
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)  # Thumbnail column fixed width
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Classification column
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Anti-spoofing column
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Match status column
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Confidence column stretches
        
        # Set thumbnail column width
        self.results_table.setColumnWidth(0, 250)
        
        # Set row height for thumbnails
        self.results_table.verticalHeader().setDefaultSectionSize(200)
        
        # Set icon size to match thumbnail size
        self.results_table.setIconSize(QSize(200, 200))
        
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        
        layout.addWidget(self.results_table)
        
        # Status label
        self.status_label = QLabel("Select a directory to begin processing")
        layout.addWidget(self.status_label)
    
    def select_directory(self):
        """Open file dialog to select directory."""
        directory = QFileDialog.getExistingDirectory(
            self, 
            "Select Directory Containing Images",
            QDir.homePath()
        )
        
        if directory:
            self.selected_directory = directory
            self.process_button.setEnabled(True)
            self.status_label.setText(f"Selected: {directory}")
            self.load_images_preview()
    
    def load_images_preview(self):
        """Load and display images from selected directory."""
        if not self.selected_directory:
            return
        
        # Clear previous results
        self.results_table.setRowCount(0)
        
        # Get all image files from directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        image_files = []
        
        directory = Path(self.selected_directory)
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
        
        self.status_label.setText(f"Loading {len(image_files)} images...")
        
        # Load images into table
        for image_path in image_files:
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            
            # Create thumbnail
            try:
                pixmap = QPixmap(str(image_path))
                if not pixmap.isNull():
                    # Scale to thumbnail size
                    thumbnail = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    
                    # Create thumbnail item
                    thumbnail_item = QTableWidgetItem()
                    thumbnail_item.setIcon(QIcon(thumbnail))
                    thumbnail_item.setData(Qt.UserRole, str(image_path))  # Store full path
                    thumbnail_item.setToolTip(str(image_path))
                    self.results_table.setItem(row, 0, thumbnail_item)
                    
            except Exception as e:
                # If image can't be loaded, just show filename
                thumbnail_item = QTableWidgetItem(image_path.name)
                thumbnail_item.setData(Qt.UserRole, str(image_path))
                thumbnail_item.setToolTip(str(image_path))
                self.results_table.setItem(row, 0, thumbnail_item)
            
            # Classify image based on filename
            filename = image_path.name.lower()
            if filename.startswith('fake'):
                classification = 'FAKE'
            else:
                classification = 'REAL'
            
            self.results_table.setItem(row, 1, QTableWidgetItem(classification))
            self.results_table.setItem(row, 2, QTableWidgetItem("-"))  # Anti-spoofing info - updated after processing
            self.results_table.setItem(row, 3, QTableWidgetItem("-"))  # Match status - updated after processing
            self.results_table.setItem(row, 4, QTableWidgetItem("-"))  # Confidence - updated after processing
        
        self.status_label.setText(f"Loaded {len(image_files)} images. Ready for face detection.")
    
    def start_processing(self):
        """Start the face detection processing."""
        if not self.selected_directory:
            QMessageBox.warning(self, "Warning", "Please select a directory first.")
            return
        
        try:
            threshold = float(self.threshold_input.text())
            if threshold < 0 or threshold > 1:
                QMessageBox.warning(self, "Warning", "Threshold must be between 0.0 and 1.0")
                return
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter a valid threshold value.")
            return
        
        # Check if images are already loaded
        if self.results_table.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "No images found in the selected directory.")
            return
        
        # Reset statistics
        self.total_processed = 0
        self.correct_matches = 0
        self.incorrect_matches = 0
        self.no_face_detected = 0
        
        # Set up UI for processing
        self.process_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.dir_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Processing - %p%")
        self.status_label.setText("Processing images...")
        
        # Start worker thread
        self.worker_thread = ProcessWorker(self.selected_directory, threshold)
        self.worker_thread.progress_updated.connect(self.update_progress)
        self.worker_thread.result_added.connect(self.add_result)
        self.worker_thread.finished.connect(self.processing_finished)
        self.worker_thread.error_occurred.connect(self.processing_error)
        self.worker_thread.start()
    
    def stop_processing(self):
        """Stop the current processing."""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.status_label.setText("Stopping processing...")
    
    def update_progress(self, value):
        """Update progress bar."""
        self.progress_bar.setValue(value)
    
    def add_result(self, image_path, classification, antispoof_info, match_status, max_confidence):
        """Add a result to the table."""
        # Find the row for this image path
        target_row = -1
        for row in range(self.results_table.rowCount()):
            thumbnail_item = self.results_table.item(row, 0)
            if thumbnail_item and thumbnail_item.data(Qt.UserRole) == image_path:
                target_row = row
                break
        
        if target_row >= 0:
            # Update existing row with face detection results
            self.results_table.setItem(target_row, 2, QTableWidgetItem(antispoof_info))
            self.results_table.setItem(target_row, 3, QTableWidgetItem(match_status))
            self.results_table.setItem(target_row, 4, QTableWidgetItem(f"{max_confidence:.3f}"))
            
            # Update statistics
            self.total_processed += 1
            if match_status == "✅ Match":
                self.correct_matches += 1
            elif match_status == "No Face":
                self.no_face_detected += 1
            else:
                self.incorrect_matches += 1
        
        # Auto-scroll to bottom
        self.results_table.scrollToBottom()
    
    def processing_finished(self):
        """Handle processing completion."""
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Complete - 100%")
        
        # Calculate statistics
        total_with_faces = self.correct_matches + self.incorrect_matches
        
        if total_with_faces > 0:
            accuracy_percentage = (self.correct_matches / total_with_faces) * 100
        else:
            accuracy_percentage = 0
        
        # Create statistics message
        stats_message = (
            f"Processing Statistics:\n\n"
            f"Total Images Processed: {self.total_processed}\n"
            f"Images with Faces Detected: {total_with_faces}\n"
            f"No Face Detected: {self.no_face_detected}\n\n"
            f"Match Results (for images with faces):\n"
            f"✅ Correct Matches: {self.correct_matches}\n"
            f"❌ Incorrect Matches: {self.incorrect_matches}\n\n"
            f"Accuracy: {accuracy_percentage:.1f}% ({self.correct_matches}/{total_with_faces})"
        )
        
        # Show statistics in message box
        QMessageBox.information(self, "Processing Complete", stats_message)
        
        # Update status label with summary
        self.status_label.setText(
            f"Complete: {self.total_processed} processed, "
            f"{self.correct_matches} correct, {self.incorrect_matches} incorrect, "
            f"Accuracy: {accuracy_percentage:.1f}%"
        )
        
        self.reset_ui_state()
    
    def processing_error(self, error_message):
        """Handle processing error."""
        QMessageBox.critical(self, "Processing Error", error_message)
        self.status_label.setText("Processing failed.")
        self.reset_ui_state()
    
    def reset_ui_state(self):
        """Reset UI to initial state."""
        self.process_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.dir_button.setEnabled(True)
        self.progress_bar.setFormat("Ready - %p%")
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.worker_thread.wait()
        event.accept()