"""
Anti-Spoofing Analysis Result Dialog
"""

import json
from qtpy.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout
from qtpy.QtCore import Qt


class AntiSpoofingResultDialog(QDialog):
    """Dialog for displaying anti-spoofing analysis results."""
    
    def __init__(self, result, parent=None, x=100, y=100):
        """
        Initialize the dialog.
        
        Args:
            result: Dictionary containing anti-spoofing analysis results
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Anti-Spoofing Analysis Results")
        self.setGeometry(x, y, 600, 1200)
        self.result = result
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Create text display for results
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("""
            QTextEdit {
                background-color: #011324;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 10px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                color: #ffffff;
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
        
        # Format and display the results
        formatted_text = self._format_results(self.result)
        self.result_text.setPlainText(formatted_text)
        
        layout.addWidget(self.result_text)
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        
        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.clicked.connect(self.copy_to_clipboard)
        
        button_layout.addWidget(copy_btn)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def _format_results(self, result):
        """Format the results for display."""
        text_parts = []
        text_parts.append("=" * 80)
        text_parts.append("ANTI-SPOOFING ANALYSIS RESULTS")
        text_parts.append("=" * 80)
        text_parts.append("")
        
        # Overall success status
        success = result.get('success', False)
        status_str = "✅ SUCCESS" if success else "❌ FAILED"
        text_parts.append(f"Status: {status_str}")
        text_parts.append("")
        
        if not success:
            error = result.get('error', 'Unknown error')
            text_parts.append(f"Error: {error}")
            text_parts.append("")
            return "\n".join(text_parts)
        
        # Image Info
        if 'image_info' in result:
            text_parts.append("-" * 80)
            text_parts.append("IMAGE INFORMATION")
            text_parts.append("-" * 80)
            img_info = result['image_info']
            text_parts.append(f"  Width: {img_info.get('width', 'N/A')} px")
            text_parts.append(f"  Height: {img_info.get('height', 'N/A')} px")
            text_parts.append(f"  Format: {img_info.get('format', 'N/A')}")
            text_parts.append("")
        
        # DeepFace Results
        if 'deepface_results' in result:
            df = result['deepface_results']
            text_parts.append("-" * 80)
            text_parts.append("DEEPFACE RESULTS")
            text_parts.append("-" * 80)
            text_parts.append(f"  Faces Detected: {df.get('faces_detected', 0)}")
            
            if df.get('max_face'):
                mf = df['max_face']
                text_parts.append("  ")
                text_parts.append("  Max Face:")
                
                # Safely format confidence
                confidence = mf.get('confidence', 'N/A')
                if isinstance(confidence, (int, float)):
                    text_parts.append(f"    Confidence: {confidence:.4f}")
                else:
                    text_parts.append(f"    Confidence: {confidence}")
                
                # Format facial area if present
                if mf.get('facial_area'):
                    fa = mf['facial_area']
                    text_parts.append("    Facial Area:")
                    text_parts.append(f"      X: {fa.get('x', 'N/A')}")
                    text_parts.append(f"      Y: {fa.get('y', 'N/A')}")
                    text_parts.append(f"      W: {fa.get('w', 'N/A')}")
                    text_parts.append(f"      H: {fa.get('h', 'N/A')}")
            else:
                text_parts.append("  No face detected")
            text_parts.append("")
        
        # Composite Anti-Spoofing Results
        if 'composite_anti_spoofing' in result:
            cas = result['composite_anti_spoofing']
            text_parts.append("-" * 80)
            text_parts.append("COMPOSITE ANTI-SPOOFING RESULTS")
            text_parts.append("-" * 80)
            
            if cas:
                # Safely format scores
                photo_score = cas.get('photo_score', 'N/A')
                real_score = cas.get('real_score', 'N/A')
                video_score = cas.get('video_score', 'N/A')
                confidence = cas.get('confidence', 'N/A')
                
                if isinstance(photo_score, (int, float)):
                    text_parts.append(f"  Photo Score: {photo_score:.4f}")
                else:
                    text_parts.append(f"  Photo Score: {photo_score}")
                
                if isinstance(real_score, (int, float)):
                    text_parts.append(f"  Real Score: {real_score:.4f}")
                else:
                    text_parts.append(f"  Real Score: {real_score}")
                
                if isinstance(video_score, (int, float)):
                    text_parts.append(f"  Video Score: {video_score:.4f}")
                else:
                    text_parts.append(f"  Video Score: {video_score}")
                
                is_real = cas.get('is_real')
                verdict_str = "🟢 REAL" if is_real else "🔴 SPOOF"
                text_parts.append(f"  Final Verdict: {verdict_str}")
                
                if isinstance(confidence, (int, float)):
                    text_parts.append(f"  Confidence: {confidence:.4f}")
                else:
                    text_parts.append(f"  Confidence: {confidence}")
                
                # Debug info if available
                if cas.get('_debug'):
                    debug = cas['_debug']
                    text_parts.append("  ")
                    text_parts.append("  Debug Information:")
                    if debug.get('v2se_result'):
                        text_parts.append(f"    V2SE Result: {debug['v2se_result']}")
                    if debug.get('all_model_observations'):
                        text_parts.append("    All Model Observations:")
                        for key, val in debug['all_model_observations'].items():
                            text_parts.append(f"      {key}: {val}")
            else:
                text_parts.append("  No anti-spoofing results available")
            text_parts.append("")
        
        text_parts.append("=" * 80)
        text_parts.append("RAW JSON RESPONSE")
        text_parts.append("=" * 80)
        text_parts.append(json.dumps(self.result, indent=2))
        
        return "\n".join(text_parts)
    
    def copy_to_clipboard(self):
        """Copy the results to clipboard."""
        from qtpy.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(self.result_text.toPlainText())
        
        # Show a temporary message
        from qtpy.QtWidgets import QMessageBox
        QMessageBox.information(self, "Copied", "Results copied to clipboard!")
