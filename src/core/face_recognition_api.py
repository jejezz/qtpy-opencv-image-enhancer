"""
Face Recognition API Client

Provides methods to interact with the face recognition server:
1. Extract faces from images
2. Process extracted faces with quality analysis
3. Test face recognition

Server: https://callfusion.ptype.co.kr:55000
Authentication: Certificate-based (certs folder)
"""

import os
import sys
import base64
import requests
import cv2
import numpy as np
import urllib3
from typing import Dict, List, Optional, Any

# Add parent directory to path for ImageAnalyzer import
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from .image_processor import ImageAnalyzer
    HAS_IMAGE_ANALYZER = True
except ImportError:
    try:
        from image_analysis_toolkit import ImageAnalyzer, ImageCorrector, ImageToolkit
        HAS_IMAGE_ANALYZER = True
    except ImportError:
        HAS_IMAGE_ANALYZER = False
        print("⚠️ ImageAnalyzer not available - quality analysis will be limited")

# Disable SSL warnings for development
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class FaceRecognitionAPI:
    """Client for face recognition API operations"""
    
    def __init__(self, 
                 server_url: str = "https://callfusion.ptype.co.kr:55000",
                 certs_dir: str = "./certs"):
        """
        Initialize the API client.
        
        Args:
            server_url: Base URL of the face recognition server
            certs_dir: Directory containing certificate files
        """
        self.server_url = server_url
        self.api_base = f"{server_url}/api/v2"
        self.certs_dir = certs_dir
        
        # Setup SSL certificates if available
        self.verify_ssl = False
        self.cert = None
        self._setup_certificates()
        
        if HAS_IMAGE_ANALYZER:
            self.analyzer = ImageAnalyzer()
        else:
            self.analyzer = None
    
    def _setup_certificates(self):
        """Setup SSL certificates for authentication"""
        cert_file = os.path.join(self.certs_dir, "client.crt")
        key_file = os.path.join(self.certs_dir, "client.key")
        ca_file = os.path.join(self.certs_dir, "ca.crt")
        
        if os.path.exists(cert_file) and os.path.exists(key_file):
            self.cert = (cert_file, key_file)
            print(f"✅ Client certificates loaded from {self.certs_dir}")
            
            if os.path.exists(ca_file):
                self.verify_ssl = ca_file
                print(f"✅ CA certificate loaded for SSL verification")
            else:
                print("⚠️ CA certificate not found - using insecure SSL")
        else:
            print(f"⚠️ Client certificates not found in {self.certs_dir} - using insecure connection")

    def save_base64_image(self, base64_data: str, output_path: str) -> bool:
        """Save base64 image data to file"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_data:
                base64_data = base64_data.split(',')[1]
            
            image_bytes = base64.b64decode(base64_data)
            with open(output_path, 'wb') as f:
                f.write(image_bytes)
            print(f"💾 Saved face image: {output_path} ({len(image_bytes)} bytes)")
            return True
        except Exception as e:
            print(f"❌ Error saving image {output_path}: {e}")
            return False

    def analyze_face_quality(self, face_image_path: str) -> Optional[Dict[str, Any]]:
        """Analyze quality of an extracted face image"""
        if not self.analyzer:
            return self._perform_simple_quality_analysis(face_image_path)
        
        try:
            # Use ImageAnalyzer for comprehensive analysis
            result = self.analyzer.detect_all_issues(face_image_path)
            
            # Extract quality metrics
            overall_quality = result.get('overall_quality_score', 0.5)
            needs_enhancement = result.get('needs_correction', True)
            
            # Extract specific analysis results
            lighting = result.get('lighting', {})
            backlight = result.get('backlight', {})
            noise = result.get('noise', {})
            blur = result.get('blur', {})
            
            # Calculate brightness and contrast
            brightness = lighting.get('mean_brightness', 100)
            contrast = lighting.get('contrast', 35)
            
            # Determine lighting quality
            if overall_quality >= 0.8:
                lighting_quality = 'excellent'
            elif overall_quality >= 0.6:
                lighting_quality = 'good'
            elif overall_quality >= 0.4:
                lighting_quality = 'fair'
            else:
                lighting_quality = 'poor'
            
            # Collect detected issues
            detected_issues = []
            if lighting.get('is_low_light', False):
                detected_issues.append('Low light detected')
            if lighting.get('is_low_contrast', False):
                detected_issues.append('Low contrast detected')
            if backlight.get('has_backlight', False):
                detected_issues.append('Backlight detected')
            if noise.get('is_noisy', False):
                detected_issues.append('Noise detected')
            if blur.get('is_blurry', False):
                detected_issues.append('Blur detected')
            
            recommendations = result.get('recommendations', [])
            priority_issues = result.get('priority_issues', [])
            
            return {
                'quality_score': overall_quality,
                'needs_enhancement': needs_enhancement,
                'lighting_quality': lighting_quality,
                'detected_issues': detected_issues,
                'analysis_type': 'image_analyzer',
                'brightness': brightness,
                'contrast': contrast,
                'noise_level': noise.get('noise_level', 'moderate'),
                'laplacian_variance': blur.get('laplacian_variance', 100),
                'recommendations': recommendations + priority_issues,
                'full_analysis': result
            }
            
        except Exception as e:
            print(f"      ℹ️ Using fallback quality analysis due to: {e}")
            return self._perform_simple_quality_analysis(face_image_path)

    def _perform_simple_quality_analysis(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Simple fallback quality analysis using basic OpenCV operations"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            
            # Noise assessment
            if laplacian_var > 150:
                noise_level = 'very_low'
                noise_score = 0.9
            elif laplacian_var > 100:
                noise_level = 'low'
                noise_score = 0.7
            elif laplacian_var > 50:
                noise_level = 'moderate'
                noise_score = 0.5
            else:
                noise_level = 'high'
                noise_score = 0.3
            
            # Detect issues
            issues = []
            if brightness < 80:
                issues.append("Image appears dark")
            elif brightness > 200:
                issues.append("Image appears bright")
            if contrast < 30:
                issues.append("Low contrast detected")
            if laplacian_var < 50:
                issues.append("High noise/blur detected")
            
            # Quality score calculation
            quality_score = 1.0
            if brightness < 60 or brightness > 220:
                quality_score -= 0.3
            if contrast < 20:
                quality_score -= 0.2
            if laplacian_var < 50:
                quality_score -= 0.3
            
            quality_score = max(0.0, min(1.0, quality_score))
            
            # Determine lighting quality
            if quality_score >= 0.8:
                lighting_quality = 'excellent'
            elif quality_score >= 0.6:
                lighting_quality = 'good'
            elif quality_score >= 0.4:
                lighting_quality = 'fair'
            else:
                lighting_quality = 'poor'
            
            return {
                'quality_score': quality_score,
                'needs_enhancement': quality_score < 0.6,
                'lighting_quality': lighting_quality,
                'detected_issues': issues,
                'analysis_type': 'simple_fallback',
                'brightness': brightness,
                'contrast': contrast,
                'noise_level': noise_level,
                'laplacian_variance': laplacian_var,
                'recommendations': []
            }
            
        except Exception as e:
            print(f"      ❌ Fallback analysis failed: {e}")
            return None

    def extract_faces_from_server(self, image_path: str, liveness_check: bool = True) -> Optional[List[Dict]]:
        """
        Step 1: Send image to server to extract faces
        
        Args:
            image_path: Path to the input image
            liveness_check: Whether to perform liveness detection
            
        Returns:
            List of face data dictionaries or None if failed
        """
        print(f"🔍 Step 1: Extracting faces from {image_path}")
        
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {'liveness_check': str(liveness_check).lower()}
                
                response = requests.post(
                    f"{self.api_base}/extract_faces",
                    files=files,
                    data=data,
                    cert=self.cert,
                    verify=self.verify_ssl
                )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   📊 Debug - API response keys: {list(result.keys())}")
                
                if result.get('success'):
                    faces = result.get('faces', [])
                    print(f"   ✅ Successfully extracted {len(faces)} faces")
                    return faces
                else:
                    print(f"   ❌ API returned error: {result.get('message', 'Unknown error')}")
                    return None
            else:
                print(f"❌ Server error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error calling extract_faces API: {e}")
            return None

    def process_extracted_faces(self, faces_data: List[Dict], base_filename: str) -> List[Dict]:
        """
        Step 2: Process extracted faces and analyze their quality
        
        Args:
            faces_data: List of face data from extract_faces_from_server
            base_filename: Base filename for saving temp files
            
        Returns:
            List of processed face data with quality analysis
        """
        print(f"🔄 Step 2: Processing extracted faces and analyzing quality")
        
        # Debug: Print the structure of faces_data
        print(f"   📊 Debug - faces_data structure:")
        for i, face in enumerate(faces_data):
            print(f"      Face {i}: keys = {list(face.keys())}")
            for key in ['face_id', 'confidence', 'antispoof_score', 'is_real']:
                if key in face:
                    print(f"               {key} = {face.get(key, 'N/A')}")
            if 'face_image' in face:
                face_img_len = len(face['face_image']) if isinstance(face['face_image'], str) else 'N/A'
                print(f"               face_image length = {face_img_len}")
            else:
                print(f"               ❌ face_image key missing!")
        
        processed_faces = []
        for i, face in enumerate(faces_data):
            face_id = face.get('face_id', i + 1)
            confidence = face.get('confidence', 0)
            is_real = face.get('is_real', True)
            
            if 'face_image' in face:
                # Temporarily save face for quality analysis
                temp_face_path = f"{base_filename}_face_{face_id}_temp_analysis.jpg"
                if self.save_base64_image(face['face_image'], temp_face_path):
                    print(f"   📊 Analyzing quality of face {face_id}...")
                    quality_info = self.analyze_face_quality(temp_face_path)
                    
                    if quality_info:
                        print(f"      📈 Quality analysis complete:")
                        print(f"         Quality score: {quality_info.get('quality_score', 0):.2f}")
                        print(f"         Lighting quality: {quality_info.get('lighting_quality', 'unknown')}")
                        print(f"         Brightness: {quality_info.get('brightness', 0):.1f}")
                        print(f"         Contrast: {quality_info.get('contrast', 0):.1f}")
                        print(f"         Needs enhancement: {quality_info.get('needs_enhancement', False)}")
                        
                        if quality_info.get('detected_issues'):
                            print(f"         Issues: {', '.join(quality_info['detected_issues'])}")
                        if quality_info.get('recommendations'):
                            print(f"         Recommendations: {', '.join(quality_info['recommendations'][:3])}...")
                    else:
                        print(f"      ⚠️ Quality analysis failed for face {face_id}")
                    
                    # Clean up temp analysis file
                    try:
                        os.remove(temp_face_path)
                    except:
                        pass
                else:
                    quality_info = None
                    print(f"   ⚠️ Could not save face {face_id} for quality analysis")
                
                processed_faces.append({
                    'face_id': face_id,
                    'confidence': confidence,
                    'is_real': is_real,
                    'face_image_data': face['face_image'],
                    'face_info': face,
                    'quality_info': quality_info,
                    'base_filename': f"{base_filename}_face_{face_id}_conf_{confidence:.2f}"
                })
                print(f"   ✅ Face {face_id}: processed in memory (confidence: {confidence:.2f}, real: {is_real})")
            else:
                print(f"   ❌ No face image data for face {face_id}")
        
        print(f"✅ Processed {len(processed_faces)} faces in memory")
        return processed_faces

    def face_recognition(self, image_path: str, description: str = "") -> Dict[str, Any]:
        """
        Face recognition using the face-only endpoint
        
        Args:
            image_path: Path to the face image
            description: Optional description for logging
            
        Returns:
            Dictionary with recognition results
        """
        print(f"🔍 Face recognition{' - ' + description if description else ''}")
        print(f"   Image: {image_path}")
        
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                response = requests.post(
                    f"{self.api_base}/recognize_face_only",
                    files=files,
                    cert=self.cert,
                    verify=self.verify_ssl
                )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   📊 API Response keys: {list(result.keys())}")
                print(f"   📊 Success: {result.get('success', False)}")
                
                if result.get('success') and result.get('best_match'):
                    match = result['best_match']
                    confidence = match.get('confidence', 0)
                    print(f"   ✅ RECOGNIZED: {match['person_name']} (confidence: {confidence:.3f})")
                    print(f"      Distance: {match.get('distance', 0):.3f}")
                    print(f"      Processing time: {result.get('processing_time_ms', 0)}ms")
                    
                    return {
                        'success': True,
                        'person_name': match['person_name'],
                        'confidence': confidence,
                        'distance': match.get('distance', 0),
                        'processing_time_ms': result.get('processing_time_ms', 0)
                    }
                else:
                    print(f"   ❌ NOT RECOGNIZED")
                    print(f"      Processing time: {result.get('processing_time_ms', 0)}ms")
                    if 'error' in result:
                        print(f"      Error: {result['error']}")
                    if 'message' in result:
                        print(f"      Message: {result['message']}")
                    
                    return {
                        'success': False,
                        'message': result.get('message', 'Not recognized'),
                        'error': result.get('error', ''),
                        'processing_time_ms': result.get('processing_time_ms', 0)
                    }
            else:
                print(f"   ❌ Server error: {response.status_code}")
                print(f"   Response: {response.text}")
                return {
                    'success': False,
                    'error': f'HTTP {response.status_code}',
                    'response_text': response.text
                }
                
        except Exception as e:
            print(f"   ❌ Error testing recognition: {e}")
            return {
                'success': False,
                'error': str(e)
            }

# Convenience functions for backward compatibility
def create_api_client(server_url: str = "https://callfusion.ptype.co.kr:55000",
                     certs_dir: str = "./certs") -> FaceRecognitionAPI:
    """Create a new API client instance"""
    return FaceRecognitionAPI(server_url, certs_dir)

def extract_faces_from_server(image_path: str, 
                            server_url: str = "https://callfusion.ptype.co.kr:55000",
                            certs_dir: str = "./certs") -> Optional[List[Dict]]:
    """Convenience function to extract faces (creates temporary client)"""
    client = FaceRecognitionAPI(server_url, certs_dir)
    return client.extract_faces_from_server(image_path)

def face_recognition(image_path: str, 
                    description: str = "",
                    server_url: str = "https://callfusion.ptype.co.kr:55000",
                    certs_dir: str = "./certs") -> Dict[str, Any]:
    """Convenience function to face recognition (creates temporary client)"""
    client = FaceRecognitionAPI(server_url, certs_dir)
    return client.face_recognition(image_path, description)