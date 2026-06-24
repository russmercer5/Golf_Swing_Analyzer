# app.py - Golf Swing Analysis WebApp (Render Compatible)
import cv2
import mediapipe as mp
import numpy as np
import os
import json
import math
import base64
import tempfile
import threading
import time
from flask import Flask, render_template, request, jsonify, session, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'golf_swing_analysis_2024_secret_key')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['JSON_AS_ASCII'] = False
CORS(app)

# Create uploads folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize MediaPipe Pose with error handling for Render
try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
except Exception as e:
    print(f"Warning: MediaPipe initialization issue: {e}")
    # Fallback for deployment
    mp_pose = None
    mp_drawing = None
    mp_drawing_styles = None

# Golf swing positions
SWING_POSITIONS = {
    1: "Setup / Address",
    2: "Club Parallel (Back)",
    3: "Left Arm Parallel (Back)",
    4: "Top of Backswing",
    5: "Left Arm Parallel (Down)",
    6: "Club Parallel (Down)",
    7: "Impact",
    8: "Club Parallel (Follow)",
    9: "Finish"
}

class GolfSwingAnalyzer:
    def __init__(self):
        if mp_pose is None:
            raise Exception("MediaPipe not available")
        # Use model_complexity=1 for better performance on Render
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points in degrees"""
        try:
            v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            return np.degrees(angle)
        except:
            return 0
    
    def get_landmark(self, landmarks, landmark_type):
        """Get landmark coordinates safely"""
        try:
            return landmarks[landmark_type.value]
        except:
            return None
    
    def calculate_metrics(self, landmarks, frame_shape, view_type='down_the_line'):
        """Calculate all biomechanical metrics from landmarks"""
        if not landmarks:
            return None
        
        try:
            metrics = {}
            
            # Get all necessary landmarks
            left_shoulder = self.get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
            right_shoulder = self.get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
            left_hip = self.get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
            right_hip = self.get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
            left_knee = self.get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
            right_knee = self.get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE)
            left_ankle = self.get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
            right_ankle = self.get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)
            
            if None in [left_shoulder, right_shoulder, left_hip, right_hip]:
                return None
            
            # Shoulder Turn (rotation of shoulders relative to hips)
            shoulder_vector = np.array([
                right_shoulder.x - left_shoulder.x,
                right_shoulder.z - left_shoulder.z
            ])
            hip_vector = np.array([
                right_hip.x - left_hip.x,
                right_hip.z - left_hip.z
            ])
            
            if np.linalg.norm(shoulder_vector) > 0 and np.linalg.norm(hip_vector) > 0:
                shoulder_turn = np.degrees(np.arctan2(
                    np.cross(shoulder_vector, hip_vector),
                    np.dot(shoulder_vector, hip_vector)
                ))
            else:
                shoulder_turn = 0
            
            # Hip Turn (rotation of hips relative to stance)
            hip_turn = np.degrees(np.arctan2(
                right_hip.z - left_hip.z,
                right_hip.x - left_hip.x
            ))
            
            # Shoulder Tilt (side bend)
            shoulder_tilt = self.calculate_angle(
                left_shoulder,
                right_shoulder,
                right_hip
            )
            
            # Hip Tilt (side bend)
            hip_tilt = self.calculate_angle(
                left_hip,
                right_hip,
                right_knee if right_knee else left_knee
            )
            
            # Shoulder Sway (lateral movement)
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            hip_center_x = (left_hip.x + right_hip.x) / 2
            shoulder_sway = abs(shoulder_center_x - hip_center_x) * 100
            
            # Hip Sway (lateral movement)
            hip_sway = abs(left_hip.x - right_hip.x) * 100
            
            # Spine angle
            spine_angle = self.calculate_angle(
                left_shoulder,
                left_hip,
                left_knee if left_knee else left_ankle
            )
            
            metrics = {
                'shoulder_turn': round(shoulder_turn, 1),
                'hip_turn': round(hip_turn, 1),
                'shoulder_tilt': round(shoulder_tilt, 1),
                'hip_tilt': round(hip_tilt, 1),
                'shoulder_sway': round(shoulder_sway, 1),
                'hip_sway': round(hip_sway, 1),
                'spine_angle': round(spine_angle, 1)
            }
            
            # Store landmarks for drawing
            metrics['landmarks'] = landmarks
            metrics['view_type'] = view_type
            
            return metrics
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return None
    
    def detect_swing_positions(self, landmarks_history, fps):
        """Auto-detect 9 swing positions based on club and arm positions"""
        if len(landmarks_history) < 10:
            return {}, {}
        
        positions = {}
        position_frames = {}
        
        try:
            # Simplified detection - use evenly spaced frames
            total_frames = len(landmarks_history)
            
            # Evenly distribute positions
            for pos in range(1, 10):
                frame_idx = int((pos - 1) * total_frames / 8)
                if 0 <= frame_idx < total_frames:
                    positions[pos] = frame_idx
                    position_frames[pos] = landmarks_history[frame_idx]
            
            # Try to find key positions for better accuracy
            for i, landmarks in enumerate(landmarks_history):
                if not landmarks:
                    continue
                    
                if i < total_frames * 0.15:
                    # Setup position
                    if 1 not in positions or abs(i - positions[1]) < total_frames * 0.1:
                        positions[1] = i
                        position_frames[1] = landmarks
                
                elif i > total_frames * 0.35 and i < total_frames * 0.45:
                    # Top of backswing
                    if 4 not in positions or abs(i - positions[4]) < total_frames * 0.1:
                        positions[4] = i
                        position_frames[4] = landmarks
                
                elif i > total_frames * 0.5 and i < total_frames * 0.6:
                    # Impact
                    if 7 not in positions or abs(i - positions[7]) < total_frames * 0.1:
                        positions[7] = i
                        position_frames[7] = landmarks
                
                elif i > total_frames * 0.85:
                    # Finish
                    if 9 not in positions or abs(i - positions[9]) < total_frames * 0.1:
                        positions[9] = i
                        position_frames[9] = landmarks
            
            return positions, position_frames
        except Exception as e:
            print(f"Error detecting positions: {e}")
            return {}, {}
    
    def draw_angle_lines(self, frame, landmarks, metrics):
        """Draw lines and angles on the frame showing metrics"""
        if not landmarks:
            return frame
        
        try:
            h, w = frame.shape[:2]
            
            # Get key landmarks
            left_shoulder = self.get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
            right_shoulder = self.get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
            left_hip = self.get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
            right_hip = self.get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
            
            # Draw lines for shoulder tilt
            if left_shoulder and right_shoulder:
                p1 = (int(left_shoulder.x * w), int(left_shoulder.y * h))
                p2 = (int(right_shoulder.x * w), int(right_shoulder.y * h))
                cv2.line(frame, p1, p2, (0, 200, 255), 3)
                
                if 'shoulder_tilt' in metrics:
                    cv2.putText(frame, f"Tilt: {metrics['shoulder_tilt']}°", 
                               (p1[0], p1[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (0, 200, 255), 2)
            
            # Draw lines for hip tilt
            if left_hip and right_hip:
                p1 = (int(left_hip.x * w), int(left_hip.y * h))
                p2 = (int(right_hip.x * w), int(right_hip.y * h))
                cv2.line(frame, p1, p2, (255, 200, 0), 3)
                
                if 'hip_tilt' in metrics:
                    cv2.putText(frame, f"Hip Tilt: {metrics['hip_tilt']}°", 
                               (p1[0], p1[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (255, 200, 0), 2)
            
            # Draw shoulder line to hip line (for turn measurement)
            if left_shoulder and right_hip:
                p1 = (int(left_shoulder.x * w), int(left_shoulder.y * h))
                p2 = (int(right_hip.x * w), int(right_hip.y * h))
                cv2.line(frame, p1, p2, (0, 255, 100), 2, cv2.LINE_AA)
                
                if 'shoulder_turn' in metrics:
                    mid_x = (p1[0] + p2[0]) // 2
                    mid_y = (p1[1] + p2[1]) // 2
                    cv2.putText(frame, f"Turn: {metrics['shoulder_turn']}°", 
                               (mid_x - 30, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (0, 255, 100), 2)
            
            # Draw spine line
            if left_shoulder and left_hip:
                p1 = (int(left_shoulder.x * w), int(left_shoulder.y * h))
                p2 = (int(left_hip.x * w), int(left_hip.y * h))
                cv2.line(frame, p1, p2, (255, 100, 255), 2, cv2.LINE_AA)
            
            return frame
        except Exception as e:
            print(f"Error drawing lines: {e}")
            return frame
    
    def process_video(self, video_path, view_type='down_the_line'):
        """Process video and extract metrics for all 9 swing positions"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return None, None
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps == 0:
                fps = 30
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Extract frames and landmarks
            landmarks_history = []
            frames = []
            
            frame_interval = max(1, fps // 10)  # Sample ~10 frames per second
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Process frame
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = self.pose.process(rgb_frame)
                    
                    if result.pose_landmarks:
                        landmarks_history.append(result.pose_landmarks.landmark)
                        frames.append(frame.copy())
                        
                        # Annotate frame
                        if mp_drawing:
                            mp_drawing.draw_landmarks(
                                frame,
                                result.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                            )
                    else:
                        landmarks_history.append(None)
                        frames.append(frame)
                
                frame_count += 1
            
            cap.release()
            
            if len(landmarks_history) < 10:
                return None, None
            
            # Detect swing positions
            positions, position_frames = self.detect_swing_positions(landmarks_history, fps)
            
            results = {}
            position_images = {}
            
            for pos, frame_idx in positions.items():
                if frame_idx < len(landmarks_history) and landmarks_history[frame_idx]:
                    metrics = self.calculate_metrics(
                        landmarks_history[frame_idx], 
                        frames[frame_idx].shape,
                        view_type
                    )
                    if metrics:
                        # Draw angle lines on the frame
                        annotated_frame = frames[frame_idx].copy()
                        annotated_frame = self.draw_angle_lines(
                            annotated_frame,
                            landmarks_history[frame_idx],
                            metrics
                        )
                        
                        # Draw landmarks
                        if mp_drawing:
                            mp_drawing.draw_landmarks(
                                annotated_frame,
                                landmarks_history[frame_idx],
                                mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                            )
                        
                        results[pos] = metrics
                        position_images[pos] = annotated_frame
            
            return results, position_images
        except Exception as e:
            print(f"Error processing video: {e}")
            return None, None

# ============================================
# FLASK ROUTES
# ============================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """Analyze uploaded video and return results"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        view_type = request.form.get('view_type', 'down_the_line')
        
        if video_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Generate unique filename
        filename = f"{uuid.uuid4().hex}.mp4"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(filepath)
        
        # Process video
        analyzer = GolfSwingAnalyzer()
        results, position_images = analyzer.process_video(filepath, view_type)
        
        if not results:
            try:
                os.remove(filepath)
            except:
                pass
            return jsonify({'error': 'Could not detect body pose in video'}), 400
        
        # Convert images to base64
        images_base64 = {}
        for pos, img in position_images.items():
            try:
                _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                images_base64[str(pos)] = base64.b64encode(buffer).decode('utf-8')
            except:
                pass
        
        # Prepare results for display
        results_display = []
        for pos in sorted(results.keys()):
            pos_data = results[pos].copy()
            # Remove landmarks from response (too large)
            pos_data.pop('landmarks', None)
            pos_data.pop('view_type', None)
            
            results_display.append({
                'position': pos,
                'position_name': SWING_POSITIONS.get(pos, f'Position {pos}'),
                'metrics': pos_data
            })
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'success': True,
            'results': results_display,
            'images': images_base64,
            'positions': SWING_POSITIONS,
            'view_type': view_type
        })
        
    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/export', methods=['POST'])
def export_data():
    """Export results as CSV or JSON"""
    try:
        data = request.json
        format_type = data.get('format', 'csv')
        results = data.get('results', [])
        
        if format_type == 'csv':
            import csv
            from io import StringIO
            
            output = StringIO()
            writer = csv.writer(output)
            
            # Header
            header = ['Position', 'Position Name', 'Shoulder Turn', 'Hip Turn', 
                     'Shoulder Tilt', 'Hip Tilt', 'Shoulder Sway', 'Hip Sway', 'Spine Angle']
            writer.writerow(header)
            
            # Data rows
            for r in results:
                m = r.get('metrics', {})
                row = [
                    r.get('position', ''),
                    r.get('position_name', ''),
                    m.get('shoulder_turn', ''),
                    m.get('hip_turn', ''),
                    m.get('shoulder_tilt', ''),
                    m.get('hip_tilt', ''),
                    m.get('shoulder_sway', ''),
                    m.get('hip_sway', ''),
                    m.get('spine_angle', '')
                ]
                writer.writerow(row)
            
            output.seek(0)
            return jsonify({
                'success': True,
                'data': output.getvalue(),
                'filename': f"swing_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            })
        
        else:
            # JSON format
            return jsonify({
                'success': True,
                'data': json.dumps(results, indent=2),
                'filename': f"swing_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results')
def results():
    return render_template('results.html')

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Server error'}), 500

# ============================================
# HTML TEMPLATES - KEPT THE SAME
# ============================================

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Golf Swing Analyzer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a1a;
            color: #fff;
            padding: 16px;
            min-height: 100vh;
        }
        .container { max-width: 500px; margin: 0 auto; }
        h1 {
            text-align: center;
            color: #4fc3f7;
            font-size: 24px;
            margin-bottom: 8px;
        }
        .subtitle {
            text-align: center;
            color: #888;
            font-size: 14px;
            margin-bottom: 20px;
        }
        .camera-container {
            background: #16213e;
            border-radius: 16px;
            overflow: hidden;
            margin-bottom: 16px;
            position: relative;
            min-height: 300px;
        }
        #videoPreview {
            width: 100%;
            display: block;
            background: #0a0a1a;
            min-height: 300px;
            object-fit: cover;
        }
        #videoPreview.hidden { display: none; }
        .camera-placeholder {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 300px;
            color: #666;
            padding: 20px;
            text-align: center;
        }
        .camera-placeholder .icon { font-size: 64px; margin-bottom: 16px; }
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }
        .controls button {
            flex: 1;
            padding: 14px 20px;
            border: none;
            border-radius: 12px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            touch-action: manipulation;
        }
        .controls button:active { transform: scale(0.95); }
        .btn-primary {
            background: #4fc3f7;
            color: #0a0a1a;
        }
        .btn-success {
            background: #66bb6a;
            color: #0a0a1a;
        }
        .btn-danger {
            background: #ef5350;
            color: #fff;
        }
        .btn-secondary {
            background: #2d3b5e;
            color: #fff;
        }
        .btn:disabled {
            opacity: 0.5;
            pointer-events: none;
        }
        .view-selector {
            display: flex;
            gap: 10px;
            margin-bottom: 16px;
        }
        .view-selector button {
            flex: 1;
            padding: 12px;
            border-radius: 10px;
            border: 2px solid #2d3b5e;
            background: transparent;
            color: #888;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .view-selector button.active {
            border-color: #4fc3f7;
            background: #16213e;
            color: #fff;
        }
        .progress-container {
            display: none;
            margin: 16px 0;
            padding: 16px;
            background: #16213e;
            border-radius: 12px;
        }
        .progress-bar {
            width: 100%;
            height: 6px;
            background: #2d3b5e;
            border-radius: 3px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: #4fc3f7;
            width: 0%;
            transition: width 0.5s;
        }
        .progress-text {
            text-align: center;
            color: #aaa;
            font-size: 14px;
            margin-top: 8px;
        }
        #statusMessage {
            text-align: center;
            padding: 12px;
            border-radius: 10px;
            margin: 10px 0;
            display: none;
        }
        #statusMessage.error { background: #2d1a1a; color: #ef5350; display: block; }
        #statusMessage.success { background: #1a2d1a; color: #66bb6a; display: block; }
        #statusMessage.info { background: #1a2d3d; color: #4fc3f7; display: block; }
        .file-info {
            color: #aaa;
            font-size: 12px;
            text-align: center;
            margin: 8px 0;
        }
        @media (max-width: 480px) {
            body { padding: 10px; }
            h1 { font-size: 20px; }
            .controls button { padding: 16px; font-size: 14px; }
            .camera-placeholder .icon { font-size: 48px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>⛳ Golf Swing Analyzer</h1>
        <p class="subtitle">Record your swing and get instant biomechanical analysis</p>
        
        <div class="camera-container">
            <video id="videoPreview" class="hidden" playsinline autoplay muted></video>
            <div class="camera-placeholder" id="cameraPlaceholder">
                <div class="icon">🎯</div>
                <div>Tap "Start Camera" to begin</div>
                <div style="font-size:12px; color:#555; margin-top:8px;">Position camera for down-the-line or front-on view</div>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn-primary" id="startCamera">📷 Start Camera</button>
            <button class="btn-success" id="captureBtn" disabled>🎬 Record & Analyze</button>
        </div>
        
        <div class="view-selector">
            <button class="active" data-view="down_the_line">Down the Line</button>
            <button data-view="front_on">Front On</button>
        </div>
        
        <div class="file-info" id="fileInfo"></div>
        
        <div class="progress-container" id="progressContainer">
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <div class="progress-text" id="progressText">Processing...</div>
        </div>
        
        <div id="statusMessage"></div>
    </div>

    <script>
        // State
        let stream = null;
        let mediaRecorder = null;
        let recordedChunks = [];
        let isRecording = false;
        let selectedView = 'down_the_line';
        let videoFile = null;
        
        const videoPreview = document.getElementById('videoPreview');
        const cameraPlaceholder = document.getElementById('cameraPlaceholder');
        const startCameraBtn = document.getElementById('startCamera');
        const captureBtn = document.getElementById('captureBtn');
        const progressContainer = document.getElementById('progressContainer');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const statusMessage = document.getElementById('statusMessage');
        const fileInfo = document.getElementById('fileInfo');
        
        // View selector
        document.querySelectorAll('.view-selector button').forEach(btn => {
            btn.addEventListener('click', function() {
                document.querySelectorAll('.view-selector button').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                selectedView = this.dataset.view;
            });
        });
        
        // Start camera
        startCameraBtn.addEventListener('click', async function() {
            try {
                const constraints = {
                    video: {
                        facingMode: 'environment',
                        width: { ideal: 1280 },
                        height: { ideal: 720 }
                    },
                    audio: true
                };
                
                stream = await navigator.mediaDevices.getUserMedia(constraints);
                videoPreview.srcObject = stream;
                videoPreview.classList.remove('hidden');
                cameraPlaceholder.style.display = 'none';
                
                this.textContent = '📹 Camera Active';
                this.className = 'btn-success';
                captureBtn.disabled = false;
                
                showStatus('Camera ready! Tap "Record & Analyze" to capture your swing.', 'info');
                
            } catch (err) {
                console.error('Camera error:', err);
                showStatus('Error accessing camera: ' + err.message, 'error');
            }
        });
        
        // Record and analyze
        captureBtn.addEventListener('click', function() {
            if (isRecording) {
                stopRecordingAndAnalyze();
                return;
            }
            
            if (!stream) {
                showStatus('Please start the camera first.', 'error');
                return;
            }
            
            startRecording();
        });
        
        function startRecording() {
            recordedChunks = [];
            
            try {
                mediaRecorder = new MediaRecorder(stream);
                
                mediaRecorder.ondataavailable = function(event) {
                    if (event.data.size > 0) {
                        recordedChunks.push(event.data);
                    }
                };
                
                mediaRecorder.onstop = function() {
                    processRecording();
                };
                
                mediaRecorder.start();
                isRecording = true;
                
                captureBtn.textContent = '⏹ Stop Recording';
                captureBtn.className = 'btn-danger';
                
                showStatus('Recording... Swing now! Tap stop when finished.', 'info');
                
                // Auto-stop after 30 seconds
                setTimeout(() => {
                    if (isRecording) {
                        stopRecordingAndAnalyze();
                    }
                }, 30000);
                
            } catch (err) {
                console.error('Recording error:', err);
                showStatus('Error recording: ' + err.message, 'error');
            }
        }
        
        function stopRecordingAndAnalyze() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                isRecording = false;
                captureBtn.textContent = '⏳ Processing...';
                captureBtn.disabled = true;
            }
        }
        
        function processRecording() {
            if (recordedChunks.length === 0) {
                showStatus('No video recorded. Please try again.', 'error');
                captureBtn.textContent = '🎬 Record & Analyze';
                captureBtn.className = 'btn-success';
                captureBtn.disabled = false;
                return;
            }
            
            // Create video blob
            const blob = new Blob(recordedChunks, { type: 'video/mp4' });
            const file = new File([blob], 'swing.mp4', { type: 'video/mp4' });
            videoFile = file;
            
            // Show file info
            fileInfo.textContent = `📁 Video: ${(blob.size / 1024 / 1024).toFixed(2)} MB`;
            
            // Upload and analyze
            uploadAndAnalyze(file);
        }
        
        function uploadAndAnalyze(file) {
            const formData = new FormData();
            formData.append('video', file);
            formData.append('view_type', selectedView);
            
            progressContainer.style.display = 'block';
            progressFill.style.width = '0%';
            progressText.textContent = 'Uploading...';
            
            fetch('/analyze', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                progressFill.style.width = '100%';
                progressText.textContent = 'Complete!';
                
                if (data.success) {
                    // Store results in session and navigate to results
                    sessionStorage.setItem('results', JSON.stringify(data));
                    window.location.href = '/results';
                } else {
                    showStatus('Error: ' + (data.error || 'Processing failed'), 'error');
                    resetControls();
                }
            })
            .catch(error => {
                showStatus('Network error: ' + error.message, 'error');
                resetControls();
            });
        }
        
        function resetControls() {
            captureBtn.textContent = '🎬 Record & Analyze';
            captureBtn.className = 'btn-success';
            captureBtn.disabled = false;
            progressContainer.style.display = 'none';
        }
        
        function showStatus(msg, type) {
            statusMessage.textContent = msg;
            statusMessage.className = type || 'info';
            statusMessage.style.display = 'block';
            
            // Auto-hide after 5 seconds
            clearTimeout(window.statusTimeout);
            window.statusTimeout = setTimeout(() => {
                statusMessage.style.display = 'none';
            }, 5000);
        }
        
        // Handle window close - stop camera
        window.addEventListener('beforeunload', function() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
        });
    </script>
</body>
</html>
"""

RESULTS_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Swing Analysis Results</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a1a;
            color: #fff;
            padding: 16px;
            min-height: 100vh;
        }
        .container { max-width: 500px; margin: 0 auto; }
        h1 {
            text-align: center;
            color: #4fc3f7;
            font-size: 22px;
            margin-bottom: 8px;
        }
        .back-btn {
            background: #2d3b5e;
            color: #fff;
            border: none;
            padding: 10px 20px;
            border-radius: 10px;
            font-size: 14px;
            cursor: pointer;
            margin-bottom: 16px;
        }
        .back-btn:active { transform: scale(0.96); }
        
        .summary-stats {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 10px;
            margin: 15px 0;
        }
        .stat-card {
            background: #16213e;
            border-radius: 10px;
            padding: 12px;
            text-align: center;
        }
        .stat-card .number {
            font-size: 24px;
            font-weight: 700;
            color: #4fc3f7;
        }
        .stat-card .label {
            font-size: 10px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .results-table {
            background: #16213e;
            border-radius: 12px;
            overflow-x: auto;
            margin: 15px 0;
            padding: 10px;
        }
        .results-table table {
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
        }
        .results-table th {
            background: #2d3b5e;
            color: #4fc3f7;
            padding: 8px 4px;
            text-align: center;
            position: sticky;
            top: 0;
        }
        .results-table td {
            padding: 6px 4px;
            text-align: center;
            border-bottom: 1px solid #2d3b5e;
            color: #ddd;
        }
        .results-table .pos-label {
            color: #ffa726;
            font-weight: 600;
            font-size: 10px;
        }
        
        .position-card {
            background: #16213e;
            border-radius: 12px;
            margin: 15px 0;
            overflow: hidden;
        }
        .position-card img {
            width: 100%;
            display: block;
        }
        .position-metrics {
            padding: 12px;
        }
        .position-metrics .pos-title {
            color: #ffa726;
            font-weight: 600;
            font-size: 16px;
            margin-bottom: 8px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 6px;
        }
        .metric-item {
            background: #1a2a3a;
            border-radius: 6px;
            padding: 6px 8px;
            text-align: center;
            font-size: 11px;
        }
        .metric-item .label {
            color: #888;
            font-size: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-item .value {
            color: #4fc3f7;
            font-weight: 600;
            font-size: 14px;
        }
        
        .export-section {
            display: flex;
            gap: 10px;
            margin: 15px 0;
        }
        .export-section button {
            flex: 1;
            padding: 14px;
            border: none;
            border-radius: 10px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .export-section button:active { transform: scale(0.96); }
        .btn-csv { background: #4fc3f7; color: #0a0a1a; }
        .btn-json { background: #ffa726; color: #0a0a1a; }
        
        .loading {
            text-align: center;
            padding: 40px 20px;
            color: #aaa;
        }
        .loading .spinner {
            display: inline-block;
            width: 40px;
            height: 40px;
            border: 4px solid rgba(255,255,255,.1);
            border-radius: 50%;
            border-top-color: #4fc3f7;
            animation: spin 1s ease-in-out infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        
        @media (max-width: 480px) {
            .results-table { font-size: 9px; }
            .results-table th, .results-table td { padding: 3px 2px; }
            .metric-grid { grid-template-columns: 1fr 1fr; }
            .summary-stats { grid-template-columns: 1fr 1fr 1fr; gap: 6px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <button class="back-btn" onclick="window.location.href='/'">← Back</button>
        <h1>📊 Swing Analysis Results</h1>
        
        <div id="loading" class="loading">
            <div class="spinner"></div>
            <p style="margin-top:15px;">Loading results...</p>
        </div>
        
        <div id="resultsContent" style="display:none;">
            <div id="summaryStats" class="summary-stats"></div>
            
            <div class="results-table" id="tableContainer"></div>
            
            <div class="export-section">
                <button class="btn-csv" onclick="exportData('csv')">📥 CSV</button>
                <button class="btn-json" onclick="exportData('json')">📄 JSON</button>
            </div>
            
            <div id="positionsContainer"></div>
        </div>
    </div>

    <script>
        let resultsData = null;
        
        document.addEventListener('DOMContentLoaded', function() {
            const dataStr = sessionStorage.getItem('results');
            
            if (!dataStr) {
                document.getElementById('loading').innerHTML = '<p>No results found. Please analyze a swing first.</p>';
                return;
            }
            
            try {
                resultsData = JSON.parse(dataStr);
                displayResults(resultsData);
            } catch (e) {
                document.getElementById('loading').innerHTML = '<p style="color:#ef5350;">Error loading results.</p>';
                console.error(e);
            }
        });
        
        function displayResults(data) {
            document.getElementById('loading').style.display = 'none';
            document.getElementById('resultsContent').style.display = 'block';
            
            const results = data.results || [];
            const images = data.images || {};
            const positions = data.positions || {};
            
            // Summary stats
            if (results.length > 0) {
                let avgShoulderTurn = 0, avgHipTurn = 0;
                results.forEach(r => {
                    avgShoulderTurn += r.metrics.shoulder_turn || 0;
                    avgHipTurn += r.metrics.hip_turn || 0;
                });
                avgShoulderTurn = Math.round(avgShoulderTurn / results.length);
                avgHipTurn = Math.round(avgHipTurn / results.length);
                
                document.getElementById('summaryStats').innerHTML = `
                    <div class="stat-card">
                        <div class="number">${results.length}</div>
                        <div class="label">Positions</div>
                    </div>
                    <div class="stat-card">
                        <div class="number">${avgShoulderTurn}°</div>
                        <div class="label">Avg Shoulder Turn</div>
                    </div>
                    <div class="stat-card">
                        <div class="number">${avgHipTurn}°</div>
                        <div class="label">Avg Hip Turn</div>
                    </div>
                `;
            }
            
            // Build table
            let tableHtml = '<table><thead><tr><th>Pos</th>';
            const metricKeys = ['shoulder_turn', 'hip_turn', 'shoulder_tilt', 'hip_tilt', 'shoulder_sway', 'hip_sway'];
            const metricLabels = ['Shoulder Turn', 'Hip Turn', 'Shoulder Tilt', 'Hip Tilt', 'Shoulder Sway', 'Hip Sway'];
            
            metricLabels.forEach(label => {
                tableHtml += `<th>${label}</th>`;
            });
            tableHtml += '</tr></thead><tbody>';
            
            results.forEach(r => {
                const posNum = r.position;
                tableHtml += `<tr><td class="pos-label">${posNum}</td>`;
                
                const m = r.metrics || {};
                metricKeys.forEach(key => {
                    const val = m[key] !== undefined ? m[key] : '—';
                    tableHtml += `<td>${val}</td>`;
                });
                tableHtml += '</tr>';
            });
            tableHtml += '</tbody></table>';
            document.getElementById('tableContainer').innerHTML = tableHtml;
            
            // Build position cards with images
            let cardsHtml = '';
            results.forEach(r => {
                const posNum = r.position;
                const posName = positions[posNum] || `Position ${posNum}`;
                const imgBase64 = images[posNum] || '';
                const m = r.metrics || {};
                
                cardsHtml += `
                    <div class="position-card">
                        ${imgBase64 ? `<img src="data:image/jpeg;base64,${imgBase64}" alt="Position ${posNum}">` : ''}
                        <div class="position-metrics">
                            <div class="pos-title">${posNum}. ${posName}</div>
                            <div class="metric-grid">
                                <div class="metric-item">
                                    <div class="label">Shoulder Turn</div>
                                    <div class="value">${m.shoulder_turn || '—'}°</div>
                                </div>
                                <div class="metric-item">
                                    <div class="label">Hip Turn</div>
                                    <div class="value">${m.hip_turn || '—'}°</div>
                                </div>
                                <div class="metric-item">
                                    <div class="label">Shoulder Tilt</div>
                                    <div class="value">${m.shoulder_tilt || '—'}°</div>
                                </div>
                                <div class="metric-item">
                                    <div class="label">Hip Tilt</div>
                                    <div class="value">${m.hip_tilt || '—'}°</div>
                                </div>
                                <div class="metric-item">
                                    <div class="label">Shoulder Sway</div>
                                    <div class="value">${m.shoulder_sway || '—'}</div>
                                </div>
                                <div class="metric-item">
                                    <div class="label">Hip Sway</div>
                                    <div class="value">${m.hip_sway || '—'}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            });
            document.getElementById('positionsContainer').innerHTML = cardsHtml;
        }
        
        function exportData(format) {
            if (!resultsData) return;
            
            fetch('/export', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    format: format,
                    results: resultsData.results
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Download file
                    const blob = new Blob([data.data], { type: 'text/plain' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = data.filename;
                    a.click();
                    URL.revokeObjectURL(url);
                } else {
                    alert('Export error: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(error => {
                alert('Export error: ' + error.message);
            });
        }
    </script>
</body>
</html>
"""

# Register routes
@app.route('/results')
def results_route():
    return RESULTS_TEMPLATE

@app.route('/')
def index_route():
    return INDEX_TEMPLATE

if __name__ == '__main__':
    # Use environment variables for Render
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print(f"Starting Golf Swing Analyzer on port {port}")
    print("Open this URL on your mobile device to use the app")
    
    app.run(debug=debug, host='0.0.0.0', port=port, threaded=True)
