# --- start: paste/replace this chunk into your app.py ---

import os
import tempfile
import shutil
import subprocess
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import torch

# your model functions (keep your existing import)
from model.model2 import process_video, process_audio, VideoFeatureExtractor, AudioFeatureExtractor, DeepfakeClassifier

app = Flask(__name__, static_folder="static", template_folder="templates")

# put uploads outside OneDrive to avoid lock issues
UPLOAD_FOLDER = r"C:\temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# allowed extensions
ALLOWED_EXT = {'.mp4', '.mov', '.avi', '.mkv'}

def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXT

@app.route('/')
def home():
    return render_template('index.html', content="Upload a video to generate output.")

@app.route('/analyze', methods=['POST'])
def handle_submit():
    # 1) basic checks
    if 'video' not in request.files:
        return render_template('index.html', content="No video file received. Please upload a valid video.")
    video_file = request.files['video']
    if video_file.filename == "":
        return render_template('index.html', content="Empty filename. Please upload a valid video.")
    if not allowed_file(video_file.filename):
        return render_template('index.html', content="Unsupported file type. Upload mp4/mov/avi/mkv.")

    # 2) save uploaded file safely
    filename = secure_filename(video_file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    video_file.save(upload_path)  # Flask closes the incoming file handle

    # 3) create a temp copy so ffmpeg/pydub can open it on Windows
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
    tmp.close()
    shutil.copyfile(upload_path, tmp.name)

    try:
        # 4) processing using the temp file path
        video_frames = process_video(tmp.name)
        audio_features = process_audio(tmp.name)

        # 5) model inference
        video_model = VideoFeatureExtractor()
        audio_model = AudioFeatureExtractor()
        classifier = DeepfakeClassifier()
        video_model.eval(); audio_model.eval(); classifier.eval()

        with torch.no_grad():
            video_feats = video_model(video_frames)
            audio_feats = audio_model(audio_features)
            prediction_tensor = classifier(video_feats, audio_feats)

        real_probability = float(prediction_tensor[0, 0])
        prediction = "Real" if real_probability > 0.49 else "Fake"

    except FileNotFoundError as e:
        print("FileNotFoundError during processing:", e)
        return render_template('index.html', content="Processing failed: required binary not found (ffmpeg?).")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors='ignore') if hasattr(e, 'stderr') else str(e)
        print("Subprocess failed:", stderr)
        return render_template('index.html', content=f"Processing failed: {stderr}")
    except Exception as e:
        print("Processing error:", str(e))
        return render_template('index.html', content=f"Error processing video: {str(e)}")
    finally:
        # 6) cleanup temp files (best-effort)
        for p in (tmp.name, upload_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as ex:
                print(f"Could not remove file {p}: {ex}")

    return render_template('index.html', content=f"The uploaded video is predicted as: {prediction}")

# --- end chunk ---
