import argparse
import io
from PIL import Image
import os
import cv2
from flask import Flask, render_template, request, redirect, url_for

from ultralytics import YOLO

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if 'file' in request.files:
            try:
                f = request.files['file']
                upload_folder = app.config['UPLOAD_FOLDER']
                if not os.path.exists(upload_folder):
                    os.makedirs(upload_folder)
                filepath = os.path.join(upload_folder, f.filename)
                print("Upload folder is", filepath)
                f.save(filepath)

                file_extension = f.filename.rsplit('.', 1)[1].lower()

                if file_extension in ['jpg', 'jpeg', 'png']:
                    img = cv2.imread(filepath)
                    is_success, buffer = cv2.imencode(".jpg", img)
                    frame = buffer.tobytes()

                    image = Image.open(io.BytesIO(frame))

                    yolo = YOLO('best.pt')
                    results = yolo(image)

                    # Count the number of trees detected
                    tree_count = len(results[0].boxes)

                    # Save the result image
                    result_img_paths = []
                    for i, result in enumerate(results):
                        result_img_path = os.path.join(upload_folder, f"result_{i}_{f.filename}")
                        result.save(result_img_path)
                        result_img_paths.append(f"uploads/result_{i}_{f.filename}")

                    return render_template('result.html', original_image=f"uploads/{f.filename}", result_images=result_img_paths, tree_count=tree_count)
                else:
                    print("Unsupported file extension")
                    return "Unsupported file extension", 400
            except Exception as e:
                print("Error during prediction:", str(e))
                return "Error during prediction: " + str(e), 500
    
    return render_template('upload.html')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing YOLO models for tree detection")
    parser.add_argument("--port", default=5000, type=int, help="Port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=True)
