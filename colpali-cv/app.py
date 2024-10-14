from flask import Flask, request, jsonify
import os

from colpali.utility.main import main

app = Flask(__name__)

API_PREFIX = "/api/colpali"

UPLOAD_FOLDER = "/Users/aditya.narayan/Desktop/ColPali-CV_Parsing/colpali-cv/colpali/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check Health of API
@app.route(API_PREFIX + "/health", methods=["GET"])
def health_check():
    return "ok", 200

# Upload PNG File
@app.route(API_PREFIX + "/upload", methods=["POST"])
def upload_img():
    if "file" not in request.files:
        return jsonify({"error":"No file part"}), 400
    
    file = request.files["file"]

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not file.filename.endswith('.png'):
        return jsonify({"error": "Upload .PNG file"}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    return jsonify({"message": "File uploaded successfully", "filename": file.filename}), 201

# Get info of uploaded files.
@app.route(API_PREFIX + "/info", methods=["GET"])
def info():
    temp_counter = -1
    filename_dict = {"total_uploads": 0, "files":{}}
    for idx, file in enumerate(os.listdir(UPLOAD_FOLDER)):
        if file.endswith('.png'):
            temp_counter += 1
            filename_dict["files"][idx] = file
            filename_dict["total_uploads"] = temp_counter
    return jsonify(filename_dict)

# Start processing
@app.route(API_PREFIX + "/process/<string:data>", methods=["POST"])
def process(data):
    inference = dict(main(data))
    return jsonify({"status":"Query Received", "Query": data, "Inference": inference}), 200



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)