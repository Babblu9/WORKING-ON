from flask import Flask, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
import subprocess
import shutil
import time
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'data/test'
RESULT_FOLDER = 'result/TOM/test/try-on'
SUCCESS_FOLDER = os.path.join(RESULT_FOLDER, 'success')  # New folder for successful outputs
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['SUCCESS_FOLDER'] = SUCCESS_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)
if not os.path.exists(SUCCESS_FOLDER):
    os.makedirs(SUCCESS_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(192, 256)):
    """Preprocess the image to the required size."""
    image = Image.open(image_path)
    image = image.resize(target_size, Image.ANTIALIAS)
    image.save(image_path)

@app.route('/try-on', methods=['POST'])
def try_on():
    if 'user_image' not in request.files or 'cloth_image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    user_image = request.files['user_image']
    cloth_image = request.files['cloth_image']

    if user_image.filename == '' or cloth_image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if user_image and allowed_file(user_image.filename) and cloth_image and allowed_file(cloth_image.filename):
        user_filename = secure_filename(user_image.filename)
        cloth_filename = secure_filename(cloth_image.filename)

        user_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'user', user_filename)
        cloth_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'cloth', cloth_filename)

        # Ensure the required directories exist
        if not os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], 'user')):
            os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'user'))
        if not os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], 'cloth')):
            os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'cloth'))

        user_image.save(user_image_path)
        cloth_image.save(cloth_image_path)

        # Preprocess the images
        preprocess_image(user_image_path)
        preprocess_image(cloth_image_path)

        # Generate the test_pairs.txt file
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'test_pairs.txt'), 'w') as f:
            f.write(f"{os.path.basename(user_image_path)} {os.path.basename(cloth_image_path)}\n")

        try:
            t = time.time()

            # Run the GMM stage
            subprocess.call(
                "python test.py --name GMM --stage GMM --workers 4 --datamode test --data_list data/test/test_pairs.txt --checkpoint checkpoints/GMM/gmm_final.pth",
                shell=True
            )

            warp_cloth = "result/GMM/test/warp-cloth"
            warp_mask = "result/GMM/test/warp-mask"
            shutil.copytree(warp_cloth, os.path.join(app.config['UPLOAD_FOLDER'], 'warp-cloth'), dirs_exist_ok=True)
            shutil.copytree(warp_mask, os.path.join(app.config['UPLOAD_FOLDER'], 'warp-mask'), dirs_exist_ok=True)

            # Run the TOM stage
            subprocess.call(
                "python test.py --name TOM --stage TOM --workers 4 --datamode test --data_list data/test/test_pairs.txt --checkpoint checkpoints/TOM/tom_final.pth",
                shell=True
            )

            print("TOTAL TIME:", time.time() - t)
            print("ALL PROCESS FINISHED")

            # Move the result image to the 'success' folder
            result_image_path = os.path.join(app.config['RESULT_FOLDER'], '00001.png')
            shutil.move(result_image_path, os.path.join(app.config['SUCCESS_FOLDER'], '00001.png'))

            # Send the result image as response
            success_image_path = os.path.join(app.config['SUCCESS_FOLDER'], '00001.png')
            return send_file(success_image_path, mimetype='image/png')

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
