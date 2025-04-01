from flask import Flask, render_template_string, request, send_from_directory
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import base64
from preprocessing import preprocess_images, extract_features
from model_training import load_model
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
model = load_model('/Users/dalm1/Desktop/reroll/Progra/par20/results/best_traditional_model.pkl')

# Load the PCA model and scaler used during training
# Load preprocessing components with error handling
try:
    scaler = load_model('/Users/dalm1/Desktop/reroll/Progra/par20/results/scaler.pkl')
    pca = load_model('/Users/dalm1/Desktop/reroll/Progra/par20/results/pca_model.pkl')
except FileNotFoundError:
    raise Exception("Preprocessing components missing! Run model training first.")

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>PneumoScan Web</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        .result-box { margin: 20px; padding: 20px; border: 2px solid #4CAF50; border-radius: 5px; }
        .plot-img { max-width: 600px; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>Diagnostic d'Imagerie Médicale</h1>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <button type="submit">Analyser</button>
    </form>

    {% if result %}
    <div class="result-box">
        <h2>Résultats :</h2>
        <p><strong>Prédiction :</strong> {{ prediction }}</p>
        <p><strong>Confiance :</strong> {{ confidence }}%</p>
        <img class="plot-img" src="data:image/png;base64,{{ plot_url }}" alt="Visualisation 3D">
    </div>
    {% endif %}
</body>
</html>
'''

def create_3d_plot(confidence):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    theta = np.linspace(0, 2*np.pi, 100)
    z = np.linspace(0, confidence, 100)
    theta, z = np.meshgrid(theta, z)
    x = z * np.cos(theta)
    y = z * np.sin(theta)

    ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)
    ax.set_zlim(0, 1)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result = None
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'Aucun fichier sélectionné'

        file = request.files['image']
        if file.filename == '':
            return 'Aucun fichier sélectionné'

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        processed_img = preprocess_images([filepath])

        n_samples = len(processed_img)
        flat_img = processed_img.reshape(n_samples, -1)
        scaled_img = scaler.transform(flat_img)
        pca_features = pca.transform(scaled_img)

        probability = model.predict_proba(pca_features)[0][1]

        result = {
            'prediction': 'PNEUMONIE' if probability > 0.5 else 'NORMAL',
            'confidence': f"{probability*100:.2f}",
            'plot_url': create_3d_plot(probability)
        }

    return render_template_string(HTML_TEMPLATE, result=result)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(port=5000, debug=True)
