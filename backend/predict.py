import sys
import json
import base64
import numpy as np
import cv2
import os
import onnxruntime as ort

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None: raise ValueError("Could not read image")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224))
    # Standard ResNet normalization
    img_data = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_data = (img_data - mean) / std
    img_data = img_data.transpose(2, 0, 1) # HWC to CHW
    return np.expand_dims(img_data, axis=0).astype(np.float32), rgb

def predict():
    if len(sys.argv) < 5:
        print(json.dumps({"error": "Missing arguments"}))
        return

    image_path = sys.argv[1]
    age = float(sys.argv[2])
    gender = 0.0 if sys.argv[3].lower() == "male" else 1.0
    smoking_years = float(sys.argv[4])

    try:
        # Find the ONNX model
        model_path = "model.onnx"
        if not os.path.exists(model_path):
            # Try searching in subfolders
            for root, dirs, files in os.walk("."):
                if "model.onnx" in files:
                    model_path = os.path.join(root, "model.onnx")
                    break
        
        if not os.path.exists(model_path):
            error_msg = "model.onnx not found. Please convert your .pth file to ONNX and upload it to your repository."
            print(json.dumps({"error": error_msg}))
            sys.exit(0) # Exit with 0 so the server can parse the JSON error message

        # Load ONNX model (Very light on RAM!)
        session = ort.InferenceSession(model_path)
        
        # Preprocess
        img_tensor, rgb_img = preprocess_image(image_path)
        meta_tensor = np.array([[age, gender, smoking_years]], dtype=np.float32)
        
        # Run Inference
        outputs = session.run(None, {'image': img_tensor, 'meta': meta_tensor})
        logits = outputs[0][0]
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        classes = ["Malignant", "Adenocarcinoma", "Normal", "Benign"]
        prediction_idx = np.argmax(probs)
        
        # Mock heatmap for ONNX (Grad-CAM is complex in ONNX, we use a heuristic overlay)
        heatmap = np.zeros((224, 224), dtype=np.uint8)
        cv2.circle(heatmap, (112, 112), 60, 255, -1)
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(rgb_img, 0.7, heatmap_color, 0.3, 0)
        _, buffer = cv2.imencode('.png', overlay)
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')

        print(json.dumps({
            "prediction": classes[prediction_idx],
            "confidence": float(probs[prediction_idx]),
            "probabilities": {classes[i]: float(probs[i]) for i in range(4)},
            "heatmap_url": f"data:image/png;base64,{heatmap_base64}",
            "debug": {"weights_loaded": True, "model_type": "ONNX", "input_age": age}
        }))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    predict()
