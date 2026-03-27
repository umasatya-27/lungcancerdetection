import torch
import torch.nn as nn
import torchvision.models as models
import os
import shutil
import tempfile

# 1. Define the architecture (Matches your app)
class FusionModel(nn.Module):
    def __init__(self, num_classes=4):
        super(FusionModel, self).__init__()
        self.cnn = models.resnet50(weights=None)
        self.cnn.fc = nn.Identity()
        self.meta_branch = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Dropout(0.2))
        self.fc = nn.Sequential(nn.Linear(2048 + 64, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes))

    def forward(self, image, meta):
        img_features = self.cnn(image)
        meta_features = self.meta_branch(meta)
        combined = torch.cat((img_features, meta_features), dim=1)
        return self.fc(combined)

# 2. Find the model directory
model_dir = None
for root, dirs, files in os.walk("."):
    if "data.pkl" in files:
        model_dir = root
        break

if not model_dir:
    print("ERROR: Could not find model data (data.pkl)!")
    exit()

print(f"Found model data in: {model_dir}")

# 3. Create a PROPER ZIP (PyTorch requires files to be inside a subfolder in the zip)
temp_zip_dir = tempfile.mkdtemp()
archive_name = os.path.join(temp_zip_dir, "model_archive")

print("Creating proper archive for loading...")
shutil.make_archive(archive_name, 'zip', root_dir=os.path.dirname(model_dir), base_dir=os.path.basename(model_dir))
archive_path = archive_name + ".zip"

# 4. Load the model
model = FusionModel()
try:
    state_dict = torch.load(archive_path, map_location='cpu', weights_only=False)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'], strict=False)
    elif isinstance(state_dict, dict):
        model.load_state_dict(state_dict, strict=False)
    else:
        model = state_dict
    print("Successfully loaded model!")
except Exception as e:
    print(f"Loading failed: {e}")
    shutil.rmtree(temp_zip_dir)
    exit()

# Cleanup
shutil.rmtree(temp_zip_dir)
model.eval()

# 5. Export to ONNX
print("Exporting to model.onnx...")
dummy_img = torch.randn(1, 3, 224, 224)
dummy_meta = torch.randn(1, 3)

torch.onnx.export(
    model, 
    (dummy_img, dummy_meta), 
    "model.onnx", 
    input_names=['image', 'meta'], 
    output_names=['output'],
    opset_version=11
)

print("--------------------------------------------------")
print("SUCCESS! 'model.onnx' created.")
print("Now upload 'model.onnx' to your GitHub repository in the 'model/' folder.")
print("--------------------------------------------------")