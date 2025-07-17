from flask import Flask, render_template, request, jsonify, url_for
import torch
import torchaudio
import os
import torch.nn as nn
from pydub import AudioSegment

app = Flask(__name__, static_folder='static') # ✅ Specify static folder
os.makedirs("uploads", exist_ok=True)

##########################################
labels = [
    "chainsaw", "clock_tick", "crackling_fire", "crying_baby", "dog",
    "helicopter", "rain", "rooster", "sea_waves", "sneezing"
]

##########################################
# Model Definition (ACDNet)
##########################################

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class SFEB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, 8, 1),
            nn.ReLU(),
            nn.Conv1d(8, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        se_weight = self.se(out)
        return out * se_weight

class TFEB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gru = nn.GRU(channels, channels, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(channels*2, channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        out = self.linear(out)
        return out.permute(0, 2, 1)

class ACDNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1)
        )
        self.layer2 = ResidualBlock(64, 128, 2)
        self.sfeb = SFEB(128)
        self.layer3 = ResidualBlock(128, 256, 2)
        self.tfeb = TFEB(256)
        self.layer4 = ResidualBlock(256, 512, 2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.sfeb(x)
        x = self.layer3(x)
        x = self.tfeb(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

##########################################
# Load pruned model weights
##########################################

MODEL_PATH = "checkpoints/acdnet_pruned_fold5.pth"
model = ACDNet(num_classes=10)
raw_state_dict = torch.load(MODEL_PATH, map_location='cpu')

clean_state_dict = {}
for k, v in raw_state_dict.items():
    if k.endswith("_orig"):
        base = k[:-5]
        mask = raw_state_dict[base + "_mask"]
        clean_state_dict[base] = v * mask
    elif k.endswith("_mask"):
        continue
    else:
        clean_state_dict[k] = v

model.load_state_dict(clean_state_dict)
model.eval()

##########################################
# Preprocessing and Flask Routes
##########################################

def preprocess_audio(filepath):
    waveform, sr = torchaudio.load(filepath)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 20000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=20000)
        waveform = resampler(waveform)

    SEGMENT_LENGTH = 30225
    L = waveform.shape[-1]
    if L >= SEGMENT_LENGTH:
        waveform = waveform[..., :SEGMENT_LENGTH]
    else:
        pad = SEGMENT_LENGTH - L
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    return waveform.unsqueeze(0)

@app.route("/")
def index():
    return render_template("index.html")

# In your app.py, find the @app.route("/predict") function and REPLACE it with this:

@app.route("/predict", methods=["POST"])
def predict():
    print("\n--- A new prediction request has started! ---") # New print statement

    if "file" not in request.files:
        print("ERROR: 'file' not in request.files") # New print statement
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == '':
        print("ERROR: No file selected for uploading") # New print statement
        return jsonify({"error": "No file selected for uploading"}), 400

    upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
    original_path = os.path.join(upload_dir, file.filename)
    
    print(f"Step 1: Saving the original uploaded file to: {original_path}") # New print statement
    file.save(original_path)

    # We will now try to convert the file.
    try:
        converted_path = os.path.join(upload_dir, "converted.wav")
        print(f"Step 2: Attempting to convert '{original_path}' to '{converted_path}' using pydub/ffmpeg.") # New print statement
        
        # Use pydub to load the original file and export it as WAV
        audio = AudioSegment.from_file(original_path)
        audio.export(converted_path, format="wav")
        
        print("Step 3: Conversion successful! The file 'converted.wav' has been created.") # New print statement

        # Now, preprocess the *converted* WAV file
        print(f"Step 4: Passing the converted file '{converted_path}' to the model for preprocessing.") # New print statement
        input_tensor = preprocess_audio(converted_path)
        
        print("Step 5: Making prediction with the model.") # New print statement
        with torch.no_grad():
            output = model(input_tensor)
            pred_idx = output.argmax(dim=1).item()
            predicted_label = labels[pred_idx]
        
        print(f"SUCCESS: Prediction is '{predicted_label}'") # New print statement
        return jsonify({"prediction": predicted_label})
        
    except Exception as e:
        # This will now print the REAL error to your terminal!
        print(f"\n!!!!!!!! AN ERROR OCCURRED !!!!!!!!")
        print(f"The error is: {str(e)}")
        print("This likely means ffmpeg is not installed correctly or not in your system's PATH.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        return jsonify({"error": f"Failed to process audio. Check the server terminal for details."}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use PORT env var or default to 5000
    app.run(host='0.0.0.0', port=port)