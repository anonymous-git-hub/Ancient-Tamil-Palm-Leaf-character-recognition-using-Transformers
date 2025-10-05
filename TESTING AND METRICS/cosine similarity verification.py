import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity

# Settings
DATASET_ROOT = r"C:\D\AK Repo\TEMP NIT NLP\COMPLETE_DATASET\Final DATASET"  # Folder with generated images
NUM_CLASSES = 75
NUM_TEST_IMAGES = 10
IMG_SIZE = 224  # Embedding model input size

# Preprocessing for embedding model
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load pre-trained embedding model (ResNet18, remove classifier head)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove last FC layer
model.eval()
model.to(device)

def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(img_tensor).squeeze().cpu().numpy().flatten()
    return emb

def test_similarity_for_class(class_id):
    class_dir = os.path.join(DATASET_ROOT, str(class_id))
    images = [f for f in os.listdir(class_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    images = images[:NUM_TEST_IMAGES]
    if len(images) < NUM_TEST_IMAGES:
        print(f"Class {class_id}: Not enough images, found {len(images)}")
        return None
    embeddings = [get_embedding(os.path.join(class_dir, img)) for img in images]
    sim_matrix = cosine_similarity(embeddings)
    # Average off-diagonal similarity
    avg_sim = (np.sum(sim_matrix) - np.trace(sim_matrix)) / (NUM_TEST_IMAGES**2 - NUM_TEST_IMAGES)
    return avg_sim

if __name__ == "__main__":
    results = []
    for class_id in range(NUM_CLASSES):
        avg_sim = test_similarity_for_class(class_id)
        if avg_sim is not None:
            print(f"Class {class_id:02d}: Avg similarity = {avg_sim:.4f}")
            results.append({"Class ID": class_id, "Avg Similarity": avg_sim})
        else:
            results.append({"Class ID": class_id, "Avg Similarity": None})

    # Save to Excel
    df = pd.DataFrame(results)
    df.to_excel("similarity_report.xlsx", index=False)
    print("\nSimilarity report saved to similarity_report.xlsx")


output:
Python 3.12.6 (tags/v3.12.6:a4a2d2b, Sep  6 2024, 20:11:23) [MSC v.1940 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.

= RESTART: C:\D\AK Repo\Ancient Tamil Palm Leaf character recognition using Transformers\cosine similarity verification.py

Warning (from warnings module):
  File "C:\Users\ASWIN\AppData\Local\Programs\Python\Python312\Lib\site-packages\torchvision\models\_utils.py", line 208
    warnings.warn(
UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.

Warning (from warnings module):
  File "C:\Users\ASWIN\AppData\Local\Programs\Python\Python312\Lib\site-packages\torchvision\models\_utils.py", line 223
    warnings.warn(msg)
UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to C:\Users\ASWIN/.cache\torch\hub\checkpoints\resnet18-f37072fd.pth

  0%|          | 0.00/44.7M [00:00<?, ?B/s]
  1%|          | 384k/44.7M [00:00<00:13, 3.37MB/s]
  2%|▏         | 1.00M/44.7M [00:00<00:11, 4.02MB/s]
  6%|▌         | 2.50M/44.7M [00:00<00:05, 8.38MB/s]
  8%|▊         | 3.75M/44.7M [00:00<00:04, 9.82MB/s]
 11%|█         | 4.88M/44.7M [00:00<00:04, 10.4MB/s]
 15%|█▌        | 6.75M/44.7M [00:00<00:03, 11.5MB/s]
 22%|██▏       | 9.88M/44.7M [00:00<00:02, 15.4MB/s]
 30%|██▉       | 13.4M/44.7M [00:00<00:01, 20.8MB/s]
 35%|███▍      | 15.5M/44.7M [00:01<00:01, 21.1MB/s]
 41%|████▏     | 18.5M/44.7M [00:01<00:01, 23.6MB/s]
 48%|████▊     | 21.2M/44.7M [00:01<00:00, 25.0MB/s]
 54%|█████▍    | 24.1M/44.7M [00:01<00:00, 26.0MB/s]
 61%|██████▏   | 27.4M/44.7M [00:01<00:00, 28.3MB/s]
 68%|██████▊   | 30.5M/44.7M [00:01<00:00, 29.4MB/s]
 76%|███████▌  | 33.9M/44.7M [00:01<00:00, 30.7MB/s]
 83%|████████▎ | 36.9M/44.7M [00:01<00:00, 27.9MB/s]
 89%|████████▊ | 39.6M/44.7M [00:02<00:00, 24.0MB/s]
 94%|█████████▍| 42.1M/44.7M [00:02<00:00, 23.6MB/s]
100%|█████████▉| 44.5M/44.7M [00:02<00:00, 23.3MB/s]
100%|██████████| 44.7M/44.7M [00:02<00:00, 19.1MB/s]
Class 00: Avg similarity = 0.8630
Class 01: Avg similarity = 0.9334
Class 02: Avg similarity = 0.8837
Class 03: Avg similarity = 0.9233
Class 04: Avg similarity = 0.9185
Class 05: Avg similarity = 0.9152
Class 06: Avg similarity = 0.8950
Class 07: Avg similarity = 0.9305
Class 08: Avg similarity = 0.9202
Class 09: Avg similarity = 0.8990
Class 10: Avg similarity = 0.9406
Class 11: Avg similarity = 0.8695
Class 12: Avg similarity = 0.9192
Class 13: Avg similarity = 0.9194
Class 14: Avg similarity = 0.8740
Class 15: Avg similarity = 0.9150
Class 16: Avg similarity = 0.8846
Class 17: Avg similarity = 0.7917
Class 18: Avg similarity = 0.9286
Class 19: Avg similarity = 0.7850
Class 20: Avg similarity = 0.7132
Class 21: Avg similarity = 0.8660
Class 22: Avg similarity = 0.8951
Class 23: Avg similarity = 0.8903
Class 24: Avg similarity = 0.8962
Class 25: Avg similarity = 0.8881
Class 26: Avg similarity = 0.7498
Class 27: Avg similarity = 0.7789
Class 28: Avg similarity = 0.8933
Class 29: Avg similarity = 0.7644
Class 30: Avg similarity = 0.9126
Class 31: Avg similarity = 0.8781
Class 32: Avg similarity = 0.7756
Class 33: Avg similarity = 0.7449
Class 34: Avg similarity = 0.8587
Class 35: Avg similarity = 0.7589
Class 36: Avg similarity = 0.8916
Class 37: Avg similarity = 0.7131
Class 38: Avg similarity = 0.9057
Class 39: Avg similarity = 0.8849
Class 40: Avg similarity = 0.9146
Class 41: Avg similarity = 0.7063
Class 42: Avg similarity = 0.8929
Class 43: Avg similarity = 0.7619
Class 44: Avg similarity = 0.9021
Class 45: Avg similarity = 0.7559
Class 46: Avg similarity = 0.8987
Class 47: Avg similarity = 0.8172
Class 48: Avg similarity = 0.8440
Class 49: Avg similarity = 0.8257
Class 50: Avg similarity = 0.8801
Class 51: Avg similarity = 0.8734
Class 52: Avg similarity = 0.9004
Class 53: Avg similarity = 0.7970
Class 54: Avg similarity = 0.8926
Class 55: Avg similarity = 0.8651
Class 56: Avg similarity = 0.8862
Class 57: Avg similarity = 0.7558
Class 58: Avg similarity = 0.8366
Class 59: Avg similarity = 0.7486
Class 60: Avg similarity = 0.8788
Class 61: Avg similarity = 0.8893
Class 62: Avg similarity = 0.8014
Class 63: Avg similarity = 0.7773
Class 64: Avg similarity = 0.8882
Class 65: Avg similarity = 0.8104
Class 66: Avg similarity = 0.8468
Class 67: Avg similarity = 0.8739
Class 68: Avg similarity = 0.8207
Class 69: Avg similarity = 0.8968
Class 70: Avg similarity = 0.8061
Class 71: Avg similarity = 0.8992
Class 72: Avg similarity = 0.9052
Class 73: Avg similarity = 0.9004
Class 74: Avg similarity = 0.9049

Similarity report saved to similarity_report.xlsx

