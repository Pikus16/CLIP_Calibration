import open_clip
from torchvision import datasets, transforms
import torch
from PIL import Image
import open_clip
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from sklearn import calibration
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch.nn as nn
import torch.optim as optim
from sklearn.tree import DecisionTreeRegressor as DT
from sklearn.isotonic import IsotonicRegression as IR
from scipy.stats import norm
from tqdm import tqdm
import pandas as pd

from util import get_preds, calc_bins, get_metrics, T_scaling, find_temp_scale, get_test_set, get_openai_prompts, get_val_set, find_temp_scale_with_q, get_text_probs, sample_quantile, sample_quantile_unpredicted, get_image_features, get_preds_from_img_features, run_modified_uts

device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
batch_size = 128

model_name = 'ViT-B-16'
pretrained_dset = 'laion400m_e31'

model, _, preprocess = open_clip.create_model_and_transforms(model_name,
    pretrained=pretrained_dset,
    device=device)
tokenizer = open_clip.get_tokenizer(model_name)

imagenet_test = datasets.ImageFolder(f'/home/ubuntu/data/Imagenet/ILSVRC/Data/CLS-LOC/val/', transform=preprocess)
text_template = 'a photo of a {}.'

## Setup LBGFS
temperature = nn.Parameter((torch.ones(1)).to(device))
args = {'temperature': temperature}
criterion = nn.CrossEntropyLoss()

# Removing strong_wolfe line search results in jump after 50 epochs
optimizer = optim.LBFGS([temperature], lr=0.001, max_iter=1000, line_search_fn='strong_wolfe')

dataloader = DataLoader(imagenet_test, batch_size=batch_size, shuffle=False, num_workers=16)

logits_list = []
labels_list = []
with torch.no_grad():
    for i, (input, label) in enumerate(tqdm(dataloader, total=len(imagenet_test) // batch_size)):
        input = input.to(device)
        image_features = model.encode_image(input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits_list.append(image_features)
        labels_list.append(label)
        #if i > 100:
        #    break
    logits = torch.cat(logits_list).to(device)
    labels = torch.cat(labels_list).to(device)

temps = []
losses = []
def _eval():
    loss = criterion(T_scaling(logits, args), labels)
    loss.backward()
    temps.append(temperature.item())
    losses.append(loss)
    return loss
optimizer.step(_eval)
temperature.item()
