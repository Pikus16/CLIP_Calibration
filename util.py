import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import norm
import torch.nn as nn
from torchvision import datasets
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def get_preds(model, tokenizer, dset, text_template='{}', temp_scaling=None, batch_size=128,
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    text = tokenizer([text_template.replace('{}',x) for x in dset.classes])
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text.to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)
    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=16)
    predictions = np.array([])
    actual = np.array([])
    probs = np.array([])
    for image, labels in dataloader:
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image.to(device))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            if temp_scaling is not None:
                image_features = torch.div(image_features, temp_scaling)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        vals, indics = text_probs.max(dim=1)
        predictions = np.append(predictions, indics.cpu().numpy())
        actual = np.append(actual, labels.numpy())
        probs = np.append(probs, vals.cpu().numpy())
    return predictions, actual, probs

def calc_bins(y_true, preds, confs, num_bins=10):
  # Assign each prediction to a bin
  bins = np.linspace(1.0 / num_bins, 1, num_bins)
  binned = np.digitize(confs, bins)

  # Save the accuracy, confidence and size of each bin
  bin_accs = np.zeros(num_bins)
  bin_confs = np.zeros(num_bins)
  bin_sizes = np.zeros(num_bins)

  for bin in range(num_bins):
    bin_sizes[bin] = len(preds[binned == bin])
    if bin_sizes[bin] > 0:
      bin_accs[bin] = np.mean(y_true[binned==bin] == preds[binned == bin])
      bin_confs[bin] = np.mean(confs[binned==bin])

  return bins, binned, bin_accs, bin_confs, bin_sizes

def get_val_set(dataset_name, classes, val_transform):
    if dataset_name == "CIFAR10":
        cifar_train = datasets.CIFAR10('/home/ubuntu/data/', train = True, transform = val_transform, download=True)
    elif dataset_name == "CIFAR100":
        cifar_train = datasets.CIFAR100('/home/ubuntu/data/', train = True, transform = val_transform, download=True)
    else:
        raise ValueError(f"{dataset_name} unknown")
    np.random.seed(0)
    cifar_val = (torch.utils.data.Subset(cifar_train, np.random.randint(0, len(cifar_train), 10000)))
    cifar_val.classes = classes
    cifar_val.targets = cifar_val.dataset.targets
    return cifar_val

def get_test_set(dataset_name, test_transform):
    if dataset_name == "CIFAR10":
        cifar_test = datasets.CIFAR10('/home/ubuntu/data/', train = False, transform = test_transform, download=True)
        num_classes = 10
    elif dataset_name == "CIFAR100":
        cifar_test = datasets.CIFAR100('/home/ubuntu/data/', train = False, transform = test_transform, download=True)
        num_classes = 100
    else:
        raise ValueError(f"{dataset_name} unknown")
    classes, _ = get_openai_prompts(dataset_name)
    cifar_test.classes = classes
    return cifar_test, num_classes

def get_overconfident_oce(y_true, preds, confs):
  ECE = 0
  MCE = 0
  bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(y_true, preds, confs)

  for i in range(len(bins)):
    overconf_conf_dif = min(0, bin_confs[i] - bin_accs[i])
    ECE += (bin_sizes[i] / sum(bin_sizes)) * overconf_conf_dif
    MCE = max(MCE, overconf_conf_dif)

  overall_acc = np.mean(y_true == preds)
  return ECE, MCE, overall_acc

def get_metrics(y_true, preds, confs):
  ECE = 0
  MCE = 0
  bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(y_true, preds, confs)

  for i in range(len(bins)):
    abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
    ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
    MCE = max(MCE, abs_conf_dif)

  overall_acc = np.mean(y_true == preds)
  return ECE, MCE, overall_acc

def sample_quantile(min, max, step, val_text_probs, device, model, tokenizer, test_set, text_template):
    temps_learned = []
    eces_learned = []
    x_axis = np.arange(min, max, step)
    for q in (x_axis):
        scaled_temp = find_temp_scale_with_q(q, val_text_probs, device)
        temps_learned.append(scaled_temp)
        predictions, actual, probs = get_preds(model, tokenizer, test_set, text_template=text_template, temp_scaling=scaled_temp, device=device)
        ece, mce, acc = get_metrics(predictions, actual, probs)
        eces_learned.append(ece)
    return temps_learned, eces_learned

def T_scaling(logits, args):
  temperature = args.get('temperature', None)
  return torch.div(logits, temperature)

def find_temp_scale_with_q(q, text_probs, device):
    ## Get threshold
    text_probs_numpy = text_probs.softmax(dim=-1).cpu().numpy()
    
    ## Setup LBGFS
    temperature = nn.Parameter((torch.ones(1)).to(device))
    args = {'temperature': temperature}
    criterion = nn.CrossEntropyLoss()

    # Removing strong_wolfe line search results in jump after 50 epochs
    optimizer = optim.LBFGS([temperature], lr=0.001, max_iter=1000, line_search_fn='strong_wolfe')

    logits_list = []
    labels_list = []
    confidences_of_js = np.sort(text_probs_numpy, axis=0)
    added = np.zeros(len(text_probs_numpy))
    for j in range(len(text_probs_numpy[0])):
        confidences_of_j = confidences_of_js[:,j]
        thresh = confidences_of_j[int(len(confidences_of_j) * q)] 
        for k in range(len(text_probs_numpy)):
            if text_probs_numpy[k][j] >= thresh:
                added[k] += 1
                logits_list.append(text_probs[k].unsqueeze(0))
                labels_list.append(j)

    logits_list = torch.cat(logits_list).to(device) # [len(dset), 100]
    labels_list = torch.FloatTensor(labels_list).to(device).long()
    
    temps = []
    losses = []
    def _eval():
        loss = criterion(T_scaling(logits_list, args), labels_list)
        loss.backward()
        temps.append(temperature.item())
        losses.append(loss)
        return loss

    optimizer.step(_eval)
    return temperature.item()#, temps, losses, added

def get_text_probs(model, tokenizer, dset,
        text_template='{}', batch_size=128, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    ## Get all text features
    text = tokenizer([text_template.replace('{}',x) for x in dset.classes])
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text.to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)

    ## Get all image features
    loader = DataLoader(dset, batch_size=batch_size, shuffle=False)
    all_val_img_features = None
    for i, data in enumerate(loader, 0):
        images = data[0].to(device)
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            if all_val_img_features is None:
                all_val_img_features = image_features
            else:
                all_val_img_features = torch.concat([all_val_img_features, image_features])
    
    ## Get text probs
    text_probs = (100.0 * all_val_img_features @ text_features.T)
    return text_probs


def find_temp_scale(model, tokenizer, val_dset, num_classes=100, text_template='{}', batch_size=128,
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

    text_probs = get_text_probs(
        model=model,
        tokenizer=tokenizer,
        dset=val_dset,
        text_template=text_template,
        batch_size=batch_size,
        device=device
    )
    def func(n_classes):
        return 1 - (1.25 * np.log10(n_classes)) / n_classes
    q = func(num_classes)

    return find_temp_scale_with_q(q, text_probs, device)


def get_openai_prompts(dset_name):
    if dset_name == 'CIFAR10':
        classes = [
            'airplane',
            'automobile',
            'bird',
            'cat',
            'deer',
            'dog',
            'frog',
            'horse',
            'ship',
            'truck',
        ]

        templates = [
            '{}',
            'a photo of a {}.',
            'a blurry photo of a {}.',
            'a black and white photo of a {}.',
            'a low contrast photo of a {}.',
            'a high contrast photo of a {}.',
            'a bad photo of a {}.',
            'a good photo of a {}.',
            'a photo of a small {}.',
            'a photo of a big {}.',
            'a photo of the {}.',
            'a blurry photo of the {}.',
            'a black and white photo of the {}.',
            'a low contrast photo of the {}.',
            'a high contrast photo of the {}.',
            'a bad photo of the {}.',
            'a good photo of the {}.',
            'a photo of the small {}.',
            'a photo of the big {}.',
        ]
    elif dset_name == 'CIFAR100':
        classes = [
            'apple',
            'aquarium fish',
            'baby',
            'bear',
            'beaver',
            'bed',
            'bee',
            'beetle',
            'bicycle',
            'bottle',
            'bowl',
            'boy',
            'bridge',
            'bus',
            'butterfly',
            'camel',
            'can',
            'castle',
            'caterpillar',
            'cattle',
            'chair',
            'chimpanzee',
            'clock',
            'cloud',
            'cockroach',
            'couch',
            'crab',
            'crocodile',
            'cup',
            'dinosaur',
            'dolphin',
            'elephant',
            'flatfish',
            'forest',
            'fox',
            'girl',
            'hamster',
            'house',
            'kangaroo',
            'keyboard',
            'lamp',
            'lawn mower',
            'leopard',
            'lion',
            'lizard',
            'lobster',
            'man',
            'maple tree',
            'motorcycle',
            'mountain',
            'mouse',
            'mushroom',
            'oak tree',
            'orange',
            'orchid',
            'otter',
            'palm tree',
            'pear',
            'pickup truck',
            'pine tree',
            'plain',
            'plate',
            'poppy',
            'porcupine',
            'possum',
            'rabbit',
            'raccoon',
            'ray',
            'road',
            'rocket',
            'rose',
            'sea',
            'seal',
            'shark',
            'shrew',
            'skunk',
            'skyscraper',
            'snail',
            'snake',
            'spider',
            'squirrel',
            'streetcar',
            'sunflower',
            'sweet pepper',
            'table',
            'tank',
            'telephone',
            'television',
            'tiger',
            'tractor',
            'train',
            'trout',
            'tulip',
            'turtle',
            'wardrobe',
            'whale',
            'willow tree',
            'wolf',
            'woman',
            'worm',
        ]

        templates = [
            '{}',
            'a photo of a {}.',
            'a blurry photo of a {}.',
            'a black and white photo of a {}.',
            'a low contrast photo of a {}.',
            'a high contrast photo of a {}.',
            'a bad photo of a {}.',
            'a good photo of a {}.',
            'a photo of a small {}.',
            'a photo of a big {}.',
            'a photo of the {}.',
            'a blurry photo of the {}.',
            'a black and white photo of the {}.',
            'a low contrast photo of the {}.',
            'a high contrast photo of the {}.',
            'a bad photo of the {}.',
            'a good photo of the {}.',
            'a photo of the small {}.',
            'a photo of the big {}.',
        ]
    else:
        raise ValueError(f"{dset_name} not supported")
    return classes, templates

def draw_reliability_graph(y_true, preds, confs, title=None):
  ECE, MCE, overall_acc = get_metrics(y_true, preds, confs)
  bins, _, bin_accs, _, _ = calc_bins(y_true, preds, confs)

  fig = plt.figure(figsize=(8, 8))
  ax = fig.gca()

  # x/y limits
  ax.set_xlim(0, 1.05)
  ax.set_ylim(0, 1)

  # x/y labels
  plt.xlabel('Confidence')
  plt.ylabel('Accuracy')

  # Create grid
  ax.set_axisbelow(True) 
  ax.grid(color='gray', linestyle='dashed')

  # Error bars
  plt.bar(bins, bins,  width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')

  # Draw bars and identity line
  plt.bar(bins, bin_accs, width=0.1, alpha=0.7, edgecolor='red', color='purple')
  plt.bar(bins, np.minimum(bins,bin_accs), width=0.1, alpha=1, edgecolor='black', color='b')

  plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)

  # Equally spaced axes
  plt.gca().set_aspect('equal', adjustable='box')

  # ECE and MCE legend
  ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
  #MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
  #acc_patch = mpatches.Patch(color='orange', label='Overall Accuracy = {:.2f}%'.format(overall_acc*100))
  plt.legend(handles=[ECE_patch])
  if title is not None:
    plt.title(title)