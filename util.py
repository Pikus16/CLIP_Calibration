import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import norm
import torch.nn as nn
from torchvision import datasets
import torch.optim as optim

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
    return cifar_val

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

def get_text_probs(model, tokenizer, val_dset,
        text_template='{}', batch_size=128, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

    


def find_temp_scale(model, tokenizer, val_dset, num_classes=100, text_template='{}', batch_size=128,
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    ## Get all text features
    text = tokenizer([text_template.replace('{}',x) for x in val_dset.classes])
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text.to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)

    ## Get all image features
    val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=False)
    all_val_img_features = None
    for i, data in enumerate(val_loader, 0):
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
    q = 1 - (0.83 / num_classes)
    return find_temp_scale_with_q(q, text_probs, device), text_probs


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