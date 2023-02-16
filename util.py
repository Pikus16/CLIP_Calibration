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

def get_image_features(model, dset, batch_size=128,
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=16)
    actual = np.array([])
    all_img_features = None
    for i, (images, labels) in enumerate(dataloader, 0):
        images = images.to(device)
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            if all_img_features is None:
                all_img_features = image_features
            else:
                all_img_features = torch.concat([all_img_features, image_features])

        actual = np.append(actual, labels.numpy())
    return all_img_features, actual

def get_preds_from_img_features(model, tokenizer, dset, image_features, text_template='{}', temp_scaling=None,
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

    text = tokenizer([text_template.replace('{}',x) for x in dset.classes])
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text.to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)

        if temp_scaling is not None:
            image_features = torch.div(image_features, temp_scaling)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs, predictions = text_probs.max(dim=1)
    return predictions.cpu().numpy(), probs.cpu().numpy()


def run_modified_uts(image_features, z_score, text_template, model, tokenizer, dset, device):

    text = tokenizer([text_template.replace('{}',x) for x in dset.classes])
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text.to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T)

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

    predicted_label = text_probs_numpy.argmax(axis=1)

    added = np.zeros(len(text_probs_numpy[0]))
    for k in range(len(text_probs_numpy[0])):
        # set where predicted_label is not k
        logits_u = text_probs_numpy[predicted_label != k,:]

        mean_softmax_yk = logits_u[:,k].mean()
        std_softmax_yk = logits_u[:,k].std()

        thresh = mean_softmax_yk + z_score * std_softmax_yk
        for j in np.where(text_probs_numpy[:,k] >= thresh)[0]:
            added[k] += 1
            logits_list.append(text_probs[j].unsqueeze(0))
            labels_list.append(k)

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
    return temperature.item()

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
        dset = datasets.CIFAR10('/home/ubuntu/data/', train = True, transform = val_transform, download=True)
    elif dataset_name == "CIFAR100":
        dset = datasets.CIFAR100('/home/ubuntu/data/', train = True, transform = val_transform, download=True)
    elif dataset_name == 'Food101':
        dset = datasets.Food101('/home/ubuntu/data/Food101/', split='test', transform=val_transform,download=True)
    elif dataset_name == 'SUN397':
        dset = datasets.SUN397('/home/ubuntu/data/SUN397/',  transform=val_transform,download=True)
        np.random.seed(0)
        dataset_inds = np.random.permutation(np.arange(0,len(dset)))
        dset = torch.utils.data.Subset(dset, dataset_inds[:20000])
    elif dataset_name == 'DTD':
        dset = datasets.DTD('/home/ubuntu/data/DTD/', split='test', transform=val_transform, download=True)
    else:
        raise ValueError(f"{dataset_name} unknown")
    if dataset_name != 'SUN397':
        np.random.seed(0)
        dset_val = torch.utils.data.Subset(dset, np.random.randint(0, len(dset), 10000))
        if dataset_name != 'Food101' and dataset_name != 'DTD':
            dset_val.targets = dset_val.dataset.targets
        else:
            dset_val.targets = dset_val.dataset._labels

    else:
        dset_val = dset
    dset_val.classes = classes
    return dset_val

def get_test_set(dataset_name, test_transform):
    if dataset_name == "CIFAR10":
        dset = datasets.CIFAR10('/home/ubuntu/data/', train = False, transform = test_transform, download=True)
        num_classes = 10
    elif dataset_name == "CIFAR100":
        dset = datasets.CIFAR100('/home/ubuntu/data/', train = False, transform = test_transform, download=True)
        num_classes = 100
    elif dataset_name == 'Food101':
        dset = datasets.Food101('/home/ubuntu/data/Food101/', split='test', transform=test_transform,download=True)
        num_classes = 101
    elif dataset_name == 'SUN397':
        dset = datasets.SUN397('/home/ubuntu/data/SUN397/',  transform=test_transform,download=True)
        np.random.seed(0)
        dataset_inds = np.random.permutation(np.arange(0,len(dset)))
        dset = torch.utils.data.Subset(dset, dataset_inds[20000:])
        num_classes = 397
    elif dataset_name == 'DTD':
        dset = datasets.DTD('/home/ubuntu/data/DTD/', split='test', transform=test_transform,download=True)
        num_classes = len(dset.classes)
    else:
        raise ValueError(f"{dataset_name} unknown")
    classes, _ = get_openai_prompts(dataset_name)
    dset.classes = classes
    return dset, num_classes

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
    all_test_img_features, actual = get_image_features(
        model,  test_set,  batch_size=128, device=device
    )
    for q in tqdm(x_axis):
        scaled_temp = find_temp_scale_with_q(q, val_text_probs, device)
        temps_learned.append(scaled_temp)
        #predictions, actual, probs = get_preds(model, tokenizer, test_set, text_template=text_template, temp_scaling=scaled_temp, device=device)
        #predictions, actual, probs = get_preds(model, tokenizer, test_set, text_template=text_template, temp_scaling=scaled_temp, device=device)
        predictions, probs = get_preds_from_img_features(
            model, tokenizer, test_set, all_test_img_features, text_template=text_template, temp_scaling=scaled_temp, device = device
        )
        ece, mce, acc = get_metrics(predictions, actual, probs)
        eces_learned.append(ece)
    return temps_learned, eces_learned

def sample_quantile_unpredicted(min, max, step, val_text_probs, device, model, tokenizer, test_set, text_template):
    temps_learned = []
    eces_learned = []
    x_axis = np.arange(min, max, step)
    all_test_img_features, actual = get_image_features(
        model,  test_set,  batch_size=128, device=device
    )
    for q in tqdm(x_axis):
        scaled_temp = find_temp_scale_with_notpredictedd(q, val_text_probs, device)
        temps_learned.append(scaled_temp)
        #predictions, actual, probs = get_preds(model, tokenizer, test_set, text_template=text_template, temp_scaling=scaled_temp, device=device)
        predictions, probs = get_preds_from_img_features(
            model, tokenizer, test_set, all_test_img_features, text_template=text_template, temp_scaling=scaled_temp, device = device
        )
        ece, mce, acc = get_metrics(predictions, actual, probs)
        eces_learned.append(ece)
    return temps_learned, eces_learned


def T_scaling(logits, args):
  temperature = args.get('temperature', None)
  return torch.div(logits, temperature)

def find_temp_scale_with_notpredictedd(q, text_probs, device):
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
    #confidences_of_ks = np.sort(text_probs_numpy, axis=0)
    #added = np.zeros(len(text_probs_numpy))
    for k in range(len(text_probs_numpy[0])):
        confidences_of_k = np.sort(text_probs_numpy[np.where(text_probs_numpy.argmax(axis=1) != k)[0],k])
        thresh = confidences_of_k[int(len(confidences_of_k) * q)] 
        for j in range(len(text_probs_numpy)):
            if text_probs_numpy[j][k] >= thresh:
                #added[j] += 1
                logits_list.append(text_probs[j].unsqueeze(0))
                labels_list.append(k)
        '''for j in np.where(text_probs_numpy[:,k] >= thresh)[0]:
            #added[j] += 1
            logits_list.append(text_probs[j].unsqueeze(0))
            labels_list.append(k)'''

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

def run_supervised_tempscaling(model, tokenizer, dset, text_template, device):
    image_features, actual = get_image_features(model, dset, batch_size=128,
        device = device)
    actual = torch.IntTensor(actual).to(device).long()

    text = tokenizer([text_template.replace('{}',x) for x in dset.classes])
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text.to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
    text_probs = (100.0 * image_features @ text_features.T)

    ## Setup LBGFS
    temperature = nn.Parameter((torch.ones(1)).to(device))
    args = {'temperature': temperature}
    criterion = nn.CrossEntropyLoss()

    # Removing strong_wolfe line search results in jump after 50 epochs
    optimizer = optim.LBFGS([temperature], lr=0.001, max_iter=1000, line_search_fn='strong_wolfe')

    temps = []
    losses = []
    def _eval():
        loss = criterion(T_scaling(text_probs, args), actual)
        loss.backward()
        temps.append(temperature.item())
        losses.append(loss)
        return loss
    optimizer.step(_eval)
    return temperature.item()

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
            #'{}',
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
            #'{}',
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
    elif dset_name == 'Food101':
        classes = [
            'apple pie',
            'baby back ribs',
            'baklava',
            'beef carpaccio',
            'beef tartare',
            'beet salad',
            'beignets',
            'bibimbap',
            'bread pudding',
            'breakfast burrito',
            'bruschetta',
            'caesar salad',
            'cannoli',
            'caprese salad',
            'carrot cake',
            'ceviche',
            'cheese plate',
            'cheesecake',
            'chicken curry',
            'chicken quesadilla',
            'chicken wings',
            'chocolate cake',
            'chocolate mousse',
            'churros',
            'clam chowder',
            'club sandwich',
            'crab cakes',
            'creme brulee',
            'croque madame',
            'cup cakes',
            'deviled eggs',
            'donuts',
            'dumplings',
            'edamame',
            'eggs benedict',
            'escargots',
            'falafel',
            'filet mignon',
            'fish and chips',
            'foie gras',
            'french fries',
            'french onion soup',
            'french toast',
            'fried calamari',
            'fried rice',
            'frozen yogurt',
            'garlic bread',
            'gnocchi',
            'greek salad',
            'grilled cheese sandwich',
            'grilled salmon',
            'guacamole',
            'gyoza',
            'hamburger',
            'hot and sour soup',
            'hot dog',
            'huevos rancheros',
            'hummus',
            'ice cream',
            'lasagna',
            'lobster bisque',
            'lobster roll sandwich',
            'macaroni and cheese',
            'macarons',
            'miso soup',
            'mussels',
            'nachos',
            'omelette',
            'onion rings',
            'oysters',
            'pad thai',
            'paella',
            'pancakes',
            'panna cotta',
            'peking duck',
            'pho',
            'pizza',
            'pork chop',
            'poutine',
            'prime rib',
            'pulled pork sandwich',
            'ramen',
            'ravioli',
            'red velvet cake',
            'risotto',
            'samosa',
            'sashimi',
            'scallops',
            'seaweed salad',
            'shrimp and grits',
            'spaghetti bolognese',
            'spaghetti carbonara',
            'spring rolls',
            'steak',
            'strawberry shortcake',
            'sushi',
            'tacos',
            'takoyaki',
            'tiramisu',
            'tuna tartare',
            'waffles',
        ]

        templates = [
            'a photo of {}, a type of food.',
        ]
    elif dset_name == 'SUN397':
        classes = [
            'abbey',
            'airplane cabin',
            'airport terminal',
            'alley',
            'amphitheater',
            'amusement arcade',
            'amusement park',
            'anechoic chamber',
            'apartment building outdoor',
            'apse indoor',
            'aquarium',
            'aqueduct',
            'arch',
            'archive',
            'arrival gate outdoor',
            'art gallery',
            'art school',
            'art studio',
            'assembly line',
            'athletic field outdoor',
            'atrium public',
            'attic',
            'auditorium',
            'auto factory',
            'badlands',
            'badminton court indoor',
            'baggage claim',
            'bakery shop',
            'balcony exterior',
            'balcony interior',
            'ball pit',
            'ballroom',
            'bamboo forest',
            'banquet hall',
            'bar',
            'barn',
            'barndoor',
            'baseball field',
            'basement',
            'basilica',
            'basketball court outdoor',
            'bathroom',
            'batters box',
            'bayou',
            'bazaar indoor',
            'bazaar outdoor',
            'beach',
            'beauty salon',
            'bedroom',
            'berth',
            'biology laboratory',
            'bistro indoor',
            'boardwalk',
            'boat deck',
            'boathouse',
            'bookstore',
            'booth indoor',
            'botanical garden',
            'bow window indoor',
            'bow window outdoor',
            'bowling alley',
            'boxing ring',
            'brewery indoor',
            'bridge',
            'building facade',
            'bullring',
            'burial chamber',
            'bus interior',
            'butchers shop',
            'butte',
            'cabin outdoor',
            'cafeteria',
            'campsite',
            'campus',
            'canal natural',
            'canal urban',
            'candy store',
            'canyon',
            'car interior backseat',
            'car interior frontseat',
            'carrousel',
            'casino indoor',
            'castle',
            'catacomb',
            'cathedral indoor',
            'cathedral outdoor',
            'cavern indoor',
            'cemetery',
            'chalet',
            'cheese factory',
            'chemistry lab',
            'chicken coop indoor',
            'chicken coop outdoor',
            'childs room',
            'church indoor',
            'church outdoor',
            'classroom',
            'clean room',
            'cliff',
            'cloister indoor',
            'closet',
            'clothing store',
            'coast',
            'cockpit',
            'coffee shop',
            'computer room',
            'conference center',
            'conference room',
            'construction site',
            'control room',
            'control tower outdoor',
            'corn field',
            'corral',
            'corridor',
            'cottage garden',
            'courthouse',
            'courtroom',
            'courtyard',
            'covered bridge exterior',
            'creek',
            'crevasse',
            'crosswalk',
            'cubicle office',
            'dam',
            'delicatessen',
            'dentists office',
            'desert sand',
            'desert vegetation',
            'diner indoor',
            'diner outdoor',
            'dinette home',
            'dinette vehicle',
            'dining car',
            'dining room',
            'discotheque',
            'dock',
            'doorway outdoor',
            'dorm room',
            'driveway',
            'driving range outdoor',
            'drugstore',
            'electrical substation',
            'elevator door',
            'elevator interior',
            'elevator shaft',
            'engine room',
            'escalator indoor',
            'excavation',
            'factory indoor',
            'fairway',
            'fastfood restaurant',
            'field cultivated',
            'field wild',
            'fire escape',
            'fire station',
            'firing range indoor',
            'fishpond',
            'florist shop indoor',
            'food court',
            'forest broadleaf',
            'forest needleleaf',
            'forest path',
            'forest road',
            'formal garden',
            'fountain',
            'galley',
            'game room',
            'garage indoor',
            'garbage dump',
            'gas station',
            'gazebo exterior',
            'general store indoor',
            'general store outdoor',
            'gift shop',
            'golf course',
            'greenhouse indoor',
            'greenhouse outdoor',
            'gymnasium indoor',
            'hangar indoor',
            'hangar outdoor',
            'harbor',
            'hayfield',
            'heliport',
            'herb garden',
            'highway',
            'hill',
            'home office',
            'hospital',
            'hospital room',
            'hot spring',
            'hot tub outdoor',
            'hotel outdoor',
            'hotel room',
            'house',
            'hunting lodge outdoor',
            'ice cream parlor',
            'ice floe',
            'ice shelf',
            'ice skating rink indoor',
            'ice skating rink outdoor',
            'iceberg',
            'igloo',
            'industrial area',
            'inn outdoor',
            'islet',
            'jacuzzi indoor',
            'jail cell',
            'jail indoor',
            'jewelry shop',
            'kasbah',
            'kennel indoor',
            'kennel outdoor',
            'kindergarden classroom',
            'kitchen',
            'kitchenette',
            'labyrinth outdoor',
            'lake natural',
            'landfill',
            'landing deck',
            'laundromat',
            'lecture room',
            'library indoor',
            'library outdoor',
            'lido deck outdoor',
            'lift bridge',
            'lighthouse',
            'limousine interior',
            'living room',
            'lobby',
            'lock chamber',
            'locker room',
            'mansion',
            'manufactured home',
            'market indoor',
            'market outdoor',
            'marsh',
            'martial arts gym',
            'mausoleum',
            'medina',
            'moat water',
            'monastery outdoor',
            'mosque indoor',
            'mosque outdoor',
            'motel',
            'mountain',
            'mountain snowy',
            'movie theater indoor',
            'museum indoor',
            'music store',
            'music studio',
            'nuclear power plant outdoor',
            'nursery',
            'oast house',
            'observatory outdoor',
            'ocean',
            'office',
            'office building',
            'oil refinery outdoor',
            'oilrig',
            'operating room',
            'orchard',
            'outhouse outdoor',
            'pagoda',
            'palace',
            'pantry',
            'park',
            'parking garage indoor',
            'parking garage outdoor',
            'parking lot',
            'parlor',
            'pasture',
            'patio',
            'pavilion',
            'pharmacy',
            'phone booth',
            'physics laboratory',
            'picnic area',
            'pilothouse indoor',
            'planetarium outdoor',
            'playground',
            'playroom',
            'plaza',
            'podium indoor',
            'podium outdoor',
            'pond',
            'poolroom establishment',
            'poolroom home',
            'power plant outdoor',
            'promenade deck',
            'pub indoor',
            'pulpit',
            'putting green',
            'racecourse',
            'raceway',
            'raft',
            'railroad track',
            'rainforest',
            'reception',
            'recreation room',
            'residential neighborhood',
            'restaurant',
            'restaurant kitchen',
            'restaurant patio',
            'rice paddy',
            'riding arena',
            'river',
            'rock arch',
            'rope bridge',
            'ruin',
            'runway',
            'sandbar',
            'sandbox',
            'sauna',
            'schoolhouse',
            'sea cliff',
            'server room',
            'shed',
            'shoe shop',
            'shopfront',
            'shopping mall indoor',
            'shower',
            'skatepark',
            'ski lodge',
            'ski resort',
            'ski slope',
            'sky',
            'skyscraper',
            'slum',
            'snowfield',
            'squash court',
            'stable',
            'stadium baseball',
            'stadium football',
            'stage indoor',
            'staircase',
            'street',
            'subway interior',
            'subway station platform',
            'supermarket',
            'sushi bar',
            'swamp',
            'swimming pool indoor',
            'swimming pool outdoor',
            'synagogue indoor',
            'synagogue outdoor',
            'television studio',
            'temple east asia',
            'temple south asia',
            'tennis court indoor',
            'tennis court outdoor',
            'tent outdoor',
            'theater indoor procenium',
            'theater indoor seats',
            'thriftshop',
            'throne room',
            'ticket booth',
            'toll plaza',
            'topiary garden',
            'tower',
            'toyshop',
            'track outdoor',
            'train railway',
            'train station platform',
            'tree farm',
            'tree house',
            'trench',
            'underwater coral reef',
            'utility room',
            'valley',
            'van interior',
            'vegetable garden',
            'veranda',
            'veterinarians office',
            'viaduct',
            'videostore',
            'village',
            'vineyard',
            'volcano',
            'volleyball court indoor',
            'volleyball court outdoor',
            'waiting room',
            'warehouse indoor',
            'water tower',
            'waterfall block',
            'waterfall fan',
            'waterfall plunge',
            'watering hole',
            'wave',
            'wet bar',
            'wheat field',
            'wind farm',
            'windmill',
            'wine cellar barrel storage',
            'wine cellar bottle storage',
            'wrestling ring indoor',
            'yard',
            'youth hostel',
        ]

        templates = [
            'a photo of a {}.',
            'a photo of the {}.',
        ]
    elif dset_name == 'DTD':
        classes = [
            'banded',
            'blotchy',
            'braided',
            'bubbly',
            'bumpy',
            'chequered',
            'cobwebbed',
            'cracked',
            'crosshatched',
            'crystalline',
            'dotted',
            'fibrous',
            'flecked',
            'freckled',
            'frilly',
            'gauzy',
            'grid',
            'grooved',
            'honeycombed',
            'interlaced',
            'knitted',
            'lacelike',
            'lined',
            'marbled',
            'matted',
            'meshed',
            'paisley',
            'perforated',
            'pitted',
            'pleated',
            'polka-dotted',
            'porous',
            'potholed',
            'scaly',
            'smeared',
            'spiralled',
            'sprinkled',
            'stained',
            'stratified',
            'striped',
            'studded',
            'swirly',
            'veined',
            'waffled',
            'woven',
            'wrinkled',
            'zigzagged',
        ]

        templates = [
            'a photo of a {} texture.',
            'a photo of a {} pattern.',
            'a photo of a {} thing.',
            'a photo of a {} object.',
            'a photo of the {} texture.',
            'a photo of the {} pattern.',
            'a photo of the {} thing.',
            'a photo of the {} object.',
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

def map_imagenet_to_readable_label():
    with open('map_clsloc.txt') as f:
        lines = [x.strip().split() for x in f.readlines()]
    return {
        x[0] : x[2].replace('_', ' ') for x in lines
    }