import torch
from sklearn.neighbors import NearestNeighbors 
import _pickle as cPickle
from PIL import Image
import torchvision
import torchvision.transforms as T
import os 
import pandas as pd
# import hubconf
import tqdm
from tqdm import tqdm_notebook

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print('device:',device)
# dinov2_vits14 = hubconf.dinov2_vits14()
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
dinov2_vits14.to(device)
def extract_features(filename):
    img = Image.open(filename)

    transform = T.Compose([
    T.Resize((224, 224), T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

    img = transform(img)[:3].unsqueeze(0)

    with torch.no_grad():
        features = dinov2_vits14(img.to('cuda'))[0]

    return features.detach().cpu().numpy()

extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def get_file_list(root_dir):
    file_list = []
    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                filepath = os.path.join(root, filename)
                if os.path.exists(filepath):
                  file_list.append(filepath)
                else:
                  print(filepath)
    return file_list

# # path to the your datasets
root_dir = '../imgs' 
train_df = pd.read_csv("Train.csv")
filenames = []

for index, row in train_df.iterrows():
    if row['filename'] not in filenames:
        if "Copy.jpg" not in row['filename']:
            filenames.append(os.path.join(root_dir, row['filename']))
filenames = sorted(filenames)
print('Total files :', len(filenames))
feature_list = []
for i in tqdm.tqdm(range(len(filenames))):
    feature_list.append(extract_features(filenames[i]))

cPickle.dump(feature_list,open('dino-all-feature-list.pickle','wb'), 5)
cPickle.dump(filenames,open('dino-all-filenames.pickle','wb'), 5)
# neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute',metric='euclidean').fit(feature_list)
# # Save the model to a file
# with open('dino-all-neighbors2.pkl', 'wb') as f:
#     pickle.dump(neighbors, f)