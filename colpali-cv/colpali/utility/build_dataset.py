import os
from PIL import Image as PILImage
from datasets import Dataset, Features, Image, Value
from datasets import load_from_disk
from tqdm import tqdm
from PIL import Image as iimg


DATASET_FOLDER = "/Users/aditya.narayan/Desktop/ColPali-CV_Parsing/colpali-cv/colpali/dataset"
UPLOAD_FOLDER = "/Users/aditya.narayan/Desktop/ColPali-CV_Parsing/colpali-cv/colpali/uploads"

width, height = 800, 600 
black_image = iimg.new('RGB', (width, height), (0, 0, 0))
black_image.save(UPLOAD_FOLDER + '/sample.png')


def load_images_and_filenames_from_folder(UPLOAD_FOLDER):
    images = []
    filenames = []
    for filename in tqdm(os.listdir(UPLOAD_FOLDER)):
        if filename.endswith(".png"):  # Filter for PNG files
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            img = PILImage.open(img_path)
            images.append(img)
            filenames.append(filename)  # Save the filename
        else:
            pass
    return images, filenames

def create_custom_dataset(images, filenames):
    # Define features for the dataset, including the image and filename
    features = Features({
        'image': Image(),
        'filename': Value("string")
    })
    dataset_dict = {'image': images, 'filename': filenames}
    
    # Create dataset with the specified features
    dataset = Dataset.from_dict(dataset_dict, features=features)
    return dataset


print("🛠️ Building Dataset")
# Update with the actual folder path
images, filenames = load_images_and_filenames_from_folder(UPLOAD_FOLDER)
# Create the custom dataset
custom_dataset = create_custom_dataset(images, filenames)
# Saving custom dataset
custom_dataset.save_to_disk(DATASET_FOLDER)

# Loading Dataset
dataset = load_from_disk(DATASET_FOLDER)
images = dataset["image"]
filenames = dataset["filename"]
print("✅ Data Loaded ")