import skimage
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from pathlib import Path
from tqdm import tqdm

# our folder path containing some images
folder_path = './dataset/'
augmented_path = './dataset_augmented/'

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

    
# dictionary of the transformations functions we defined earlier
available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'horizontal_flip': horizontal_flip
}


image_dir = Path(folder_path)
folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]

for i, direc in tqdm(enumerate(folders)):
    for file in tqdm(direc.iterdir()):
        img = skimage.io.imread(file)
        
        for transform in available_transformations:
            transformed_image = available_transformations[transform](img)                
            new_file_path = augmented_path+direc.name+"/"+transform+"_"+file.name
            sk.io.imsave(new_file_path, transformed_image)
            
                
                
                
                

    