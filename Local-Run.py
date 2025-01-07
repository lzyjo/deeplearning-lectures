



# Checking and setting cwd 
import os

## Check the current working directory
cwd = os.getcwd()
print(f"Current working directory: {cwd}")
project_path = os.getenv('PROJECT_PATH', r"C:\Users\me20332\OneDrive - University of Bristol\MscR\GitHub\a-PyTorch-Tutorial-to-Object-Detection")
os.chdir(project_path)  # Set the current working directory if needed
## Check the current working directory
cwd = os.getcwd()
print(f"Current working directory: {cwd}")



# Importing data
# The data is stored in a tar file. We will extract the contents of the tar file to the current working directory.

import tarfile

## Check if the extracted directories already exist
## 2007 Data
if not os.path.exists('VOCdevkit/VOC2007'):
    file_path = os.getenv('VOC2007_TEST_PATH', r"C:\Users\me20332\OneDrive - University of Bristol\MscR\Running Model Resources\VOCtest_06-Nov-2007.tar")  # Define the file_path variable with the actual path to your tar file

    ## Open the tar file in read mode ('r')
    with tarfile.open(file_path, 'r') as tar:
        # Extract all members of the archive to the current working directory
        tar.extractall(path=os.getcwd())
    file_path = os.getenv('VOC2007_TRAIN_PATH', r"C:\Users\me20332\OneDrive - University of Bristol\MscR\Running Model Resources\VOCtrainval_06-Nov-2007.tar")  # Define the file_path variable with the actual path to your tar file

    with tarfile.open(file_path, 'r') as tar:
        tar.extractall(path=os.getcwd())
        print('2007 Train Done')

else:
    print("2007 Data already extracted.")

## 2012 Data
if not os.path.exists('VOCdevkit/VOC2012'):
    file_path = os.getenv('VOC2012_TRAIN_PATH', r"C:\Users\me20332\OneDrive - University of Bristol\MscR\Running Model Resources\VOCtrainval_11-May-2012.tar")  # Define the file_path variable with the actual path to your tar file

    with tarfile.open(file_path, 'r') as tar:
        tar.extractall(path=os.getcwd())
        print('2012 Train Done')
else:
    print("2012 Data already extracted.")
    
## Check if data in cwd
if os.path.exists('VOCdevkit'):
    print("VOCdevkit is present in the current working directory.")
else:
    print("VOCdevkit is not present in the current working directory.")
    


#Creating datalists for the train and test data
from utils import create_data_lists

if not (os.path.exists('TEST_images.json') and os.path.exists('TEST_objects.json') and os.path.exists('TRAIN_images.json') and os.path.exists('TRAIN_objects.json')):
    voc07_path = r"C:\Users\me20332\OneDrive - University of Bristol\MscR\GitHub\a-PyTorch-Tutorial-to-Object-Detection\VOCdevkit\VOC2007"  # Define the voc07_path variable with the actual path to your VOC2007 data
    voc12_path = r"C:\Users\me20332\OneDrive - University of Bristol\MscR\GitHub\a-PyTorch-Tutorial-to-Object-Detection\VOCdevkit\VOC2012"
    output_folder = './'

    create_data_lists(voc07_path=voc07_path,
                      voc12_path=voc12_path,
                      output_folder=output_folder)
    
    print("Done creating datalists.")

else:
    print("Datalists already created: There are 16551 training images containing a total of 49653 objects. "
          "Files have been saved to /content/a-PyTorch-Tutorial-to-Object-Detection. "
          "There are 4952 test images containing a total of 14856 objects. "
          "Files have been saved to /content/a-PyTorch-Tutorial-to-Object-Detection.")
    

# Training the model 
# AND THEN 
# Evaluate the model

## Check if model is already trained and present 
if os.path.exists('checkpoint_ssd300.pth'):
    print('Model already trained: checkpoint_ssd300.pth present in cwd')
else:
    print('Model not trained or present: checkpoint_ssd300.pth not present in cwd')

os.system('python eval.py')

"""Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 78/78 [1:12:28<00:00, 55.75s/it]
{'aeroplane': 0.790989875793457,
 'bicycle': 0.8309301733970642,
 'bird': 0.7649813294410706,
 'boat': 0.7225903868675232,
 'bottle': 0.46215584874153137,
 'bus': 0.8695299625396729,
 'car': 0.8656850457191467,
 'cat': 0.883727490901947,
 'chair': 0.5916000604629517,
 'cow': 0.8243222832679749,
 'diningtable': 0.758436381816864,
 'dog': 0.854961633682251,
 'horse': 0.8753499984741211,
 'motorbike': 0.8315761685371399,
 'person': 0.7886016964912415,
 'pottedplant': 0.5129949450492859,
 'sheep': 0.7923182845115662,
 'sofa': 0.7978971004486084,
 'train': 0.8620267510414124,
 'tvmonitor': 0.7486801743507385}

Mean Average Precision (mAP): 0.771"""


# Inference 

os.system('python detect.py')
"""Using Pillow 9.5.0:
 C:\Users\me20332\OneDrive - University of Bristol\MscR\GitHub\a-PyTorch-Tutorial-to-Object-Detection\model.py:501: UserWarning: indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen/native/IndexingUtils.h:28.)
  image_boxes.append(class_decoded_locs[1 - suppress])
C:\Users\me20332\OneDrive - University of Bristol\MscR\GitHub\a-PyTorch-Tutorial-to-Object-Detection\model.py:503: UserWarning: indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen/native/IndexingUtils.h:28.)
  image_scores.append(class_scores[1 - suppress])
C:\Users\me20332\OneDrive - University of Bristol\MscR\GitHub\a-PyTorch-Tutorial-to-Object-Detection\detect.py:88: DeprecationWarning: getsize is deprecated and will be removed in Pillow 10 (2023-07-01). Use getbbox or getlength instead. 
  text_size = font.getsize(det_labels[i].upper())"""

img_path = 'VOCdevkit/VOC2007/JPEGImages/009958.jpg'
original_image = PIL.Image.open(img_path, mode='r')
original_image = original_image.convert('RGB')

import detect as detect

detect.detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200).show()
