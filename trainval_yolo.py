from sklearn.model_selection import train_test_split
import os, shutil

DATASET_DIR = './test/v2'
SPLIT_DATASET_DIR = './test/v2_split'

imgflist = []
for img in os.listdir(DATASET_DIR + '/images'):
    imgpath = DATASET_DIR + '/images/' + img
    txtpath = DATASET_DIR + '/labels/' + img[:-4] + '.txt'
    imgflist.append(
        (imgpath, txtpath)
    )

train, test = train_test_split(imgflist, test_size=0.2)

n=0


print('copy training set...')
# training directories
os.makedirs(SPLIT_DATASET_DIR + '/images/train')
os.makedirs(SPLIT_DATASET_DIR + '/labels/train')
for imagepath, labelpath in train:
    copyto_imagepath = SPLIT_DATASET_DIR + '/images/train/' + imagepath.split('/')[-1]
    copyto_labelpath = SPLIT_DATASET_DIR + '/labels/train/' + labelpath.split('/')[-1]
    shutil.copy(imagepath, copyto_imagepath)
    shutil.copy(labelpath, copyto_labelpath)
    print(n, copyto_imagepath, end='\r')
    n += 1
print('\ndone, {}/{}'.format(n, len(imgflist)))


print('copy test set...')
# test directories
os.makedirs(SPLIT_DATASET_DIR + '/images/test')
os.makedirs(SPLIT_DATASET_DIR + '/labels/test')
for imagepath, labelpath in test:
    copyto_imagepath = SPLIT_DATASET_DIR + '/images/test/' + imagepath.split('/')[-1]
    copyto_labelpath = SPLIT_DATASET_DIR + '/labels/test/' + labelpath.split('/')[-1]
    shutil.copy(imagepath, copyto_imagepath)
    shutil.copy(labelpath, copyto_labelpath)
    print(n, copyto_imagepath, end='\r')
    n += 1
print('\ndone, {}/{}'.format(n, len(imgflist)))
