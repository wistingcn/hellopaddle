import cv2
import numpy as np 
import matplotlib.pyplot as plt
%matplotlib inline
import paddle.vision.transforms as T
import glob
import codecs
import os
import random
import shutil
from PIL import Image

orig_img = cv2.imread('data/sports/consolidated/speed skating/106.jpg')
print(orig_img.shape)

plt.imshow(orig_img)          #根据数组绘制图像
plt.show() 

# 计算图像数据整体均值和方差


def get_mean_std(image_path_list):
    print('Total images:', len(image_path_list))
    max_val, min_val = np.zeros(3), np.ones(3) * 255
    mean, std = np.zeros(3), np.zeros(3)
    for image_path in image_path_list:
        image = cv2.imread(image_path)
        for c in range(3):
            mean[c] += image[:, :, c].mean()
            std[c] += image[:, :, c].std()
            max_val[c] = max(max_val[c], image[:, :, c].max())
            min_val[c] = min(min_val[c], image[:, :, c].min())

    mean /= len(image_path_list)
    std /= len(image_path_list)

    mean /= max_val - min_val
    std /= max_val - min_val

    return mean, std


mean, std = get_mean_std(glob.glob('data/sports/consolidated/speed skating/*.jpg'))
print('mean:', mean)
print('std:', std)


class MyImageNetDataset(paddle.io.Dataset):
    def __init__(self,
                 num_samples,
                 num_classes):
        super(MyImageNetDataset, self).__init__()

        self.num_samples = num_samples
        self.num_classes = num_classes
        self.transform = T.Compose([
            T.Resize(size=(224, 224)),
            T.ToTensor(),
            T.Normalize(mean=127.5, std=127.5)])

    def __getitem__(self, index):
        image = np.random.randint(low=0, high=256, size=(512, 512, 3))
        label = np.random.randint(low=0, high=self.num_classes, size=(1,))

        image = image.astype('float32')
        label = label.astype('int64')

        image = self.transform(image)

        return image, label

    def __len__(self):
        return self.num_samples
      
      
train_dataset = MyImageNetDataset(num_samples=1200, num_classes=1000)
print(len(train_dataset))

image, label = train_dataset[0]
print(image.shape, label.shape)

for image, label in train_dataset:
    print(image.shape, label.shape)
    break
    
train_dataloader = paddle.io.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    drop_last=False)

for step, data in enumerate(train_dataloader):
    image, label = data
    print(step, image.shape, label.shape)
    


# 获取项目
all_file_dir = 'data/sports/'
all_file_dir_train = 'data/sports/' + 'train/'
speed_skating = 'speed skating'
class_list = [c for c in os.listdir(all_file_dir_train) if os.path.isdir(os.path.join(all_file_dir_train, c)) and not c.startswith('.')]
# 重新排序
class_list.sort()

# 配置数据目录
train_image_dir = all_file_dir_train  
test_image_dir = all_file_dir + 'test/'
valid_image_dir = all_file_dir + 'valid/'

def generate_list(dir_name):
    file_list=[]
    for sub_dir in class_list:
        label=class_list.index(sub_dir)
        img_paths=os.listdir(os.path.join(dir_name, sub_dir))
        for img_path in img_paths:
            file_list.append(os.path.join(sub_dir,img_path)+'\t%d'% label +'\n')
    return file_list

# 保存列表
def save_list2file(mylist, filename):
    with open(filename, 'w') as f:
        f.writelines(mylist)


train_list= generate_list(train_image_dir)
test_list= generate_list(test_image_dir)
valid_list= generate_list(valid_image_dir)

save_list2file(train_list,'train_list.txt')
save_list2file(test_list,'test_list.txt')
save_list2file(valid_list,'valid_list.txt')

from paddle.io import Dataset
import paddle.vision.transforms as T
import numpy as np
from PIL import Image

class SportsDataset(Dataset):
    def __init__(self, mode, transform):
        self.transform=transform
        super(SportsDataset, self).__init__()
        """
        初始化函数
        """
        self.data = []
        self.mode=mode
        with open('{}_list.txt'.format(mode)) as f:
            for line in f.readlines():
                info = line.strip().split('\t')
                if len(info) > 0:
                    # print(info[0].strip())
                    self.data.append([info[0].strip(), info[1].strip()])
            

    def get_origin_data(self):
        return self.data

    def __getitem__(self, index):
        """
        根据索引获取单个样本
        """
        image_file, label = self.data[index]
        
        image = Image.open(os.path.join('data/sports/{}'.format(self.mode), image_file))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        return image, np.array(label, dtype='int64')

    def __len__(self):
        return len(self.data)


train_transform=T.Compose([
    T.RandomHorizontalFlip(),   #随机水平反转,默认0.5概率
    T.ColorJitter(0.4, 0.4, 0.4, 0.4), #随机调整图像的亮度，对比度，饱和度和色调。
    T.ToTensor(),                       # 数据的格式转换和标准化 HWC => CHW  
    T.Normalize()                       # 图像归一化
])

valid_transform=T.Compose([
    T.ToTensor(),                       # 数据的格式转换和标准化 HWC => CHW  
    T.Normalize()                       # 图像归一化
])

test_transform=T.Compose([
    T.Resize((224,224)), 
    T.ToTensor(),                       # 数据的格式转换和标准化 HWC => CHW  
    T.Normalize()                       # 图像归一化
])
train_dataset = SportsDataset(mode='train', transform=train_transform)
test_dataset = SportsDataset(mode='test', transform=test_transform)
valid_dataset = SportsDataset(mode='valid', transform=valid_transform)



for i in range(1,10):
    plt.subplot(2,5,i)
    filename=os.path.join(all_file_dir_train + speed_skating, str(i).zfill(3)+'.jpg')
    img=plt.imread(filename)
    plt.imshow(img)
    
    
#载入模型
from paddle.vision.models import MobileNetV2
model=MobileNetV2(num_classes=70)

import paddle.nn as nn
import paddle
model=paddle.Model(model)
model.prepare(optimizer=paddle.optimizer.Adam(parameters=model.parameters()),
              loss=paddle.nn.CrossEntropyLoss(),
              metrics=paddle.metric.Accuracy())

# 训练
model.fit(train_dataset,
          test_dataset,
          epochs=100,
          batch_size=50,
          drop_last=True,
          log_freq=1,
          shuffle=True,
          verbose=2)   

eval_result = model.evaluate(test_dataset, verbose=1)
print(eval_result)

predict_result = model.predict(test_dataset)
print(predict_result)
