
import torch 
import torch.nn as nn 

import math 
import random
import blobfile as bf 
from mpi4py import MPI  
from PIL import Image
import numpy as np 
import os 
from datasets.transforms import center_crop_arr,random_crop_arr

class AHNIR_Dataset(nn.Module):
    def __init__(self,root_dataset,img_size,
                 classes=["HE","MAS","PAS","PASM"],
                 class_cond=False,
                 shard=0,
                 num_shard=1, 
                 random_crop=False,
                 random_flip=True):
        super(AHNIR_Dataset,self).__init__()
        self.root_dir=root_dataset
        self.img_size=img_size
        self.random_crop=random_crop
        self.random_flip=random_flip
        self.classes=classes
        self.class_cond=class_cond
        self.shard=shard
        self.num_shards=num_shard
        # Listing all image files recursively from the root dataset directory
        self.local_images = self._list_images_file_recursively()

        # Applying sharding to the list of image files
        if self.num_shards > 1:
            self.local_images = self.local_images[self.shard::self.num_shards]
            
        
    def _list_images_file_recursively(self):
        # Initialize the list to store image paths
        image_paths = []

        # Iterate over each class directory to collect image file paths
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                        image_paths.append([os.path.join(root, file),class_name])
        return image_paths
    
    def __len__(self):
        return len(self.local_images)

    def __getitem__(self,index):
        path,class_name=self.local_images[index]
        pil_image_source=Image.open(path)
        pil_image=pil_image_source.convert("RGB")
        # pil_image_gray=pil_image_source.convert("L")
        # pil_image_gray=Image.merge("RGB", (pil_image_gray, pil_image_gray, pil_image_gray))
        # Apply transformations if specified
        if self.random_crop:
            arr = random_crop_arr(pil_image,self.img_size)
            # arr_gray = random_crop_arr(pil_image_gray,self.img_size)
        else:
            arr = center_crop_arr(pil_image,self.img_size)
            # arr_gray = center_crop_arr(pil_image_gray,self.img_size)
        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]
            # arr_gray = arr_gray[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1
        # arr_gray = arr_gray.astype(np.float32) / 127.5 - 1
        
        out_dict={}
        # Return image and corresponding class label index
        label_kwargs={"HE":torch.tensor(0,dtype=torch.int16),"MAS":torch.tensor(1,dtype=torch.int16),"PAS":torch.tensor(2,dtype=torch.int16),"PASM":torch.tensor(3,dtype=torch.int16)}
        if self.class_cond:
            label_index = label_kwargs[class_name]#self.classes.index(class_name) if class_name in self.classes else None
            if label_index is not None:
                out_dict['y']=np.array(label_index,dtype=np.int16)
        return  np.transpose(arr, [2, 0, 1]), out_dict  # ,np.transpose(arr_gray, [2, 0, 1])


def get_ahnir_dataloader(dataset,batch_size,deterministic=False,num_workers=1,drop_last=True):
    if deterministic:
        return torch.utils.data.DataLoader(dataset,batch_size=batch_size,
                                           shuffle=False,num_workers=num_workers,
                                           drop_last=drop_last)
    else:
        return torch.utils.data.DataLoader(dataset,batch_size=batch_size,
                                           shuffle=True,num_workers=num_workers,
                                           drop_last=drop_last)

# if __name__ == "__main__":
#     dataset=AHNIR_Dataset(root_dataset="/opt/data/private/virtualStain/ANHIR/train",
#                           img_size=256,shard=MPI.COMM_WORLD.Get_rank(),
#                           num_shard=MPI.COMM_WORLD.Get_size())
#     data=get_ahnir_dataloader(dataset,1,True)
#     for i,d in enumerate(data):
#         print(d['image'].shape)
#         print(d['label'])
#         import cv2 
#         sample_ = ((d['image'] + 1) * 127.5).clamp(0, 255).to(torch.uint8)
#         sample_ = sample_.permute(0, 2, 3, 1)
#         sample_ = sample_.contiguous()
        
#         img=Image.fromarray(sample_.cpu().numpy()[0])
#         img.save("res.png")
        
#         break
    