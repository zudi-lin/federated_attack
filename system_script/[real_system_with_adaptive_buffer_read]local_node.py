#!/usr/bin/env python
# coding: utf-8


### Header classes and functions of this program

import time
import random
import os, sys
sys.path.append(os.path.abspath('../'))
from utils.system_utils import *

import numpy as np
import string
import logging
import h5py
import boto3
from botocore.exceptions import ClientError
import torch
import torch.nn.functional as F
from model.cifar import *

from utils.utils import preprocess_image, recreate_image
from basic_script.train_adv import *

ACCESS_KEY = "AKIAYSSR3P6HBYS35VUA"
SECRET_KEY = "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"
REGION = "us-east-1"
BUCKET = "2019.harvard.cs244r"

client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name=REGION
)

response = client.list_buckets()['Buckets']


def download_and_pick_a_dateset():
    dataset_name_list = check_prefix(client, BUCKET, REMOTE_ORIGINAL_IMAGE_FOLDER)
    print(dataset_name_list)
    mask_len = len(REMOTE_ORIGINAL_IMAGE_FOLDER)
    for dataset_name in dataset_name_list:
        if (".h5" in dataset_name) and not (dataset_name in dataset_process_times):
            dataset_name = dataset_name[mask_len:]
            rtn = download_file(client, BUCKET, REMOTE_ORIGINAL_IMAGE_FOLDER, dataset_name,
                                NODE_LOCAL_ORIGINAL_IMAGES_PATH + dataset_name)
            dataset_process_times[dataset_name] = 0
            print("Original image dataset %s downloaded!" % dataset_name)
    min_process_time = 100000
    for dataset_name in dataset_process_times:
        if dataset_process_times[dataset_name] < min_process_time:
            least_processed_dataset = dataset_name
            min_process_time = dataset_process_times[dataset_name]
    return dataset_name
### End of the header part ###



### Initialize constants
ATTACK_MAXIMUM_ROUNDS = 10000

REGISTER_WAIT_RESPONSE_SECOND = 0.5
ROUND_WAIT_DURATION = 0

IDENTITY = 'aa'+''.join(random.SystemRandom().choice(string.ascii_lowercase + string.digits)
                        for _ in range(6)) # Create node identity
print("Indentity created: "+IDENTITY)

NODE_LOCAL_GENERIC_PATH = "../node_storage/"
NODE_LOCAL_PATH = "../node_storage/"+IDENTITY+"/"
NODE_LOCAL_ORIGINAL_IMAGES_PATH = NODE_LOCAL_PATH + "original_images/"


REMOTE_REGISTER_FOLDER = "register/"
REMOTE_ORIGINAL_IMAGE_FOLDER  = "original_images/"
REMOTE_ADV_IMAGE_FOLDER = "adv_images/"
REMOTE_ORIGINAL_DATASET_NAME = "test_128.h5"

IMAGE_FILE_SUFFIX = ".jpg"

BATCH_SIZE = 16


### Intialize local storage
if not os.path.exists(NODE_LOCAL_GENERIC_PATH):
    os.mkdir(NODE_LOCAL_GENERIC_PATH)
if not os.path.exists(NODE_LOCAL_PATH):
    os.mkdir(NODE_LOCAL_PATH)
if not os.path.exists(NODE_LOCAL_ORIGINAL_IMAGES_PATH):
    os.mkdir(NODE_LOCAL_ORIGINAL_IMAGES_PATH)
print("Local storage initialized!")

### Intialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ResNet18().to(device)

### Register to the global server and download original images
registered = False
while not registered:
    message_file = NODE_LOCAL_PATH+IDENTITY
    message = open(message_file, "w+")
    message.close()

    rtn = download_file(client, BUCKET, REMOTE_REGISTER_FOLDER, IDENTITY, message_file)
    if not rtn:
        rtn = upload_file(client, message_file, BUCKET, REMOTE_REGISTER_FOLDER, IDENTITY)
        print("Register message sent!")

    # Wait until receive adv_num
    while (True):
        if rtn:
            message = open(message_file, "r")
            content = message.read()
            if content!="" and content[0]>='0' and content[0]<='9':
                adv_num = int(content)
                message.close()
                registered = True
                print("Register confirmed and receievd Adv No. ", adv_num)
                break
            message.close()
        time.sleep(REGISTER_WAIT_RESPONSE_SECOND)
        rtn = download_file(client, BUCKET, REMOTE_REGISTER_FOLDER, IDENTITY, message_file)



### Start attack process
system_start_time = time.time()
round_cnt = 0
dataset_process_times = {}
timeout = False

print(dataset_process_times)
while not timeout:
    curr_dataset = download_and_pick_a_dateset()
    dataset_complete = False
    fl = h5py.File(NODE_LOCAL_ORIGINAL_IMAGES_PATH+REMOTE_ORIGINAL_DATASET_NAME,'r')
    images = np.array(fl['image'])
    labels = np.array(fl['label'])
    identifiers = np.array(fl['identifier'])
    fl.close()
    total_image_num = len(labels)
    print(f"Dataset {curr_dataset} loaded!")

    # Read each orginal image. The name is in the form of "image2.jpg"
    processed_image_num = 0
    while processed_image_num < total_image_num:
        print("Round "+str(round_cnt)+" start!")
        next_batch_num = np.min([processed_image_num + BATCH_SIZE, total_image_num])
        image_list = images[processed_image_num: next_batch_num]
        label_list = labels[processed_image_num: next_batch_num]
        identifier_list = identifiers[processed_image_num: next_batch_num]
        processed_image_num = next_batch_num
        print(f"To attack images {identifier_list[0]} ~ {identifier_list[-1]}:")

        ######################################################################
        # The start of generating adversarial images
        # the input batch of images = image_list
        # the labels of the input batch images = label_list
        # the identifiers of the input batch images = identifier_list

        # please store the generated batch of adv images ==> adv_image_list, AS A NUMPY ARRAY!!!
        # please store the original labels of the batch ==> adv_label_list
        # please store the corresponding identifiers of the batch ==> adv_identifier_list
        # if you do not change the order the images, then
        # it should be "adv_identifier_list == identifier_list" and "adv_label_list==label_list"

        # Prepare data to input in the neural network:
        x = torch.stack([preprocess_image(np.array(image)) for image in image_list])
        y = torch.from_numpy(np.array(label_list)).long()

        # Define the model and device of local machine, the following is a toy example
        Generate_Adv = GenAdv(net, device, F.cross_entropy)
        adv_image_list, adv_label_list = Generate_Adv.generate_adv(x, y)

        # Recreate images from transformed images.
        adv_image_list = np.stack([recreate_image(adv_image) for adv_image in adv_image_list])
        adv_label_list = np.array(adv_label_list.numpy(), dtype=int)
        adv_identifier_list = np.array(identifier_list)

        # Save adv image to local storage first
        #adv_file_name = "adv0_tm425635841.h5"
        adv_file_name = f"adv{str(adv_num)}_tm{str(time_tag(time.time()))}.h5"
        fl=h5py.File(NODE_LOCAL_PATH + adv_file_name, 'w')
        fl.create_dataset('image', data=adv_image_list)
        fl.create_dataset('label', data=adv_label_list)
        fl.create_dataset('identifier', data=adv_identifier_list)
        fl.close()


        # Upload new version of adv image to cloud (can handle the first upload)
        upload_file(client, NODE_LOCAL_PATH + adv_file_name, BUCKET, REMOTE_ADV_IMAGE_FOLDER, adv_file_name)
        print("Uploaded "+ adv_file_name+"!")

        # The end of generating adversial images
        ######################################################################

        round_cnt +=1

        # terminate the system based on the round counter
        if round_cnt>=ATTACK_MAXIMUM_ROUNDS:
            timeout = True
            break

        time.sleep(ROUND_WAIT_DURATION) # Do NOT change this, otherwise AWS budget will be exhausted rapidly!!!

        # if the system runs too long, close it automatically.
        system_curr_time = time.time()
        if system_curr_time - system_start_time >= 5 * 60:
            timeout = True
            break
    break

if timeout:
    print("Warning: Time out!")
else:
    print("Mission complete!")
