#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Header classes and functions of this program

from system_utils import *
import time
import random
import os
import numpy as np
import string
import logging
import h5py
import boto3
from botocore.exceptions import ClientError

# ACCESS_KEY = "AKIAYSSR3P6HBYS35VUA"
# SECRET_KEY = "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"
# BUCKET = "2019.harvard.cs244r"
# REGION = "us-east-1"

ACCESS_KEY = "AKIAZHARFTUU4CYJRSPM"
SECRET_KEY = "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"
BUCKET = "tmp201905111159"
REGION = "us-east-1"

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


# In[2]:


### Initialize constants

REGISTER_WAIT_RESPONSE_SECOND = 0.2
ATTACK_MAXIMUM_ROUNDS = 10

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

BANDWIDTH_SIMULATION_MODE = True
CAPACITY_CHANGE_DURATION = 1000
BANDWIDTH_NOISE_SCALE = 0


### Intialize local storage
if not os.path.exists(NODE_LOCAL_GENERIC_PATH): 
    os.mkdir(NODE_LOCAL_GENERIC_PATH)
if not os.path.exists(NODE_LOCAL_PATH): 
    os.mkdir(NODE_LOCAL_PATH)
if not os.path.exists(NODE_LOCAL_ORIGINAL_IMAGES_PATH): 
    os.mkdir(NODE_LOCAL_ORIGINAL_IMAGES_PATH)
print("Local storage initialized!")


# In[3]:


### Register to the global server and download original images
registered = False
while not registered:
    message_file = NODE_LOCAL_PATH+IDENTITY
    message = open(message_file, "w")
    message.close()

    rtn = download_file(client, BUCKET, REMOTE_REGISTER_FOLDER, IDENTITY, message_file)
    if not rtn:
        upload_file(client, message_file, BUCKET, REMOTE_REGISTER_FOLDER, IDENTITY)
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
    
    


# In[18]:


### Start attack process
system_start_time = time.time()
round_cnt = 0
dataset_process_times = {}
timeout = False

if BANDWIDTH_SIMULATION_MODE:
    BANDWIDTH_SCALE = 200
    tst_capacity = np.random.rand() * BANDWIDTH_SCALE
    #tst_upload_speed = np.random.rand() * BANDWIDTH_SCALE
    #tst_upload_speed = 1
    tst_upload_speed = tst_capacity
    
message_file = NODE_LOCAL_PATH+IDENTITY
monitor_period_start_time = time.time()
cmd_timetag = 0

while not timeout:
    
    if BANDWIDTH_SIMULATION_MODE:
        # Read bandwidth constraint from the global
        rtn = download_file(client, BUCKET, REMOTE_REGISTER_FOLDER, IDENTITY+"_cmd", message_file+"_cmd")
        tst_upload_cmd = 0 # Default command
        if rtn:
            message = open(message_file+"_cmd", "r")
            content = message.readlines()
            if len(content)>0 and int(content[1])!= cmd_timetag:
                tst_upload_cmd = int(content[0])
                cmd_timetag = int(content[1])


        print("Upload cmd: %f, Current upload speed: %f" % (tst_upload_cmd, tst_upload_speed))
        # Set the current upload bandwidth
        if tst_upload_cmd == -1: 
            tst_upload_speed = tst_upload_speed / 1.1
        elif tst_upload_cmd == 1 and tst_upload_speed < tst_capacity:
            tst_upload_speed = tst_upload_speed * 1.1
        tst_final_bandwidth = tst_upload_speed + np.random.randn() * BANDWIDTH_SCALE * BANDWIDTH_NOISE_SCALE
        tst_final_bandwidth = np.max([tst_final_bandwidth, 0])
        tst_final_bandwidth = np.min([tst_final_bandwidth, tst_capacity])

        # Send the current upload bandwith to the global
        message = open(message_file+"_state", "w")
        message.write(str(int(tst_final_bandwidth)) + '\n' + str(int(tst_capacity)))
        message.close()
        upload_file(client, message_file+"_state", BUCKET, REMOTE_REGISTER_FOLDER, IDENTITY+"_state")
        print("Final upload bandwidth: %f, Current capacity: %f" % (tst_final_bandwidth, tst_capacity))
        
        
        # Simulate the change in the capacity in next period
        curr_time = time.time()
        print(curr_time, monitor_period_start_time)
        if curr_time - monitor_period_start_time > CAPACITY_CHANGE_DURATION:
            print("\nLocal network capacity CHANGES!")
            tst_capacity += np.random.randn() * BANDWIDTH_SCALE * 0.1
            tst_capacity = np.max([tst_capacity, 0])
            monitor_period_start_time = curr_time
    
    time.sleep(0.05) # Do NOT change this, otherwise AWS budget will be exhausted rapidly!!!
    
    # if the system runs too long, close it automatically.
    system_curr_time = time.time()
    if system_curr_time - system_start_time >= 5 * 60:
        timeout = True
        break
        
    '''
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
        
        # Example:
        adv_image_list = np.array(image_list) # blank attack
        adv_label_list = np.array(label_list) # This should not be changed if the order of images is not changed
        adv_identifier_list = np.array(identifier_list)
        time.sleep(3) # Simulate the situation where each adv image requires 5 seconds to create
        
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
            
        time.sleep(1) # Do NOT change this, otherwise AWS budget will be exhausted rapidly!!!
    
        # if the system runs too long, close it automatically.
        system_curr_time = time.time()
        if system_curr_time - system_start_time >= 60:
            timeout = True
            break
    '''


# In[ ]:





# In[16]:





# In[ ]:




