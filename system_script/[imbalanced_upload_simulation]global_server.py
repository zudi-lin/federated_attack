#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Header classes and functions of this file
from system_utils import *
import os
import time
import shutil
import numpy as np
import logging
import h5py
import boto3
from botocore.exceptions import ClientError
from sortedcontainers import SortedSet

ACCESS_KEY = "Your_AWS_S3_ACCESS_KEY"
SECRET_KEY = "Your_AWS_S3_SECRET_KEY"
REGION = "us-east-1"
BUCKET = "Your_AWS_S3_BUCKET"

client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name=REGION
)

response = client.list_buckets()['Buckets']

def register_new_nodes():
    # Examine whether there is a new local attacker
    rtn = client.list_objects_v2(Bucket=BUCKET, Prefix=REMOTE_REGISTER_FOLDER)
    if rtn['KeyCount']<=0: return 
    
    current_nodes = rtn["Contents"]
    for node in current_nodes:
        identity = node['Key'][len(REMOTE_REGISTER_FOLDER):]
        if not (identity in register_nodes):
            register_file_path = SERVER_LOCAL_REGISTER_PATH+identity
            #rtn = download_file(client, BUCKET, "register", identity, register_file_path)
            register_file = open(register_file_path, "w+")
            adv_num = len(register_nodes)
            register_file.write(str(adv_num))
            register_file.close()
            rtn = upload_file(client, register_file_path, BUCKET, REMOTE_REGISTER_FOLDER, identity)
            register_nodes[identity] = 1
            adv_processed_num[adv_num] = 0
            print("Detect new node "+identity+" -->"+" allocated adv NO. "+str(adv_num) +"!")
    return 

def get_remote_adv_file_directory(file_buffer, n_file_buffered):
    # file_buffer is a set of lists, the keys are adv no
    # and each adv no has a list of files represented by timetag in the filename
    # n_file_buffered is the total files in the file_buffer
    adv_file_list = check_prefix(client, BUCKET, REMOTE_ADV_IMAGE_FOLDER)
    mask_len = len(REMOTE_ADV_IMAGE_FOLDER)
    for file_name in adv_file_list:
        rtn = file_name[mask_len:].replace(DATA_FILE_SUFFIX,'').split('_')
        adv_no = int(rtn[0][3:])
        adv_timetag = int(rtn[1][2:])
        if adv_no in file_buffer:
            sub_timetag_dict = file_buffer[adv_no]
            if not (adv_timetag in sub_timetag_dict):
                sub_timetag_dict.add(adv_timetag)
                n_file_buffered += 1
        else:
            file_buffer[adv_no] = SortedSet([adv_timetag])
            n_file_buffered += 1                
    return file_buffer, n_file_buffered
    
    
def select_prior_files_from_file_buffer(file_buffer, n_file_buffered, adv_processed_num):
    n_file_process = 0
    process_filename_list = []
    while n_file_buffered > 0 and n_file_process < FILE_PROCESS_BATCH_SIZE:
        adv_priority = sorted(file_buffer, key=lambda x: adv_processed_num[x])
        for adv_no in adv_priority:
            adv_timetag = file_buffer[adv_no].pop(0)
            adv_filename = f"adv{adv_no}_tm{adv_timetag}" + DATA_FILE_SUFFIX
            download_file(client, BUCKET, REMOTE_ADV_IMAGE_FOLDER, adv_filename, 
                          SERVER_LOCAL_ADV_IMAGES_PATH+adv_filename)
            delete_file(client, BUCKET, REMOTE_ADV_IMAGE_FOLDER, adv_filename)
            process_filename_list.append(adv_filename)
            print(f"Dataset {adv_filename} is downloaded and the remote version is deleted!")
            if len(file_buffer[adv_no])==0:
                file_buffer.pop(adv_no, None)
            adv_processed_num[adv_no] += 1
            n_file_process += 1
            n_file_buffered -= 1
            if n_file_process >= FILE_PROCESS_BATCH_SIZE: break
                
    return process_filename_list, file_buffer, n_file_buffered, adv_processed_num

def get_image_date_from_file_list(process_filename_list):
    update_image_data_table = {}
    update_image_adv_table = {} # For Debug purpose only
    for adv_filename in process_filename_list:
        fl = h5py.File(SERVER_LOCAL_ADV_IMAGES_PATH+adv_filename,'r')
        images = np.array(fl['image'])
        labels = np.array(fl['label'])
        identifiers = np.array(fl['identifier'])
        fl.close()
        n_images = len(labels)
        for i in range(n_images):
            curr_id = identifiers[i]
            if  curr_id in update_image_data_table:
                update_image_data_table[curr_id].append(images[i])
                update_image_adv_table[curr_id].append(adv_filename[:6]) # Debug 
            else:
                update_image_data_table[curr_id] = [images[i]]
                update_image_adv_table[curr_id] = [adv_filename[:6]] # Debug
                
    return update_image_data_table, update_image_adv_table
    
### End of the header part ###


# In[3]:


### Initialize constants
REMOTE_REGISTER_FOLDER = "register/"
REMOTE_ORIGINAL_IMAGE_FOLDER  = "original_images/"
REMOTE_ADV_IMAGE_FOLDER = "adv_images/"
REMOTE_AGG_IMAGE_FOLDER = "agg_images/"
REMOTE_ORIGINAL_IMAGE_DATASET = "test_128.h5"

SERVER_LOCAL_PATH = "../global_storage/"
SERVER_LOCAL_REGISTER_PATH = SERVER_LOCAL_PATH + "register/"
SERVER_LOCAL_ORIGINAL_IMAGES_PATH = SERVER_LOCAL_PATH + "original_images/"
SERVER_LOCAL_AGGREGATED_IMAGES_PATH = SERVER_LOCAL_PATH + "aggregated_images/"
SERVER_LOCAL_ADV_IMAGES_PATH = SERVER_LOCAL_PATH + "images_from_nodes/"
SERVER_LOCAL_ORIGINAL_IMAGE_DATASET = REMOTE_ORIGINAL_IMAGE_DATASET

IMAGE_FILE_SUFFIX = ".jpg"
DATA_FILE_SUFFIX = ".h5"

MINIMUM_FILE_BUFFER_SIZE = 5
FILE_PROCESS_BATCH_SIZE = 5

### Clean and re-initialize local storage of the global server
if not os.path.exists(SERVER_LOCAL_PATH): 
    os.mkdir(SERVER_LOCAL_PATH)
if not os.path.exists(SERVER_LOCAL_ORIGINAL_IMAGES_PATH): 
    os.mkdir(SERVER_LOCAL_ORIGINAL_IMAGES_PATH)
server_local_original_image_dataset = SERVER_LOCAL_ORIGINAL_IMAGES_PATH + SERVER_LOCAL_ORIGINAL_IMAGE_DATASET
if not os.path.exists(server_local_original_image_dataset): 
    download_file(client, BUCKET, REMOTE_ORIGINAL_IMAGE_FOLDER, REMOTE_ORIGINAL_IMAGE_DATASET,
                  server_local_original_image_dataset)
    print(f"Original dataset {REMOTE_ORIGINAL_IMAGE_DATASET} downloaded!")
    
if os.path.exists(SERVER_LOCAL_REGISTER_PATH): 
    shutil.rmtree(SERVER_LOCAL_REGISTER_PATH)    
if os.path.exists(SERVER_LOCAL_ADV_IMAGES_PATH): 
    shutil.rmtree(SERVER_LOCAL_ADV_IMAGES_PATH)    
if os.path.exists(SERVER_LOCAL_AGGREGATED_IMAGES_PATH): 
    shutil.rmtree(SERVER_LOCAL_AGGREGATED_IMAGES_PATH)    

os.mkdir(SERVER_LOCAL_REGISTER_PATH)
os.mkdir(SERVER_LOCAL_ADV_IMAGES_PATH)
os.mkdir(SERVER_LOCAL_AGGREGATED_IMAGES_PATH)
print("Local storage initialized!")


### Clean and re-initialize the cloud
remove_files_by_prefix(client, BUCKET, REMOTE_REGISTER_FOLDER)
remove_files_by_prefix(client, BUCKET, REMOTE_ADV_IMAGE_FOLDER)
remove_files_by_prefix(client, BUCKET, REMOTE_AGG_IMAGE_FOLDER)
#remove_files_by_prefix(client, BUCKET, REMOTE_ORIGINAL_IMAGE_FOLDER)


# In[5]:


### Initialize variables
register_nodes = {}

fl = h5py.File(SERVER_LOCAL_ORIGINAL_IMAGES_PATH+SERVER_LOCAL_ORIGINAL_IMAGE_DATASET,'r')
original_images = np.array(fl['image'])
original_labels = np.array(fl['label'])
original_identifiers = np.array(fl['identifier'])
fl.close()
id_to_idx = dict(zip(original_identifiers, np.arange(len(original_identifiers), dtype=int)))

### Lunch the global server
system_start_time = time.time()
round_cnt = 0
adv_processed_num = {} # the number of files processed for each adv_no
file_buffer = {}
n_file_buffered = 0
while (True):
    time.sleep(1) # Do NOT change this, otherwise AWS budget will be exhausted rapidly!!!
    print(f"Round {round_cnt} started!")
    round_cnt += 1
    
    # Check whether there are new nodes to be registered 
    register_new_nodes()
    print(f"Checking new nodes is finished!")
    
    # Get directories new adv data file from REMOTE
    if n_file_buffered < MINIMUM_FILE_BUFFER_SIZE:
        print("Fetch remote directory!")
        file_buffer, n_file_buffered = get_remote_adv_file_directory(file_buffer, n_file_buffered)
                
    
    # Select a prioritized subset to download and delete the remote version after download
    process_filename_list, file_buffer, n_file_buffered, adv_processed_num =     select_prior_files_from_file_buffer(file_buffer, n_file_buffered, adv_processed_num)
     
    print("process_filename_list: ", process_filename_list)
    print("adv_processed_num: ", adv_processed_num)
    if len(process_filename_list)<=0: 
        print("No new file found!")
        continue
        
    # Generate a matrix of updated original images and the adv images
    update_image_data_table, update_image_adv_table = get_image_date_from_file_list(process_filename_list)
    print("update_image_adv_table: ", update_image_adv_table)
    
    
    
    ######################################################################
    # The start of real aggregation
    for curr_id in update_image_data_table:  
        # Chech whether there exists previous aggregation result, read if yes, otherwise, use initial value
        summary_file_name = f"summary_{curr_id}.npy"
        if os.path.exists(SERVER_LOCAL_AGGREGATED_IMAGES_PATH+summary_file_name):
            summary = np.load(SERVER_LOCAL_AGGREGATED_IMAGES_PATH+summary_file_name)
        else: 
            summary = 0 # PLEASE Change to default value when there's no previous aggregation result for this img
         
        # For the current image identifier "curr_id", the new adv images for this image is
        # in the list "update_image_data_table[curr_id]" <-- This is a list of images, not a single element!!!!
        # The correct label of this image is "original_labels[id_to_idx[curr_id]]"
        # The original image is "original_images[id_to_idx[curr_id]]"
        # The summary of previous aggregation result of this image is stored in file "summary_file_name"
        
        # Example:
        curr_original_image = original_images[id_to_idx[curr_id]]
        aggregated_img = curr_original_image + summary + np.max(update_image_data_table[curr_id]) # Blank aggr
        new_summary = summary + np.max(update_image_data_table[curr_id]) # blank summary
        
        
        # Save the aggregated image to local storage
        agg_image_name = f"agg{curr_id}_tm{time_tag(time.time())}" + IMAGE_FILE_SUFFIX
        fl = open(SERVER_LOCAL_AGGREGATED_IMAGES_PATH+agg_image_name, 'wb')
        fl.write(aggregated_img)  
        fl.close()
        
        # Save the new summary of this image to local storage
        np.save(SERVER_LOCAL_AGGREGATED_IMAGES_PATH + summary_file_name, new_summary)
        
        # Upload new version of adv image to cloud (can handle the first upload)
        upload_file(client, SERVER_LOCAL_AGGREGATED_IMAGES_PATH + agg_image_name, BUCKET, 
                    REMOTE_AGG_IMAGE_FOLDER, agg_image_name)
        #update_remote_adv_image(adv_image_name)
        print("Aggregated image " + agg_image_name + " uploaded!")
        
    # The end of real aggregation
    ######################################################################
    
    #wait for some seconds
    time.sleep(3) 
    
    # if the system runs too long, close it automatically.
    system_curr_time = time.time()
    if system_curr_time - system_start_time >= 5 * 60:
        break
    #break


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


fl=h5py.File(SERVER_LOCAL_ORIGINAL_IMAGES_PATH+"test_128.h5",'r')
images = np.array(fl['image'])
labels = np.array(fl['label'])
identifiers = np.array(fl['identifier'])
fl.close()
print(images.shape)
print(labels.shape)
print(type(labels[0]))
print(labels)
print(identifiers)


upload_file(client, SERVER_LOCAL_ORIGINAL_IMAGES_PATH+"test_128.h5", 
            BUCKET, REMOTE_ORIGINAL_IMAGE_FOLDER, "test_128.h5")
#fl=h5py.File(SERVER_LOCAL_ORIGINAL_IMAGES_PATH+"test_128.h5", 'w')
#fl.create_dataset('image', data=images[:128])
#fl.create_dataset('label', data=labels[:128])
#fl.create_dataset('identifier', data=identifiers[:128])
#fl.close()


# In[ ]:





# In[ ]:





# In[ ]:




