### Header classes and functions of this file

import os, sys
import time
import shutil
import numpy as np
import logging
import boto3
import cv2

from utils.utils import preprocess_image, recreate_image
from utils.system_utils import *
sys.path.append(os.path.abspath('../'))
from train_adv import aggregate_adv_noise
from botocore.exceptions import ClientError


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
            print("Detect new node "+identity+" -->"+" allocated adv NO. "+str(adv_num) +"!")
    return

### End of the header part ###


### Initialize constants
SERVER_LOCAL_PATH = "../global_storage/"
SERVER_LOCAL_REGISTER_PATH = SERVER_LOCAL_PATH + "register/"
SERVER_LOCAL_ORIGINAL_IMAGES_PATH = SERVER_LOCAL_PATH + "original_images/"
SERVER_LOCAL_AGGREGATED_IMAGES_PATH = SERVER_LOCAL_PATH + "aggregated_images/"
SERVER_LOCAL_ADV_IMAGES_PATH = SERVER_LOCAL_PATH + "images_from_nodes/"

REMOTE_REGISTER_FOLDER = "register/"
REMOTE_ORIGINAL_IMAGE_FOLDER  = "original_images/"
REMOTE_ADV_IMAGE_FOLDER = "adv_images/"
REMOTE_AGG_IMAGE_FOLDER = "agg_images/"

IMAGE_FILE_SUFFIX = ".jpg"

### Clean and re-initialize local storage of the global server
if not os.path.exists(SERVER_LOCAL_PATH):
    os.mkdir(SERVER_LOCAL_PATH)
if not os.path.exists(SERVER_LOCAL_ORIGINAL_IMAGES_PATH):
    os.mkdir(SERVER_LOCAL_ORIGINAL_IMAGES_PATH)

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
#remove_files_by_prefix(client, BUCKET, REMOTE_ADV_IMAGE_FOLDER)
remove_files_by_prefix(client, BUCKET, REMOTE_AGG_IMAGE_FOLDER)
remove_files_by_prefix(client, BUCKET, REMOTE_ORIGINAL_IMAGE_FOLDER)


### Rename original images and upload to cloud
original_image_name_list = np.sort(os.listdir(SERVER_LOCAL_ORIGINAL_IMAGES_PATH))
cnt = 0
for file_name in original_image_name_list:
    if not (IMAGE_FILE_SUFFIX in file_name): continue
    new_file_name = "img"+str(cnt)+".jpg"
    src = SERVER_LOCAL_ORIGINAL_IMAGES_PATH + file_name
    des = SERVER_LOCAL_ORIGINAL_IMAGES_PATH + new_file_name
    os.rename(src, des)
    upload_file(client, des, BUCKET, REMOTE_ORIGINAL_IMAGE_FOLDER, new_file_name)
    cnt += 1

### Initialize variables
register_nodes = {}
original_image_name_list = []
for file_name in np.sort(os.listdir(SERVER_LOCAL_ORIGINAL_IMAGES_PATH)):
    if not (IMAGE_FILE_SUFFIX in file_name): continue
    original_image_name_list.append(file_name)
n_original_images = len(original_image_name_list)
image_version_dict_list = [{} for i in range(n_original_images)]

print(original_image_name_list)

### Lunch the global server
system_start_time = time.time()
while (True):

    # Check whether there are new nodes to be registered
    register_new_nodes()


    # Aggregate adv images
    for i, image_name in enumerate(original_image_name_list):
        # Collect adv images on the i-th original images
        print("To try " + image_name + ":")
        image_name = image_name.replace(IMAGE_FILE_SUFFIX, "")
        rtn = client.list_objects_v2(Bucket=BUCKET, Prefix=REMOTE_ADV_IMAGE_FOLDER+image_name)
        if rtn['KeyCount']<=0:
            print("No new adv images.\n")
            continue

        updated = False # whether the adv images of the i-th original images have been updated
        curr_image_version_dict = image_version_dict_list[i]
        adv_images = rtn["Contents"]
        for adv_image in adv_images:
            adv_image_full_name = adv_image['Key'][len(REMOTE_ADV_IMAGE_FOLDER):]
            rtn = adv_image_full_name.replace(IMAGE_FILE_SUFFIX,"").split('_')
            adv_str = rtn[1]
            tm_str = rtn[2]
            if (not (adv_str in curr_image_version_dict)) or (curr_image_version_dict[adv_str]!=tm_str):
                # a new adv image found
                if (not (adv_str in curr_image_version_dict)) or (curr_image_version_dict[adv_str]<tm_str):
                    # the new adv image is the latest one
                    # download the latest one
                    download_file(client, BUCKET, REMOTE_ADV_IMAGE_FOLDER, adv_image_full_name,
                                  SERVER_LOCAL_ADV_IMAGES_PATH+adv_image_full_name)
                    # remove the local old one
                    if adv_str in curr_image_version_dict:
                        old_adv_image_path = (SERVER_LOCAL_ADV_IMAGES_PATH+image_name+"_"+adv_str+"_"
                                              +curr_image_version_dict[adv_str]+IMAGE_FILE_SUFFIX)
                        if os.path.exists(old_adv_image_path): os.remove(old_adv_image_path)
                    # update the image_version_dict
                    curr_image_version_dict[adv_str] = tm_str
                    updated = True
                    print("New adversarial image "+ adv_image_full_name+" downloaded!")

                # remove this new one from the remote as it has been found and processed
                delete_file(client, BUCKET, REMOTE_ADV_IMAGE_FOLDER, adv_image_full_name)

        if updated:
            # Do aggregation
            # Collect all adv images of the i-th original images
            adv_image_name_list = []
            for file_name in np.sort(os.listdir(SERVER_LOCAL_ADV_IMAGES_PATH)):
                if file_name.startswith(image_name): adv_image_name_list.append(file_name)
            print("To aggregate images in ", adv_image_name_list, ":")

            ### Generate aggregated image
            # blank implementation, please replace it
            agg_image_name = (image_name + "_agg_tm"
                              + str(time_tag(time.time()))
                              + IMAGE_FILE_SUFFIX)
            '''
            file = open(SERVER_LOCAL_ADV_IMAGES_PATH+adv_image_name_list[0], 'rb')
            agg_image_data = file.read()
            file.close()

            # Save adv image to local storage first
            agg_image_file = open(SERVER_LOCAL_AGGREGATED_IMAGES_PATH + agg_image_name, "wb")
            agg_image_file.write(agg_image_data)
            agg_image_file.close()
            print("Aggregated image " + agg_image_name + " created!")
            '''
            global_original_image = cv2.imread(SERVER_LOCAL_ORIGINAL_IMAGES_PATH+image_name+'.jpg', 1)
            preprocess_global_original_image = preprocess_image(global_original_image)
            server_local_adv_noise = []
            for i in range(len(adv_image_name_list)):
                local_adv_image = cv2.imread(SERVER_LOCAL_ADV_IMAGES_PATH+adv_image_name_list[i]+'.jpg', 1)
                preprocess_local_adv_image = preprocess_image(global_original_image)
                server_local_adv_noise.append(local_adv_image - preprocess_globale_original_image)

            # aggregation
            global_aggregated_image =\
                aggregate_adv_noise(preprocess_global_original_image, server_local_adv_noise)

            recreated_global_aggregated_image = recreate_image(global_aggregated_image)

            # Save adv image to local storage first
            cv2.imwrite(SERVER_LOCAL_AGGREGATED_IMAGES_PATH + agg_image_name,\
                recreated_global_aggregated_image)
            print("Aggregated image " + agg_image_name + " created!")


            # Upload new version of adv image to cloud (can handle the first upload)
            upload_file(client, SERVER_LOCAL_AGGREGATED_IMAGES_PATH + agg_image_name, BUCKET,
                        REMOTE_AGG_IMAGE_FOLDER, agg_image_name)
            #update_remote_adv_image(adv_image_name)
            print("Aggregated image " + agg_image_name + " uploaded!")

        print("")

    #wait for some seconds
    time.sleep(5)


    time.sleep(1) # Do NOT change this, otherwise AWS budget will be exhausted rapidly!!!

    # if the system runs too long, close it automatically.
    system_curr_time = time.time()
    if system_curr_time - system_start_time >= 30 * 60:
        break
    #break
