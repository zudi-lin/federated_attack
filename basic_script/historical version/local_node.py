### Header classes and functions of this program


import time
import random
import os, sys
import numpy as np
import string
import logging
import boto3
import cv2
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath('../'))
from utils import preprocess_image, recreate_image
from utils.system_utils import *
from train_adv import *
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
### End of the header part ###


### Initialize constants

REGISTER_WAIT_RESPONSE_SECOND = 5
ATTACK_MAXIMUM_ROUNDS = 4

IDENTITY = 'aa'+''.join(random.SystemRandom().choice(string.ascii_lowercase + string.digits)
                        for _ in range(6)) # Create node identity
print("Indentity created: "+IDENTITY)

NODE_LOCAL_GENERIC_PATH = "../node_storage/"
NODE_LOCAL_PATH = "../node_storage/"+IDENTITY+"/"
NODE_LOCAL_ORIGINAL_IMAGES_PATH = NODE_LOCAL_PATH + "original_images/"

REMOTE_REGISTER_FOLDER = "register/"
REMOTE_ORIGINAL_IMAGE_FOLDER  = "original_images/"
REMOTE_ADV_IMAGE_FOLDER = "adv_images/"

IMAGE_FILE_SUFFIX = ".jpg"


### Intialize local storage
if not os.path.exists(NODE_LOCAL_GENERIC_PATH):
    os.mkdir(NODE_LOCAL_GENERIC_PATH)
if not os.path.exists(NODE_LOCAL_PATH):
    os.mkdir(NODE_LOCAL_PATH)
if not os.path.exists(NODE_LOCAL_ORIGINAL_IMAGES_PATH):
    os.mkdir(NODE_LOCAL_ORIGINAL_IMAGES_PATH)
print("Local storage initialized!")



### Register to the global server and download original images
registered = False
while not registered:
    message_file = NODE_LOCAL_PATH+IDENTITY
    message = open(message_file, "w+")
    message.close()

    rtn = upload_file(client, message_file, BUCKET, REMOTE_REGISTER_FOLDER, IDENTITY)
    print("Register message sent!")

    # Wait until receive adv_num
    while (True):
        time.sleep(REGISTER_WAIT_RESPONSE_SECOND)
        rtn = download_file(client, BUCKET, REMOTE_REGISTER_FOLDER, IDENTITY, message_file)
        if rtn:
            message = open(message_file, "r")
            content = message.read()
            if content!="" and content[0]>='0' and content[0]<='9':
                adv_num = int(content)
                message.close()
                registered = True
                print("Register confirmed and receievd Adv No. ", adv_num)
                download_dir(client, BUCKET, REMOTE_ORIGINAL_IMAGE_FOLDER, NODE_LOCAL_ORIGINAL_IMAGES_PATH)
                print("Original images downloaded!")
                break
            message.close()



### Start attack process
system_start_time = time.time()
round_cnt = 0
original_image_name_list = []
for file_name in np.sort(os.listdir(NODE_LOCAL_ORIGINAL_IMAGES_PATH)):
    if not (IMAGE_FILE_SUFFIX in file_name): continue
    original_image_name_list.append(file_name)
n_original_images = len(original_image_name_list)
print("Original image list: ", original_image_name_list)
print("")

while (True):
    print("Round "+str(round_cnt)+" start!")

    # Read each orginal image. The name is in the form of "image2.jpg"
    for image_name in original_image_name_list:
        # image_file = open(NODE_LOCAL_ORIGINAL_IMAGES_PATH+image_name, 'rb')
        print("To attack "+image_name+":")

        # Generate adv image name
        # the adv image name must be in the form "imageX_advY_verZ.jpg"
        # X = original image NO.; Y = adv_num; Z = a randomly generated version number
        # the version number allows you to generate adv image with different attacker-training epoch
        adv_image_name = (image_name.replace(IMAGE_FILE_SUFFIX, "")
                          + "_adv" + str(adv_num)
                          + "_tm" + str(time_tag(time.time()))
                          + IMAGE_FILE_SUFFIX
                         ) # e.g., IMAGE_FILE_SUFFIX = ".jpg"

        # Generate adversarial image
        # pass


        # adv_image_data = image_file.read()  #blank attack
        # image_file.close()

        local_original_image = cv2.imread(NODE_LOCAL_ORIGINAL_IMAGES_PATH+image_name, 1)
        preprocess_original_image = preprocess_image(local_original_image)
        time.sleep(5) # Simulate the situation where each adv image requires 5 seconds to create


        # To do: decompose features x and label y
        # x, y = preprocess_original_image.unsqueeze(0),

        # To do: import net and device
        # net, device =

        Generate_Adv = GenAdv(net, device, F.cross_entropy)
        local_adv_image, local_adv_label = Generate_Adv.generate_adv(x, y)

        recreated_local_adv_image = recreate_image(local_adv_image)

        # Save adv image to local storage first
        cv2.imwrite(NODE_LOCAL_PATH+adv_image_name, recreated_local_adv_image)
        # adv_image_file.write(adv_image_data)
        # adv_image_file.close()

        # Upload new version of adv image to cloud (can handle the first upload)
        upload_file(client, NODE_LOCAL_PATH + adv_image_name, BUCKET, REMOTE_ADV_IMAGE_FOLDER, adv_image_name)
        print("Uploaded "+ adv_image_name+"!")


    round_cnt +=1

    # terminate the system based on the round counter
    if round_cnt>=ATTACK_MAXIMUM_ROUNDS: break

    time.sleep(1) # Do NOT change this, otherwise AWS budget will be exhausted rapidly!!!

    # if the system runs too long, close it automatically.
    system_curr_time = time.time()
    if system_curr_time - system_start_time >= 30 * 60:
        break
