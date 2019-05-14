from system_utils import *
import time
import logging
import boto3
from botocore.exceptions import ClientError

# Connecting to the AWS server using the client class in boto3
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

'''
Running a real time local attacker that uploads one image per 30 seconds 
and stops uploading if recieves a massage from the global attacker
'''

round_cnt = 0
system_start_time = time.time()
while (True):
    print("Round "+str(round_cnt)+" start!")
    
    #download control massage, a way to communicate with the gloabel attacker
    rtn = download_file(client, BUCKET, "adv1", "state.txt", "../sys_test_data/adv1_rec_state.txt")
    if rtn: 
        state = open("../sys_test_data/adv1_rec_state.txt").read()
        if "disconnect" in state: 
            print("Terminated by massage from the global attacker.")
            break
    
    #upload phrase
    rtn = upload_file(client, "../sys_test_data/test"+str(round_cnt)+".jpg", 
                      BUCKET, "adv1", "test"+str(round_cnt)+".jpg")
    if not rtn: break
    
    #wait for some seconds
    time.sleep(30)
    
    round_cnt +=1
    
    # terminate the system based on the round counter
    if round_cnt>=5: break
    
    # if the system runs too long, close it automatically.
    system_curr_time = time.time()
    if system_curr_time - system_start_time >= 30 * 60:
        break
