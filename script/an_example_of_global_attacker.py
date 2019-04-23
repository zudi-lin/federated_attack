from system_utils import *
import time
import logging
import boto3
from botocore.exceptions import ClientError

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

# Creat a massage as a local file
state_file = open("../sys_test_data/adv1_state.txt","w+")
state_file.write("disconnect\n")
state_file.close()


# Send the massage to the attacker
upload_file(client, "../sys_test_data/adv1_state.txt", BUCKET, "adv1", 'state.txt')

#delete_file(client, BUCKET, "adv1", 'state.txt')