'''Upload data and model weights to Amazon S3'''

import os, sys
import time
import shutil
import logging
import boto3
from botocore.exceptions import ClientError

sys.path.append(os.path.abspath('../'))
from utils.system_utils import *

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

filename = 'adv_data.h5'
upload_file(client, filename, BUCKET, 'adv_cifar10/', filename)