'''Upload data and model weights to Amazon S3'''

from system_utils import *
import os
import time
import shutil
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

filename = 'adv_data.h5'
upload_file(client, filename, BUCKET, 'adv_cifar10/', filename)