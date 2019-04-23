"""
API of AWS s3.client, please refer:
    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#client
"""

import time
import logging
import boto3
from botocore.exceptions import ClientError

def upload_file(client, local_file_name, bucket_name, remote_folder_name, remote_file_name):
    """Upload a file to an S3 bucket
    :param file_name: File to upload
    :param remote_folder_name: S3 Folder to store the file
    :param remote_file_name: name of the file to store
    :return: True if file was uploaded, else False
    """
    
    if '/' in remote_folder_name or '/' in remote_file_name:
        print("Error: remote_folder_name and remote_file_name cannot contain '/'!")
        return 
    full_remote_file_name = remote_folder_name + "/" + remote_file_name
    # Upload the file
    try:
        response = client.upload_file(local_file_name, bucket_name, full_remote_file_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def download_file(client, bucket_name, remote_folder_name, remote_file_name, local_file_name):
    full_remote_file_name = remote_folder_name + "/" + remote_file_name
    response = client.list_objects_v2(Bucket=bucket_name, Prefix=full_remote_file_name)
    if response['KeyCount']<=0:
        return False
    else: client.download_file(bucket_name, full_remote_file_name, local_file_name)
    return True

def delete_file(client, bucket_name, remote_folder_name, remote_file_name):
    full_remote_file_name = remote_folder_name + "/" + remote_file_name
    return client.delete_object(Bucket=bucket_name, Key=full_remote_file_name)