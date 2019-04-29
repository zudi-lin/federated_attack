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

    full_remote_file_name = remote_folder_name + remote_file_name
    # Upload the file
    try:
        response = client.upload_file(local_file_name, bucket_name, full_remote_file_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def download_file(client, bucket_name, remote_folder_name, remote_file_name, local_file_name):
    full_remote_file_name = remote_folder_name + remote_file_name
    response = client.list_objects_v2(Bucket=bucket_name, Prefix=full_remote_file_name)
    if response['KeyCount']<=0:
        return False
    else: 
        client.download_file(bucket_name, full_remote_file_name, local_file_name)
    return True

def download_dir(client, bucket_name, remote_dir, local_dir):
    rtn = client.list_objects_v2(Bucket=bucket_name, Prefix=remote_dir)
    if rtn['KeyCount']<=0: return 0
    
    files = rtn["Contents"]
    for file in files:
        file_name = file['Key'][len(remote_dir):]
        client.download_file(bucket_name, file['Key'], local_dir+file_name)
        
    return rtn['KeyCount']

# remove files by prefix
def remove_files_by_prefix(client, bucket_name, prefix):
    rtn = client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if rtn['KeyCount']<=0: 
        print("No prefix matched in clear_files()!  Prefix="+prefix)
        return 
    
    current_nodes = rtn["Contents"]
    for node in current_nodes:
        file_name = node['Key'][len(prefix):]
        delete_file(client, bucket_name, prefix, file_name)
    print("Deleted all files with PREFIX " + prefix +" in the path!")
    return 

def delete_file(client, bucket_name, remote_folder_name, remote_file_name):
    full_remote_file_name = remote_folder_name + remote_file_name
    return client.delete_object(Bucket=bucket_name, Key=full_remote_file_name)

def time_tag(time):
    return int((time - int(time) + int(time) % 1000000 ) * 1000)