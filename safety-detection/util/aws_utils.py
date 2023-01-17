import boto3
import os
import uuid

from botocore.exceptions import NoCredentialsError
from loguru import logger
# Set the profile for boto3 account
boto3.setup_default_session(profile_name='jp')

class sqs_transfer():
    def __init__(self,sqs_queue_name = 'uat_sqs_violations_v1') -> None:
        self.sqs = boto3.resource('sqs',endpoint_url="https://sqs.ap-southeast-1.amazonaws.com")
        self.sqs_queue_name = sqs_queue_name

    def push_msg(self,kinesis_name,msg_type,violation_id,category,subcategory,object_url):

        queue = self.sqs.get_queue_by_name(QueueName=self.sqs_queue_name)
        response = queue.send_message(MessageBody='safety_violation', MessageAttributes={
            'msg_type':{
                'StringValue': msg_type,
                'DataType': 'String'
            },
            'kinesis_name':{
                'StringValue': kinesis_name,
                'DataType': 'String'
            },
            'category':{
                'StringValue': category,
                'DataType': 'String'
            },
            'subcategory':{
                'StringValue': subcategory,
                'DataType': 'String'
            },
            'obj_url':{
                'StringValue': object_url,
                'DataType': 'String'
            },
            'violation_id':{
                'StringValue': violation_id,
                'DataType': 'String'
            }
        }
        )
        logger.info("Msg pushed with" + str(response.get('MessageId')) + str(response.get('MD5OfMessageBody')))
        return True
    

class s3_transfer():
    def __init__(self,region_name='ap-southeast-1',aws_bucket_name='uat-jp-violations') -> None:
        self.region_name = region_name
        self.aws_bucket_name = aws_bucket_name
    
    def gen_uuid(self):
        return str(uuid.uuid4())

    def s3_file_transfer(self,local_file,s3_file,remove_local = True,violation_id=None):
        s3 = boto3.client('s3',self.region_name)
        if violation_id == None:
            violation_id = self.gen_uuid()

        try:
            with open(local_file, "rb") as f:
                s3.upload_fileobj(f, self.aws_bucket_name,'%s/%s' % (violation_id,s3_file))
            # If local file to be removed after successful insertion in s3 bucket
            if remove_local:
                os.remove(local_file)
            logger.info("Success in upload object to s3 with id " + str(violation_id))
            return violation_id,str('/' + violation_id + '/' + s3_file)
        except FileNotFoundError:
            logger.debug("Issue at S3 sending FileNotFound")
            return False
        except NoCredentialsError:
            logger.debug("Issue at S3 sending Credential error")
            return False
        except Exception as e:
            logger.debug("Issue at S3 sending ",e)
            return False 