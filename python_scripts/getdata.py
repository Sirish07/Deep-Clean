import pdb
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.handlers import disable_signing
import tarfile
import urllib.request
import time

prefix = "data/uploads/EEG_Eyetracking_CMI_data_compressed/"
s3 = boto3.resource('s3')
s3.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
bucket = s3.Bucket(name="fcp-indi")
FilesNotFound = True

count = 0
for obj in enumerate(bucket.objects.filter(Prefix=prefix)):
	if count<45:
		count += 1
		continue
	if count > 49:
		break
	stime = time.time()
	thetarfile = "https://fcp-indi.s3.amazonaws.com/" + obj[1].key
	ftpstream = urllib.request.urlopen(thetarfile)
	thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
	thetarfile.extractall(path = './scratch/Dataset/')
	etime = time.time()
	print('The time taken for subject %s is %s'%(count,etime-stime))
	count += 1

