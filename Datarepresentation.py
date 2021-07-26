#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
fd = pd.read_csv('metadata.csv')


# In[2]:


print(fd.columns)


# In[3]:


print(fd.head(10))


# In[4]:


ax = fd['finding'].value_counts(dropna=False).plot.pie(y='Finding', legend = True, autopct='%2.0f%%', figsize = (10,10), title = 'Finding Distribution')


# In[5]:


ax = fd['sex'].value_counts(dropna=False).plot.pie(y='sex', legend = True, autopct='%2.0f%%', figsize = (5,5), title = 'Sex Distribution')


# In[6]:


out = pd.cut(fd['age'], bins=np.arange(0,110,10).tolist(), include_lowest=True)
ax = out.value_counts(sort=False, dropna=False).plot.bar(rot=0, color="b", figsize=(15,6), title= "Age Distribution")
#ax.set_xticklabels([c[1:-1].replace(","," to") for c in out.cat.categories])
plt.show()


# In[7]:


ax = fd['survival'].value_counts(dropna=False).plot.pie(y='survival', legend = True, autopct='%2.0f%%', figsize = (5,5), title = 'Survival Distribution')


# In[8]:


ax = fd['view'].value_counts(dropna=False).plot.pie(y='view', legend = True, autopct='%2.0f%%', figsize = (8,8), title = 'View distribution')


# In[9]:


ax = fd['modality'].value_counts(dropna=False).plot.pie(y='modality', legend = True, autopct='%2.0f%%', figsize = (5,5), title = 'Modality Distribution')


# In[10]:


fd['country'] = fd['location'].apply(lambda x: x.split(',')[-1].replace(" ","") if x is not np.nan else "Unknown")
ax = fd['country'].value_counts(dropna=False).plot.pie(y='country', legend = True, autopct='%2.0f%%', figsize = (10,10), title = 'Location of the patient')


# In[11]:


from lxml.html import parse
from sys import stdin
import pandas as pd
import re
import json
import numpy as np
import os
import skimage
import pydicom


# In[12]:


# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 11:22:31 2020

@author: zoshaikh
"""

import os
import shutil
from urllib.parse import urlparse
from scipy.stats import truncnorm
from subprocess import run
import time
from tqdm import tqdm
import pickle
import os
import pandas as pd

global_speedup = 2

def string_or_empty(string):
    if string is None:
        return ""
    return string

def filename_from_url(url):
    "Determine the filename that the given url should be downloaded to."
    return os.path.basename(urlparse(url).path)

def find_new_entries(old_data, new_cases):
    "Iterate through the URLs in new_cases and return metadata from the ones that are not already in old_data."
    new_data = []
    for new_case in new_cases:
        if not new_case._in(old_data):
            data = new_case.get_data()
            new_data.append(data)
            yield data

def deduplicate_filename(retrieve_filename, img_dir):
    print("Starting deduplicate")
    files = os.listdir(img_dir)
    test_filename = retrieve_filename
    name, ext = os.path.splitext(retrieve_filename)
    i = 1
    while test_filename in files:
        test_filename = name + "-" + str(i) + ext
        i += 1
    print("Done deduplicating filename")
    return test_filename

def output_candidate_entries(standard, columns, out_name, img_dir, resource_cache, retry=False):
    "Save the candidate entries to a file."
    pickle_name = out_name + "_pickled_data"
    out_df = pd.DataFrame(columns=columns)
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    all_records = []
    for record in standard:
        all_records.append(record)
        with open(pickle_name, "wb") as handle:
            pickle.dump(all_records, handle)
        patient = clean_standard_data(record)
        all_filenames = []
        #print(patient)
        for image in patient["images"]:
            #print(image)
            for url in image["url"]:
                retrieve_filename = deduplicate_filename(
                    filename_from_url(url),
                    img_dir
                )
                try:
                    #print("ResourceCache", url, list(ResourceCache.data))
                    #print("being transferred to", retrieve_filename)
                    full_destination = os.path.join(img_dir, retrieve_filename)
                    #print("full_destination", full_destination)
                    source = resource_cache.get(url)
                    #print("source", source)
                    shutil.copyfile(
                        source,
                        full_destination
                    )
                except Exception as e:
                    print(e)
                    #pdb.set_trace()
                    print("Oh no! Failed to download!")
                    raise
                else:
                    all_filenames.append(retrieve_filename)
                break
            else:
                all_filenames.append("")
        out_df = out_df.append(
            standard_to_metadata_format(
                patient,
                all_filenames
            ),
            ignore_index=True
        )
        out_df.to_csv(out_name)

def wget(src, filename):
    if run(["wget", src, "-O", filename]).returncode != 0:
        raise ValueError

def wait(n):
    """Wait a random period of time"""
    time_to_sleep = truncnorm.rvs(n/2, n*2, loc=n, scale=n/4) / global_speedup
    granularity = 10
    for i in tqdm(range(int(time_to_sleep * granularity))):
        time.sleep(1/granularity)

def clean_standard_data(standard_record):
    "Sanitize data already in an interoperable format."
    def sanitize_sex(sex):
        sex = sex.lower()
        if "f" in sex or "w" in sex:
            return "F"
        else:
            return "M"

    def sanitize_age(age):
        for possible_age in age.split(" "):
            try:
                return int(possible_age)
            except:
                pass
    def sanitize_finding(finding):
        if "covid19" in finding.lower().replace("-","").replace(" ",""):
            return "COVID-19"
        else:
            return finding
    print(standard_record)
    sex = standard_record["patient"]["sex"]
    if not sex in ["M", "F"]:
        standard_record["patient"]["sex"] = sanitize_sex(sex)

    standard_record["patient"]["age"] = sanitize_age(
        standard_record["patient"]["age"]
    )

    standard_record["patient"]["finding"] = sanitize_finding(
        standard_record["patient"]["finding"]
    )
    return standard_record

def dictionary_walk(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            yield from dictionary_walk(value)
        else:
            yield (key, value)

def standard_to_metadata_format(standard_record, filenames):
    "Convert data in an interoperable format to the format in metadata.csv"
    all_rows = []
    images = standard_record["images"]
    standard_patient = standard_record["patient"]
    for image, filename in zip(images, filenames):
        patient_row = {}
        #Update with all entries. 'misc' will be removed on conversion to dataframe.
        for key, value in dictionary_walk(standard_patient):
            print(key, value)
            patient_row[key] = value
        patient_row["clinical_notes"] = (
            string_or_empty(patient_row.pop("clinical_history")) + " " + image["image_description"]
        )
        patient_row.update(standard_record["document"])
        modality = image["modality"]
        if modality == "X-ray":
            folder = "images"
        elif modality == "CT":
            folder = "volumes"
        else:
            raise ValueError
        patient_row["modality"] = modality
        patient_row["folder"] = folder
        patient_row["filename"] = filename
        print(patient_row)
        all_rows.append(patient_row)
    return all_rows


# In[13]:


import torch
import torchvision
import torchxrayvision as xrv
from tqdm import tqdm
import pandas as pd
import numpy as np
import dicom2nifti
import sys


# In[14]:


conda install pytorch torchvision -c pytorch


# In[15]:


conda install pytorch -c pytorch


# In[16]:


pip3 install torchvision


# In[ ]:


pip install torchxrayvision


# In[18]:


d_covid19 = xrv.datasets.COVID19_Dataset(views=["PA", "AP", "AP Supine"],
                                         imgpath="../images",
                                         csvpath="metadata.csv")
print(d_covid19)


# In[19]:


missing = []
for i in range(len(d_covid19)):
    idx = len(d_covid19)-i-1
    try:
        # start from the most recent
        a = d_covid19[idx]
    except KeyboardInterrupt:
        break;
    except:
        missing.append(d_covid19.csv.iloc[idx].filename)
        print("Error with {}".format(i) + d_covid19.csv.iloc[idx].filename)
        print(sys.exc_info()[1])


# In[18]:


missing


# In[3]:


for i in ['76093afc.jpg',
 'fd389adb.jpg',
 '48b98ca2.jpg',
 '5cc1a119.jpg',
 '81089cb4.jpg',
 '3c3a2a35.jpg',
 'beef3909.jpg',
 '8251e162.jpg',
 '47e3c811.jpg',
 '9ff4a125.jpg',
 '54da03cf.jpg',
 'f8a26220.jpg',
 'bf333f43.jpg',
 '851007d4.jpg',
 'a2e1a826.jpg',
 '8a81d526.jpg',
 '8abe5256.jpg',
 'f6cfef3f.jpg',
 '7998d604.jpg',
 '626eb0ef.jpg',
 'a3111116.jpg',
 '78e9a055.jpg',
 '839e1363.jpg',
 '7a17a765.jpg',
 '7929f04a.jpg',
 'd59d92b8.jpg',
 '91cda775.jpg',
 '7bb9327c.jpg',
 '981e154a.jpg',
 '40954b81.jpg',
 '5c87a1d7.jpg',
 'bf39b989.jpg',
 '924f9c14.jpg',
 '96da829b.jpg',
 'b4e9a53a.jpg',
 'b606e1d0.jpg',
 'd6b8d378.jpg',
 'ac00512e.jpg',
 '17ad0a56.jpg',
 '8e438fce.jpg',
 '3d388a98.jpg',
 'a7abee59.jpg',
  'cf35d0c4.jpg',
 '93fd0adb.jpg',
 '1bc3008e.jpg',
 '1930e42f.jpg',
 '3161e216.jpg',
 '0578e08b.jpg',
 'b10c49ca.jpg',
 '7a2d2695.jpg',
 'fce2b5d4.jpg',
 '59cb1744.jpg',
 '88267e40.jpg',
 'a132d8b6.jpg',
 '5f7a99b2.jpg',
 '563118e4.jpg',
 'bb4c4038.jpg',
 'bace1e45.jpg',
 'add529f3.jpg',
 '6f7008af.jpg',
 'b39206a9.jpg',
 '9a9b2393.jpg',
 'd22964a4.jpg',
 '6c5b3802.jpg',
 'c08a4f41.jpg',
 'ffe8b4cb.jpg',
 'f567c33c.jpg',
 '00d96e05.jpg',
 'd15bf071.jpg',
 '7a030330.jpg',
 '3c8a0876.jpg',
 'bd85e252.jpg',
 'b6e58409.jpg',
 '9f3f2d91.jpg',
 '6b5af975.jpg',
 'c9280a30.jpg']:
    print(i.replace(".jpg",""))


# In[30]:


csv = pd.read_csv("metadata.csv", dtype="str")


# In[31]:


csv = csv[~csv.filename.isin(missing)]


# In[25]:


csv


# In[26]:


ax = fd['finding'].value_counts(dropna=False).plot.pie(y='Finding', legend = True, autopct='%2.0f%%', figsize = (10,10), title = 'Finding Distribution')


# In[28]:


import pandas as pd
import sys, os
import numpy as np
import os
from pathlib import Path

metadata = pd.read_csv("metadata.csv")
images = sorted(Path("images").iterdir(), key=os.path.getmtime, reverse=True)

header = """
<html><head>
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap.min.css">
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap-theme.min.css">
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/js/bootstrap.min.js"></script>
</head>
<body>
<div class="container">
"""

footer = "</div></body></html>\n"

with open("database.htm","w") as f:
    f.write(header)
    for i, image in enumerate(images):
        print(image.name)
        m = metadata[metadata.filename == image.name]
        if m.iloc[0].modality != "X-ray":
            continue
        f.write("<div class='row'>")
        
        f.write("<div class='col-sm-6'>")
        f.write("<center>{}</center>".format(image.name))
        f.write("<img style='width:100%' src='../images/{}'>".format(image.name))
        f.write("</div>")
        
        
        f.write("<div class='col-sm-6'>")
        f.write(m.T.to_html(classes="table small") + "\n")
        f.write("</div>")
        f.write("</div>")
        f.write("<hr>")
        if i > 10:
            break
    f.write(footer)


# In[29]:



import torch
import torchvision
import torchxrayvision as xrv
from tqdm import tqdm
import sys

# print stats
for views in [["PA","AP"],["AP Supine"]]:
  print(xrv.datasets.COVID19_Dataset(views=views,
                                       imgpath="images",
                                       csvpath="metadata.csv"))

d_covid19 = xrv.datasets.COVID19_Dataset(views=["PA", "AP", "AP Supine"],
                                       imgpath="images",
                                       csvpath="metadata.csv")
print(d_covid19)

for i in tqdm(range(len(d_covid19))):
  idx = len(d_covid19)-i-1
  try:
      # start from the most recent
      a = d_covid19[idx]
  except KeyboardInterrupt:
      break;
  except:
      print("Error with {}".format(i) + d_covid19.csv.iloc[idx].filename)
      print(sys.exc_info()[1])


# In[36]:


from lxml.html import parse
from sys import stdin
import pandas as pd
import re
import json
import dicom2nifti
import numpy as np
import os
import skimage
import pydicom


# In[34]:


pip install dicom2nifti


# In[35]:


conda install -c conda-forge dicom2nifti


# In[37]:


case = "covid-19-pneumonia-40"


# In[54]:


get_ipython().system('wget -O case-{case}.htm  https://radiopaedia.org/cases/{case}')


# In[ ]:





# In[ ]:





# In[33]:



pip install sklearn


# In[34]:


import sklearn

print('The scikit-learn version is {}.'.format(sklearn.__version__))


# In[7]:


from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd


# In[38]:


fd = pd.read_csv('metadata.csv')


# In[39]:


print(fd.columns)


# In[40]:


pip install keras


# In[ ]:





# In[2]:


pip install tensorflow


# In[26]:


pip install dicom2nifti


# In[ ]:





# In[8]:


pip install keras_applications==1.0.4


# In[9]:


pip install keras_preprocessing==1.0.2 --no-deps


# In[8]:


import matplotlib
import numpy as np
from matplotlib import pyplot as plt


# In[9]:


Methods=['VGG-19', 'Inception v3', 'ResNet50', 'Xception']
Accuracy=[0.906, 0.905, 0.903, 0.9001]
Precision=[0.92, 0.915, 0.91, 0.90]
Sensitivity=[0.8975, 0.8753, 0.87, 0.8623]
F1=[0.92, 0.915, 0.91, 0.90]


# In[10]:


xpos = np.arange(len(Methods))
xpos


# In[44]:


plt.xticks(xpos,Methods)

plt.bar(xpos-0.1, Accuracy, width = 0.2, Label="Accuracy")

plt.legend()


# In[11]:


import numpy as np
import matplotlib.pyplot as plt

N = 5
Accuracy=(0.906, 0.905, 0.903, 0.9001, 0.9701)

Methods=['VGG-19', 'Inception v3', 'ResNet50', 'Xception', 'DCSL']
ind = np.arange(N)  # the x locations for the groups
width = 0.15       # the width of the bars

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
rects1 = ax.bar(ind, Accuracy, width, color='orange')


Precision=(0.92, 0.915, 0.91, 0.90, 0.9700)
rects2 = ax.bar(ind+width, Precision, width, color='seagreen')

Sensitivity=[0.8975, 0.8753, 0.87, 0.9107, 0.9709]
rects3 = ax.bar(ind+width+width, Precision, width, color='red')

F1=[0.92, 0.915, 0.91, 0.90, 0.9698]
rects4 = ax.bar(ind+width+width+width, Precision, width, color='blue')



ax.set_xticks(ind + width / 2)
ax.set_xticklabels( (Methods ))

ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), ('Accuracy', 'Precision','Sensitivity','F1-score'),bbox_to_anchor=(1.05, 1), loc=2 )


plt.show()


# In[13]:


from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K


# In[ ]:





# In[23]:


pip install smote_variants==0.1.12


# In[2]:


conda install -c conda-forge imbalanced-learn


# In[9]:


import smote-variants as sv


# In[19]:


import pandas as pd
df=pd.read_csv('metadata.csv')
df


# In[18]:


df['RT_PCR_positive'].isna().sum()


# In[16]:


df['RT_PCR_positive'].value_counts()


# In[26]:


import pandas as pd
import os
import shutil
#for positive data
FILE_PATH="metadata.csv"
IMAGE_PATH="images"


# In[27]:


df=pd.read_csv(FILE_PATH)
print(df.shape)


# In[28]:


df.head()


# In[29]:


TARGET_DIR="Dataset/Covid"
if not os.path.exists(TARGET_DIR):
    os.mkdir(TARGET_DIR)
    print("COVID Folder created")


# In[40]:


cnt = 0

for (i,row) in df.iterrows():
    if row["finding"]=="Pneumonia/Viral/COVID-19" and row["view"]=="PA":
        filename =row["filename"]
        image_path =os.path.join(IMAGE_PATH,filename)
        image_copy_path=os.path.join(TARGET_DIR,filename)
        shutil.copy2(image_path,image_copy_path)
        print("Moving image", cnt)
        cnt +=1
        


# In[34]:


df.finding.value_counts()


# In[42]:


#Sampling of images from Kaggle
import random
KAGGLE_FILE_PATH= "chest_xray/train/NORMAL"
TARGET_NORMAL_DIR="Dataset/Normal"


# In[44]:


image_names=os.listdir(KAGGLE_FILE_PATH)

images_names
# In[45]:


image_names


# In[46]:


random.shuffle(image_names)


# In[48]:


for i in range(142):
    image_name= image_names[i]
    image_path=os.path.join(KAGGLE_FILE_PATH, image_name)
    target_path=os.path.join(TARGET_NORMAL_DIR,image_name)
    shutil.copy2(image_path,target_path)
    print("Copying image", i)


# In[35]:


import pandas as pd
import shutil


# In[36]:


x=pd.read_csv("metadata.csv")


# In[37]:


COVID_CASES = [row['filename'] for index,row in x.iterrows() if row['RT_PCR_positive'] == 'Y']
Normal_CASES = [row['filename'] for index,row in x.iterrows() if row['RT_PCR_positive'] == 'Unclear']


# In[38]:


print(COVID_CASES)


# In[39]:


print(Normal_CASES)


# In[40]:


x['RT_PCR_positive'].value_counts()


# In[41]:


from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir("images") if isfile(join("images", f))]


# In[49]:


#onlyfiles


# In[48]:


COVID="COVID"
Normal="Normal"
for i in COVID_CASES:
    if str(i)!='nan':
        shutil.copy("images/"+str(i),"COVID")
        print("COVID/"+i)


# In[46]:


for i in COVID_CASES:
    if str(i)=='nan':
        print('yes')


# In[ ]:




