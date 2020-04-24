#!/usr/bin/env python
# coding: utf-8

# # Attardi Output Data Processor

# The set of output files generated from running the script <a href="https://github.com/attardi/wikiextractor"> WikiExtractor.py </a> is further processed here to generate the set of Anchor-URL mappings to be used for building our Wikifier Corpus. 
# Here, BeautifulSoup is being used to extract all the anchor tags from the Web page.

# ### Import libraries

# In[1]:


import os
import json
import urllib.parse
from bs4 import BeautifulSoup


# ### Declare the directories for input/output
# - rootdir => Directory of Attardi processed output files
# - corpus_path => Directory of the corpus to store output files
# - output_file => File output generated which is of the form `<source_uri> <target_uri> <anchor string>`

# In[2]:


rootdir = "../output/"
corpus_path = "../corpus/"
output_file = corpus_path + "source_target_anchor.csv"
DELIMITER =' '
ERROR = 'error'


# Extract all the anchor links from a HTML file

# In[3]:


def get_anchors(text):
    soup = BeautifulSoup(text, "html.parser")
    anchor_list = soup.find_all('a')
    return anchor_list


# In[4]:


def proper_url(text):
    if text != None:
        text = text.replace(' ', '_')
        if 'http://' in text or 'https://' in text:
            return text
        else:
            target_page = "https://en.wikipedia.org/wiki/"+text
            return target_page
    else:
        return ERROR


# In[5]:


def file_writer(filename, content):
    opener = open(filename, "a")
    opener.write(content)
    opener.close


# In[6]:


for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        filename = os.path.join(subdir, file)
        
        #Parsing each wiki_files        
        opener = open(filename, "r")
        content = opener.readlines()
        
        for element in content:
            data = json.loads(element)
            title, text = urllib.parse.quote(data['title']), data['text']
            source_uri = proper_url(title)
            
            if source_uri == ERROR:
                continue
            
            for anchor in get_anchors(text):
                anchor_href = anchor.get('href')
                target_uri = proper_url(anchor_href)
                if target_uri == ERROR:
                    continue
                
                record = DELIMITER.join([source_uri,  target_uri, anchor.text]) + "\n"
                file_writer(output_file, record)
                
        opener.close()

