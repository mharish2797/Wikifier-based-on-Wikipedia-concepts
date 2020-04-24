#!/usr/bin/env python
# coding: utf-8

# # Wikifier Corpus Builder

# After running the `Attardi Output Data Processor`, the following script is executed to generate the following corpus data:
# 
# - anchor-id.csv => `<anchor_id>,<anchor_string>`
# - concept-id.csv => `<concept_id>,<concept_URI>`
# - anchor-concepts.csv => `<anchor_id>,<concept_id>`
# - anchor-entropy.csv => `<anchor_id>,<entropy_value>,<edge_value>`
# - concept-concepts.csv => `<concept_id>,<incoming_concept_id>`

# In[2]:


import sys
import csv
from itertools import chain
from numpy import log10 as log
from csv import reader, writer

csv.field_size_limit(sys.maxsize)


# ### Declare the directories for input/output
# - corpus_path => Directory of the corpus to store output files
# - input_file => File input which is of the form `<source_uri> <target_uri> <anchor string>`

# In[2]:


CORPUS_PATH = "../corpus/"
input_file = CORPUS_PATH + "source_target_anchor.csv"
DELIMITER = " "
FILE_EXTENSION = ".csv"

corpus_files = dict()
corpus_file_list = ["anchor-id","concept-id", "concept-concepts", "anchor-concepts", "anchor-entropy"]
for filename in corpus_file_list:
    corpus_files[filename] = CORPUS_PATH + filename + FILE_EXTENSION


# ### Corpus Processor
# Class that defines data structures and methods to collect corpus data that is being generated.

# In[4]:


class CorpusProcessor():
    
    def __init__(self):
        self.anchors, self.concepts = dict(), dict()
        self.cc, self.ac, self.entropy  = dict(), dict(), dict()
        self.counter = { "a": 0, "c": 0}
        
    def update_counter(self, element, key):
        if key == "a" and element not in self.anchors:
            self.anchors[element] = key + str(self.counter[key])
            self.counter[key] += 1
        elif key == "c" and element not in self.concepts:
            self.concepts[element] = key + str(self.counter[key])
            self.counter[key] += 1
    
    def update_target(self, source, target):
        if target not in self.cc:
            self.cc[target] = set([source])
        else:
            self.cc[target].add(source)
        
    def update_ac(self, anchor, target):
        if anchor not in self.ac:
            self.ac[anchor] = dict()
            self.ac[anchor][target] = 1
        else:
            if target in self.ac[anchor]:
                self.ac[anchor][target] += 1
            else:
                self.ac[anchor][target] = 1
            
    def update_ac_entropy(self):
        for anchor in self.ac:
            N = sum(self.ac[anchor].values())
            summer = 0
            for target in self.ac[anchor]:
                niN = self.ac[anchor][target]/N
                summer -= niN * log(niN)
                self.ac[anchor][target] = niN
            self.entropy[anchor] = summer


# ### Get input data

# In[5]:


def is_valid_record(line):
    if len(line)<3:
        return False
    return True

def get_input(filename):
    input_data = list()
    with open(filename, newline = "\n") as csvfile:
        records = reader(csvfile, delimiter=DELIMITER)
        for row in records:
            if is_valid_record(row):
                record = row[:2] + [ DELIMITER.join(row[2:]) ]
                parsed_record = [word.strip() for word in record]
                input_data.append(parsed_record)
    return input_data


# ### Parse input data and generate corpus data

# In[6]:


def parse_input(corpus_object, input_data):
    counter = 0
    total_count = len(input_data)
    for line in input_data:
        counter+=1
        if counter%1_000_000 == 0:
            print(counter, "/", total_count, " records processed")
            print(100*counter/total_count, "% Done")
        source_uri, target_uri, anchor_text = line
        corpus_object.update_counter(anchor_text, "a")
        for uri in [source_uri, target_uri]:
            corpus_object.update_counter(uri, "c")
        
        source_id, target_id = corpus_object.concepts[source_uri], corpus_object.concepts[target_uri]
        anchor_id = corpus_object.anchors[anchor_text]
        
        corpus_object.update_target(source_id, target_id)
        corpus_object.update_ac(anchor_id, target_id)
        
    corpus_object.update_ac_entropy()


# ### Write output data to files

# In[7]:


def write_data(filename, data):
    with open(filename, 'w', newline='\n') as csvfile:
        csv_writer = writer(csvfile)
        for record in data:
            csv_writer.writerow(record)

def object_writer(obj):
    write_data(corpus_files["anchor-id"], [(v,k) for k,v in obj.anchors.items()])
    write_data(corpus_files["concept-id"], [(v,k) for k,v in obj.concepts.items()])
    write_data(corpus_files["anchor-entropy"], [(k,v) for k,v in obj.entropy.items()])
    
    write_data(corpus_files["concept-concepts"], chain.from_iterable( [ [(k,v) for v in v_list ]for k,v_list in obj.cc.items() ]))
    write_data(corpus_files["anchor-concepts"], chain.from_iterable( [ [(k,c,v) for c, v in v_list.items()] for k,v_list in obj.ac.items() ]) )


# ### Main function

# In[ ]:


corpus_object =  CorpusProcessor()
input_data = get_input(input_file)

parse_input(corpus_object, input_data)
object_writer(corpus_object)

