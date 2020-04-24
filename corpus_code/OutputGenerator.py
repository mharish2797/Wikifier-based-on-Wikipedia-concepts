#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:18:51 2020

@author: harish
"""
'''
## Final Output Generator

<b> Import Libraries </b>
'''
from csv import reader, writer
import sys
import csv
import os

csv.field_size_limit(sys.maxsize)

FILE_EXTENSION = ".csv"

CORPUS_PATH = "../corpus/"
CORPUS_DELIMITER = ","

corpus_files = dict()
corpus_file_list = ["anchor-id","concept-id", "concept-concepts", "anchor-concepts", "anchor-entropy"]
for filename in corpus_file_list:
    corpus_files[filename] = CORPUS_PATH + filename + FILE_EXTENSION
    
input_path = "../input/"
input_file = input_path + "input.txt"

WIKI_PATH = "../wiki_temp/"
if not os.path.exists(WIKI_PATH):
    os.makedirs(WIKI_PATH)
    
wiki_files = dict()
wiki_file_list = [ "wiki_anchors","wiki_bipartite_edges","wiki_concept_edges", "wiki_output", "wiki_pagerank" ]

for filename in wiki_file_list:
    wiki_files[filename] = WIKI_PATH + filename + FILE_EXTENSION

def get_readable_output(anchor_dict, concept_dict, pagerank_dict):
    result = list()
    for anchor in pagerank_dict:
        for concept in pagerank_dict[anchor]:
            result.append([anchor_dict[anchor], concept_dict[concept]])
    return result

def final_output_writer(file_name, items):
    with open(file_name, 'w', newline='\n') as write_obj:
        csv_writer = writer(write_obj)
        for data in items:
            csv_writer.writerow(data)

def id_reader(filename):
    result = dict()
    with open(filename, newline='\n') as csvfile:
        red = reader(csvfile, delimiter=CORPUS_DELIMITER)
        for a,b in red:
            result[a] = b
    return result

def pagerank_reader(filename, CONCEPT_LIMIT):
    result = dict()
    with open(filename, newline='\n') as csvfile:
        red = reader(csvfile, delimiter=',')
        for row in red:
            if row[0] not in result:
                result[row[0]] = [row[1]]
            else:
                if len(result[row[0]]) <= CONCEPT_LIMIT:
                    result[row[0]].append(row[1])
    return result

CONCEPT_LIMIT = 5

anchor_dict = id_reader(corpus_files["anchor-id"])
concept_dict = id_reader(corpus_files["concept-id"])

pagerank_dict = pagerank_reader(wiki_files["wiki_pagerank"], CONCEPT_LIMIT)
wiki_output = get_readable_output(anchor_dict, concept_dict, pagerank_dict)
final_output_writer(wiki_files["wiki_output"], wiki_output)