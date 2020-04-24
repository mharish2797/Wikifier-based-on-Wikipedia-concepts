#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:09:47 2020

@author: harish
"""
'''
## Bipartite Edges Generator

1. For each mention in the wiki_anchors, get all the corresponding wikipedia URI that it points to.
2. For each mention, URI pair create an edge (m,c) this is stored in `anchor-concepts.csv`
3. Assign the value for each edge as,

        P(m→c) =  number of hyperlinks with m pointing to c
        		  __________________________________________
        		   
                  number of hyperlinks with m as anchor text

<b> Import Libraries </b>
'''
import csv
import sys
from csv import writer
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

class BipartiteEdgeGenerator:
    
    def __init__(self, wiki_anchors, anchor_concept_file):
        anchors = wiki_anchors
        anchor_concept_dict = self.get_anchor_concept_dict(anchors, anchor_concept_file)
        self.bipartite_edges = self.get_bipartite_edges(anchor_concept_dict)
    
    def get_anchor_concept_dict(self, anchors, anchor_concept_file):
        temp_dict = dict()
        with open(anchor_concept_file, "r") as fileptr:
            temp_list = fileptr.readlines()
            for line in temp_list:
                temp = line.strip().split(CORPUS_DELIMITER)
                if temp[0] in anchors:
                    if temp[0] in temp_dict:
                        temp_dict[temp[0]].append(temp[1:])
                    else:
                        temp_dict[temp[0]] = [temp[1:]]
        return temp_dict
        
    def get_bipartite_edges(self, anchor_concept_dict):
        result = list()
        for anchor in anchor_concept_dict:
            edge_list = anchor_concept_dict[anchor]
            result += [ [anchor] + edge for edge in edge_list ]
        return result
        
    def output_writer(self, file_name):
        with open(file_name, 'w', newline='\n') as write_obj:
            csv_writer = writer(write_obj)
            for data in self.bipartite_edges:
                csv_writer.writerow(data)
        temp = [[a,b,float(c)] for a,b,c in self.bipartite_edges]
        return temp

def get_wiki_anchors(wiki_anchor_file):
    anchors = list()
    with open(wiki_anchor_file, "r") as fileptr:
        temp_list = fileptr.readlines()
        for line in temp_list:
            anchors += line.strip().split(CORPUS_DELIMITER)
    return set(anchors)

try:
    wiki_anchors
except NameError:
    wiki_anchors = get_wiki_anchors(wiki_files["wiki_anchors"])

bipartite_object = BipartiteEdgeGenerator(wiki_anchors, corpus_files["anchor-concepts"])
wiki_bipartite_edges = bipartite_object.output_writer(wiki_files["wiki_bipartite_edges"])