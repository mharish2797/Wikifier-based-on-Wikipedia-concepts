#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:14:03 2020

@author: harish
"""

'''

## Concept-Concept Edges Generator

If `Lc` is the set of wikipedia URI that points to c and N is the number of concepts in wikipedia then Semantic relatedness SR between concepts c and c' is

   `SR(c, c′) = 1 − [(log max{|Lc |, |Lc′|} − log|Lc ∩ L c′ |) / (log N − log min{|Lc |, |Lc′ |)]`
    

1. For each wikipedia URI in the graph generated calculate the Semantic relatedness SR with every other wikipedia URI in the graph
2. For each (c, c') in the graph, if `SR(c,c')` > 0, then create the edge `c→c'` and assign the value for edge as,

    `P(c→c') = SR(c, c') / [For all c'' sum(SR(c, c''))]`

<b> Import Libraries </b>
'''

from itertools import combinations
from numpy import log10 as log
from csv import reader, writer
import csv
import sys
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

class ConceptEdgeGenerator():

    def __init__(self, cc_file, candidate_concepts):
        self.cc_dict_file_name = cc_file
        self.cc_edges = list()
        self.concept_ids = candidate_concepts

    def read_cc_file(self, filename):
        cc_dict = dict()
        with open(filename, newline='\n') as csvfile:
            red = reader(csvfile, delimiter=CORPUS_DELIMITER)
            for s,t in red:
                if s in cc_dict:
                    cc_dict[s].add(t)
                else:
                    cc_dict[s] = set([t])
        return cc_dict

    def get_candidate_concept_ids(self, filename):
        concept_ids = set()
        with open(filename, newline='\n') as csvfile:
            red = reader(csvfile, delimiter=',')
            for row in red:
                concept_ids.add(row[1])
        return concept_ids
                
#   Concept - Concept Edges
    def augment_sr_to_concept(self, c1, c2, temp_concepts, SR):
        temp_concepts[c1] = temp_concepts[c1]+SR if c1 in temp_concepts else SR
        temp_concepts[c2] = temp_concepts[c1]+SR if c2 in temp_concepts else SR    
        return temp_concepts

    def get_predecessors(self):
        cc_lookup_dict = self.read_cc_file(self.cc_dict_file_name)
        self.logN = log(len(cc_lookup_dict))
        
        predecessors = dict()
        for cid in self.concept_ids:
            if cid in cc_lookup_dict:
                predecessors[cid] = cc_lookup_dict[cid]
        return predecessors
    
    def get_concept_concept_edge_values(self):
        temp_concepts = dict()
        result_dict = dict()
        
        predecessors = self.get_predecessors()
        
        for c1, c2 in combinations(self.concept_ids, 2):
            predecessors_c1 = predecessors[c1]
            predecessors_c2 = predecessors[c2]
            intersects = predecessors_c1.intersection(predecessors_c2)

            if len(intersects) > 0:
                SR = 1
                len_pred_c1 = len(predecessors_c1)
                len_pred_c2 = len(predecessors_c2)
                max_len = max(len_pred_c1, len_pred_c2)
                min_len = min(len_pred_c1,len_pred_c2)
                SR -= (log(max_len) - log(len(intersects))) / (self.logN - log(min_len))
                result_dict[(c1,c2)] = SR
                temp_concepts = self.augment_sr_to_concept(c1, c2, temp_concepts, SR)
                
        return result_dict, temp_concepts

    def add_concept_concept_edges(self):
        cc_dict, SR_count_dict = self.get_concept_concept_edge_values()
        for c1, c2 in cc_dict:
            SR_c1_c2 = cc_dict[(c1,c2)]
            
            self.cc_edges.append([c1, c2, SR_c1_c2/SR_count_dict[c1]])
            self.cc_edges.append([c2, c1, SR_c1_c2/SR_count_dict[c2]])

    def output_writer(self, file_name):
        with open(file_name, 'w', newline='\n') as write_obj:
            csv_writer = writer(write_obj)
            for data in self.cc_edges:
                csv_writer.writerow(data) 
        return self.cc_edges

def get_candidate_concepts(filename):
    concept_ids = set()
    with open(filename, newline='\n') as csvfile:
        red = reader(csvfile, delimiter=CORPUS_DELIMITER)
        for row in red:
            concept_ids.add(row[1])
    return concept_ids

try:
    wiki_bipartite_edges
except NameError:
    candidate_concepts = get_candidate_concepts(wiki_files["wiki_bipartite_edges"])
else:
    candidate_concepts = set( [c for a,c,v in wiki_bipartite_edges ] )

concept_edge_object = ConceptEdgeGenerator(corpus_files["concept-concepts"], candidate_concepts)
concept_edge_object.add_concept_concept_edges()
wiki_concept_edges = concept_edge_object.output_writer(wiki_files["wiki_concept_edges"])   