#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:15:42 2020

@author: harish
"""
'''
## Pagerank Generator

In order to compute the Pagerank for the generated graph, we are using <a href = "https://graph-tool.skewed.de/static/doc/quickstart.html" >Graph_tool </a> instead of NetworkX library as it is extremely faster.

<b> Import Libraries </b>
'''

import graph_tool as gt
from graph_tool import centrality as ct
from csv import reader, writer
import numpy as np
import os

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

class Graph:
    
    def __init__(self, bipartite_edges, concept_edges):
        
        self.graph = gt.Graph()
        self.pagerank = dict()
        self.anchor_dictionary = dict()
        self.massive_dict = dict()
        self.counter = 0
        edges, weights = self.get_edges_weights(bipartite_edges, bipartite_edges)
        self.edges = np.array(edges)
        self.weights = np.array(weights)
        
        self.graph.add_edge_list(self.edges, hashed=True, string_vals=True)
        self.ew = self.graph.new_edge_property("double")
        self.ew.a = self.weights 
        self.graph.ep['edge_weight'] = self.ew
        self.pr_result = list()
        
    def calculate_pageranks(self):
        pagerank_dict = ct.pagerank(self.graph, weight = self.ew)
        result_dict = dict()

        for anchor in self.anchor_dictionary:
            result_dict[anchor] = [(concept, pagerank_dict[self.massive_dict[concept]]) for concept in self.anchor_dictionary[anchor]]
            result_dict[anchor].sort(key = lambda x: x[1], reverse = True)
        self.pagerank = result_dict
        self.pr_result = [(anchor, concept, score) for anchor in self.pagerank for concept, score in self.pagerank[anchor]]
    
    def get_anchor_dictionary(self, ac_edges):
        for anchor, concept in ac_edges:
            if anchor not in self.anchor_dictionary:
                self.anchor_dictionary[anchor] = [concept]
            else:
                self.anchor_dictionary[anchor].append(concept)
        
    def edge_indexer(self, data):
        edges = list()
        weights = list()
        for a,b,c in data:
            for item in [a, b]:
                if item not in self.massive_dict:
                    self.massive_dict[item] = self.counter
                    self.counter += 1
            edges.append([a,b])
            weights.append(c)
        return edges, weights
    
    def get_edges_weights(self, bipartite_edges, concept_edges):
        ac_edges, ac_weights = self.edge_indexer(bipartite_edges)
        self.get_anchor_dictionary(ac_edges)
        cc_edges, cc_weights = self.edge_indexer(concept_edges)
        return ac_edges+cc_edges, ac_weights+cc_weights
    
    def output_writer(self, file_name):
        with open(file_name, 'w', newline='\n') as write_obj:
            csv_writer = writer(write_obj)
            for data in self.pr_result:
                csv_writer.writerow(data)
        return self.pr_result

def get_edges_data(filename):
    edges = list()
    with open(filename, newline='\n') as csvfile:
        red = reader(csvfile, delimiter=CORPUS_DELIMITER)
        for a,b,c in red:
            edges.append([a, b, float(c)])
    return edges

try:
    wiki_bipartite_edges
except:
    wiki_bipartite_edges = get_edges_data(wiki_files["wiki_bipartite_edges"])
    
try:
    wiki_concept_edges
except:
    wiki_concept_edges = get_edges_data(wiki_files["wiki_concept_edges"])


graph_object = Graph(wiki_bipartite_edges, wiki_concept_edges)
graph_object.calculate_pageranks()

wiki_pagerank = graph_object.output_writer(wiki_files["wiki_pagerank"])