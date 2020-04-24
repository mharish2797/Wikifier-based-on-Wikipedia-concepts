#!/usr/bin/env python
# coding: utf-8

# # Wikifier

# <b> The Wikifier consists of stages as follows: </b>
# 1. Anchor generation
# 2. Bipartite Graph edges generation
# 3. Concept-Concept edges generation
# 4. Pagerank generation

# ## Anchor Generator

# <b> Import Libraries </b>

# In[1]:


import string
from nltk.corpus import stopwords
import spacy
from csv import reader, writer
import csv
import sys
import os

csv.field_size_limit(sys.maxsize)
stopword_set = set(stopwords.words('english'))


# <b> Indicate the Input-Output directory paths </b>

# In[2]:


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


# ### Methods to generate Anchors
# 
# #### Using NGRAM Words for Anchors
# - After cleaning the input, generate all possible n-grams.
# - For each n-gram generated from the input, check if it is an anchor and store it in a dictionary along with its start and end position.
# - Construct a inverted anchor dictionary with (start,end) as key and anchor string as value. 
# - Eliminate all sub-mention anchors from the inverted anchor dictionary and return the dictionary.
# 
# #### Using Noun phrases for Anchors
# 
# - Instead of generating all possible candidates from a given text, we use the noun phrases in a text as Anchors
# - We use Spacy tool to extract Noun phrases from a text and use them directly as anchors
# - Punctuation are not removed from the text
# 
# ### Reducing Ambiguous Anchors
# 
# **Entropy of the mention** is the amount of uncertainty regarding the link target given the fact that its anchor text.
# 
#    **Entropy** `H(a) = – sum ([(ni/n) * log(ni/n)])`
# 
# If this entropy is above a user-specified threshold (e.g. 3 bits), we completely ignore the mention as being too ambiguous to be of any use. For mentions that pass this heuristic, we sort the target pages in decreasing order of `ni` and use only the top few of them (e.g. top 20) as candidates in our mention-concept graph.
# 
# ### Advanced Anchor filtering
# 
# - Ignore candidates for which `ni` itself is below a certain threshold (e.g. `ni` < 2), the idea being that if such a phrase occurs only once as the anchor text of a link pointing to that candidate, this may well turn out to be noise and is best disregarded.
# 
# - The Wikifier can also be configured to ignore certain types of concepts based on their Wikidata class membership. This can be useful to exclude from consideration Wikipedia pages that do not really correspond to what is usually thought of as entities (e.g. “List of…” pages).
# 
# - Ignore any mention that consists entirely of stopwords and/or very common words (top 200 most frequent words in the Wikipedia for that particular language).

# In[7]:


class AnchorExtractor():

    def __init__(self, anchor_id_file):
        self.anchors = list()
        self.elastic_index = self.get_anchor_id(anchor_id_file)
        
    def get_anchor_id(self, anchor_id_file):
        result = dict()
        with open(anchor_id_file, "r") as fileptr:
            temp_list = fileptr.readlines()
            for line in temp_list:
                temp = line.strip().split(CORPUS_DELIMITER, 1)
                if len(temp) > 1:
                    result[temp[1]] = temp[0]
        return result
    
    def word_to_id(self, anchors):
        return [ self.elastic_index[phrase] for phrase in anchors if self.is_in_elastic_index(phrase)]
    
    def set_anchors(self):
        self.anchors = self.word_to_id(set(self.anchors))
        
    def get_anchors(self):
        return self.anchors
    
    def filter_stopwords(self):
        temp = [ phrase for phrase in self.anchors if phrase.lower() not in stopword_set and not phrase.isnumeric()]
        result = list()
        for phrase in temp:
            filtered_phrase = self.phrase_filter(phrase)
            if filtered_phrase != None:
                result.append(filtered_phrase)
        self.anchors = result
     
    #Entropy
    def entropy_reader(self, entropy_file):
        result = dict()
        with open(entropy_file, newline='\n') as csvfile:
            red = reader(csvfile, delimiter= CORPUS_DELIMITER)
            for row in red:
                result[row[0]] = float(row[1])
        return result
    
    def entropy_filter(self, entropy_tuple):
        file, limit = entropy_tuple
        entropy_dict = self.entropy_reader(file)
        temp = self.get_anchors()
        self.anchors = [anchor for anchor in temp if entropy_dict[anchor] < limit]
    
    #Noun Anchors
    def is_in_elastic_index(self,string):
        return string in self.elastic_index  
    
    def phrase_filter(self, phrase):
        punct_set = set(string.punctuation)
        list_of_words = phrase.split(" ")
        for i, val in enumerate(phrase.split()):
            if val not in punct_set and val.lower() not in stopword_set:
                if i==0:
                    return phrase
                return " ".join(list_of_words[i:])
            
    def get_noun_anchors(self, input):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(input)
        nouns = set()
        for chunk in doc.noun_chunks:
            phrase = self.phrase_filter(chunk.text.strip())
            if phrase != None and not phrase.isnumeric():
                nouns.add(phrase)
        self.anchors = [anchor for anchor in nouns if self.is_in_elastic_index(anchor)]
    
    #N-gram Anchors
    def clean(self, input):
        punc_dict = dict.fromkeys(string.punctuation)
        self.translator = str.maketrans(punc_dict)
        for character in self.translator:
            self.translator[character] = " "   
        input = input.strip().translate(self.translator)
        input = " ".join(input.split())
        return input

    def generate_ngrams(self,string):
        string_list = string.split()
        candidate_grams = dict()
        seen_grams = dict()
        n = len(string_list)
        for i in range(n):
            for j in range(i+1,n+1):
                new_string = " ".join(string_list[i:j])
                if new_string in seen_grams:
                    if new_string in candidate_grams:
                        candidate_grams[new_string].append((i,j-1))
                else:
                    seen_grams[new_string] = 1
                    if self.is_in_elastic_index(new_string):
                        #print(new_string)
                        candidate_grams[new_string] = [(i,j-1)]

        return candidate_grams,seen_grams

    def generate_inverted_anchor_grams(self,candidate_grams):
        index_grams = dict()
        for key,values in candidate_grams.items():
            for co_ordinates in values:
                index_grams[co_ordinates] = key
        return index_grams

    def eliminate_sub_mentions(self, index_grams):
        iterator = list(index_grams.keys())
        iterator = sorted(iterator, key=lambda element: (element[1], -element[0]))
        flag = False
        n = len(iterator)
        for i in range(n-1):
            a,b = iterator[i]
            c,d = iterator[i+1]
            if a>=c and b<=d:
                del index_grams[(a,b)]
                flag = True
        if flag:
            return self.eliminate_sub_mentions(index_grams)
        return index_grams


# **Reading the Input file**

# In[8]:


def input_reader(filename):
    input = ""
    with open(filename, "r") as fileptr:
        line_list = fileptr.readlines()
        for line in line_list:
            input += " " + line.strip()
    return input

def anchor_writer(data, file_name):
    with open(file_name, 'w', newline='\n') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(data)


# In[9]:


def extract_anchors(flags, anchor_id_file, output_file):
    
    anchor_extractor_object = AnchorExtractor(anchor_id_file)
    input = input_reader(flags["input"])
    if flags["process"] == "ngrams" or flags["process"] == "all":
        clean_input = anchor_extractor_object.clean(input)
        
        #generate candidate anchor grams
        candidate_grams,seen_grams = anchor_extractor_object.generate_ngrams(clean_input)
        
        #Construct inverted index dictionary
        index_grams = anchor_extractor_object.generate_inverted_anchor_grams(candidate_grams)
        
        #generate the mentions
        anchor_grams = anchor_extractor_object.eliminate_sub_mentions(index_grams.copy())
        anchor_extractor_object.anchors = [word for word in list(anchor_grams.values()) ]
        anchor_extractor_object.filter_stopwords()
        anchor_extractor_object.set_anchors()
        ngram_anchors = set(anchor_extractor_object.get_anchors())
        
    if flags["process"] == "nouns" or flags["process"] == "all":
        anchor_extractor_object.get_noun_anchors(input)
        anchor_extractor_object.set_anchors()
        noun_anchors = set(anchor_extractor_object.get_anchors())
    
    if flags["process"] == "all":
        anchor_extractor_object.anchors = list( ngram_anchors.intersection(noun_anchors) )
            
    if flags["entropy"][1] > 0:
        anchor_extractor_object.entropy_filter(flags["entropy"])
    
    result_anchors = anchor_extractor_object.get_anchors()
    anchor_writer(result_anchors, output_file)
    return result_anchors


# In[33]:


entropy_limit = 3
flags = dict()

flags["input"] = input_file

# Allowed values nouns, ngrams, all
flags["process"] = "nouns"

flags["entropy"] = (corpus_files["anchor-entropy"], entropy_limit)


# In[11]:


wiki_anchors = extract_anchors(flags, corpus_files["anchor-id"], wiki_files["wiki_anchors"])


# In[12]:


wiki_anchors


# ## Bipartite Edges Generator

# 1. For each mention in the wiki_anchors, get all the corresponding wikipedia URI that it points to.
# 2. For each mention, URI pair create an edge (m,c) this is stored in `anchor-concepts.csv`
# 3. Assign the value for each edge as,
# 
#         P(m→c) =  number of hyperlinks with m pointing to c
#         		  __________________________________________
#         		   
#                   number of hyperlinks with m as anchor text

# <b> Import Libraries </b>

# In[3]:


import csv
import sys
from csv import writer

csv.field_size_limit(sys.maxsize)


# In[13]:


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


# In[14]:


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


# In[15]:


bipartite_object = BipartiteEdgeGenerator(wiki_anchors, corpus_files["anchor-concepts"])
wiki_bipartite_edges = bipartite_object.output_writer(wiki_files["wiki_bipartite_edges"])


# In[16]:


wiki_bipartite_edges[0]


# ## Concept-Concept Edges Generator

# If `Lc` is the set of wikipedia URI that points to c and N is the number of concepts in wikipedia then Semantic relatedness SR between concepts c and c' is
# 
#    `SR(c, c′) = 1 − [(log max{|Lc |, |Lc′|} − log|Lc ∩ L c′ |) / (log N − log min{|Lc |, |Lc′ |)]`
#     
# 
# 1. For each wikipedia URI in the graph generated calculate the Semantic relatedness SR with every other wikipedia URI in the graph
# 2. For each (c, c') in the graph, if `SR(c,c')` > 0, then create the edge `c→c'` and assign the value for edge as,
# 
#     `P(c→c') = SR(c, c') / [For all c'' sum(SR(c, c''))]`

# <b> Import Libraries </b>

# In[4]:


from itertools import combinations
from numpy import log10 as log
from csv import reader, writer
import csv
import sys

csv.field_size_limit(sys.maxsize)


# In[20]:


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


# In[21]:


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


# In[ ]:


concept_edge_object = ConceptEdgeGenerator(corpus_files["concept-concepts"], candidate_concepts)
concept_edge_object.add_concept_concept_edges()
wiki_concept_edges = concept_edge_object.output_writer(wiki_files["wiki_concept_edges"])   


# In[ ]:


wiki_concept_edges[0]


# ## Pagerank Generator

# In order to compute the Pagerank for the generated graph, we are using <a href = "https://graph-tool.skewed.de/static/doc/quickstart.html" >Graph_tool </a> instead of NetworkX library as it is extremely faster.

# <b> Import Libraries </b>

# In[5]:


import graph_tool as gt
from graph_tool import centrality as ct
from csv import reader, writer
import numpy as np


# In[14]:


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


# In[15]:


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


# In[16]:


graph_object = Graph(wiki_bipartite_edges, wiki_concept_edges)
graph_object.calculate_pageranks()

wiki_pagerank = graph_object.output_writer(wiki_files["wiki_pagerank"])


# In[18]:


wiki_pagerank[-1]


# ## Final Output Generator

# <b> Import Libraries </b>

# In[6]:


from csv import reader, writer
import sys
import csv

csv.field_size_limit(sys.maxsize)


# In[8]:


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


# In[9]:


def id_reader(filename):
    result = dict()
    with open(filename, newline='\n') as csvfile:
        red = reader(csvfile, delimiter=CORPUS_DELIMITER)
        for a,b in red:
            result[a] = b
    return result


# In[10]:


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


# In[11]:


CONCEPT_LIMIT = 5

anchor_dict = id_reader(corpus_files["anchor-id"])
concept_dict = id_reader(corpus_files["concept-id"])

pagerank_dict = pagerank_reader(wiki_files["wiki_pagerank"], CONCEPT_LIMIT)
wiki_output = get_readable_output(anchor_dict, concept_dict, pagerank_dict)
final_output_writer(wiki_files["wiki_output"], wiki_output)


# In[12]:


wiki_output


# In[ ]:




