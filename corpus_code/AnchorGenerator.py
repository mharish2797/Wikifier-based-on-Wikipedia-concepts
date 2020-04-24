#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:01:12 2020

@author: harish
"""
"""
Anchor Generator

<b> Import Libraries </b>
"""
import string
from nltk.corpus import stopwords
import spacy
from csv import reader, writer
import csv
import sys
import os

csv.field_size_limit(sys.maxsize)
stopword_set = set(stopwords.words('english'))

"""
<b> Indicate the Input-Output directory paths </b>
"""
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

'''
### Methods to generate Anchors

#### Using NGRAM Words for Anchors
- After cleaning the input, generate all possible n-grams.
- For each n-gram generated from the input, check if it is an anchor and store it in a dictionary along with its start and end position.
- Construct a inverted anchor dictionary with (start,end) as key and anchor string as value. 
- Eliminate all sub-mention anchors from the inverted anchor dictionary and return the dictionary.

#### Using Noun phrases for Anchors

- Instead of generating all possible candidates from a given text, we use the noun phrases in a text as Anchors
- We use Spacy tool to extract Noun phrases from a text and use them directly as anchors
- Punctuation are not removed from the text

### Reducing Ambiguous Anchors

**Entropy of the mention** is the amount of uncertainty regarding the link target given the fact that its anchor text.

   **Entropy** `H(a) = – sum ([(ni/n) * log(ni/n)])`

If this entropy is above a user-specified threshold (e.g. 3 bits), we completely ignore the mention as being too ambiguous to be of any use. For mentions that pass this heuristic, we sort the target pages in decreasing order of `ni` and use only the top few of them (e.g. top 20) as candidates in our mention-concept graph.

### Advanced Anchor filtering

- Ignore candidates for which `ni` itself is below a certain threshold (e.g. `ni` < 2), the idea being that if such a phrase occurs only once as the anchor text of a link pointing to that candidate, this may well turn out to be noise and is best disregarded.

- The Wikifier can also be configured to ignore certain types of concepts based on their Wikidata class membership. This can be useful to exclude from consideration Wikipedia pages that do not really correspond to what is usually thought of as entities (e.g. “List of…” pages).

- Ignore any mention that consists entirely of stopwords and/or very common words (top 200 most frequent words in the Wikipedia for that particular language).
'''

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
'''
**Reading the Input file**
'''
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



entropy_limit = 3
flags = dict()

flags["input"] = input_file

# Allowed values nouns, ngrams, all
flags["process"] = "nouns"

flags["entropy"] = (corpus_files["anchor-entropy"], entropy_limit)

wiki_anchors = extract_anchors(flags, corpus_files["anchor-id"], wiki_files["wiki_anchors"])