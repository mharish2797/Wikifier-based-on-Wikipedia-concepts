# Wikifier
Wikifier is a tool to annotate documents based on wikipedia concepts.<br>
<b>Dataset:</b> <a href="https://dumps.wikimedia.org/" target="_blank">Wikipedia dump</a><br>
<b>Data Extraction tool:</b> <a href= "https://github.com/attardi/wikiextractor" target="_blank">Attardi's WikiExtractor</a><br>
<b>Base research paper:</b> <a href="https://ailab.ijs.si/dunja/SiKDD2017/Papers/Brank_Wikifier.pdf" target="_blank"> Brank Wikifier</a><br>

## Steps to generate data corpus for the wikifier:
1. After processing the XML dump using Wikiextractor, run the `notebooks/Attardi Output Data Processor.ipynb` or `corpus_code/AttardiOutputDataProcessor.py` to generate intermediate data `corpus/source_target_anchor.csv` which will be roughly of size 12GB.
2. Now run `notebooks/Wikifier Corpus Builder.ipynb` or `corpus_code/WikifierCorpusBuilder.py` to generate the corpus dataset `corpus/["anchor-id","concept-id", "concept-concepts", "anchor-concepts", "anchor-entropy"]` each roughly having size between 150 MB - 700MB
3. The `corpus/source_target_anchor.csv` is no longer required and can be deleted if needed.

## Running the Wikifier:
1. Place the input data in `input/` folder and run the `notebooks/Wikifier.ipynb` or `corpus_code/Wikifier.py` to get the wikified results which is in `wiki_temp/wiki_output.csv`.
2. If each stage of the wikifier is to be visualized, run the `notebooks/Attardi Output Data Processor.ipynb` stage by stage and check the `wiki_temp` for the intermediate results or run the following python files in `corpus_code/` separately and see the output that is being generated.
    * AnchorGenerator.py
        * i/p: `input.txt`
        * corpus-input: [`anchor-id.csv`, `anchor-entropy.csv`]
        * o/p: `wiki_anchors.csv` 
    * BipartiteEdgeGenerator.py
        * i/p: `wiki_anchors.csv`  
        * corpus-input: `anchor-concepts.csv`
        * o/p: `wiki_bipartite_edges.csv`
    * ConceptEdgeGenerator.py
        * i/p: `wiki_bipartite_edges.csv` 
        * corpus-input: `concept-concepts.csv`
        * o/p: `wiki_concept_edges.csv`
    * PageRankGenerator.py
        * i/p: [`wiki_bipartite_edges.csv`, `wiki_concept_edges.csv`] 
        * o/p: `wiki_pagerank.csv`
    * OutputGenerator.py
        * i/p: `wiki_pagerank.csv`  
        * corpus-input: [`anchor-id.csv`, `concept-id.csv`]
        * o/p: `wiki_output.csv`