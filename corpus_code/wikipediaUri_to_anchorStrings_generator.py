from csv import writer
FILE_EXTENSION = ".csv"

CORPUS_PATH = "../corpus/"
CORPUS_DELIMITER = ","

corpus_files = dict()
corpus_file_list = ["anchor-id","concept-id", "concept-concepts", "anchor-concepts", "anchor-entropy", "wikipediaURI-anchorString"]
for filename in corpus_file_list:
    corpus_files[filename] = CORPUS_PATH + filename + FILE_EXTENSION
    
def file_writer(data, file_name):
    with open(file_name, 'w', newline='\n', encoding="utf8") as write_obj:
        csv_writer = writer(write_obj)
        for record in data:
            csv_writer.writerow(record)
        
def read_id_data( filename):
    result = dict()
    with open(filename, "r",  encoding="utf8") as fileptr:
        temp_list = fileptr.readlines()
        for line in temp_list:
            temp = line.strip().split(CORPUS_DELIMITER, 1)
            if len(temp) > 1:
                result[temp[0]] = temp[1]
    return result

def read_uri_data(filename, anchor_id, concept_id):
    result = dict()
    with open(filename, "r",  encoding="utf8") as fileptr:
        temp_list = fileptr.readlines()
        for line in temp_list:
            temp = line.strip().split(CORPUS_DELIMITER)
            if len(temp) > 1:
                key, value = temp[:2]
                anchor = anchor_id[key]
                uri = concept_id[value]
                if uri in result:
                    result[uri].append(anchor)
                else:
                    result[uri] = [anchor]
    return result

anchor_id = read_id_data(corpus_files["anchor-id"])
concept_id = read_id_data(corpus_files["concept-id"])
anchor_concept = read_uri_data(corpus_files["anchor-concepts"], anchor_id, concept_id)

for k,val in anchor_concept.items():
    anchor_concept[k] = "|".join(val)
    
file_writer(anchor_concept.items(), corpus_files["wikipediaURI-anchorString"])

