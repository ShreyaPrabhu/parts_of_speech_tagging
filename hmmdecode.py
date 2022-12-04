import math
import sys
import json
import numpy as np

# When storing model into file using json.dumps, having dict key as tuple throws error. Hence forming string
def form_dict_key(word_1, word_2):
    return str(word_1) + ":" + str(word_2)

# load model
parameters_dict = None
with open('hmmmodel.txt') as f:
    parameters_dict = json.load(f)

transition_prob_dict = parameters_dict["transition_probabilities"]
emission_prob_dict = parameters_dict["emission_probabilities"]
unique_tags_count_dict = parameters_dict["known_tags_count"]
unique_word_count_dict = parameters_dict["known_words_count"]
open_class_candidates = parameters_dict["open_class"]
number_tag = parameters_dict["number_tag"]

all_known_words = unique_word_count_dict.keys()
all_known_tags = unique_tags_count_dict.keys()

# viterbi decoding algorithm for pos tagging
def viterbi_decoding(words, all_known_words, all_known_tags):
    column_names = ["START_WORD"]
    column_names.extend(words)
    column_names.append("END_WORD")
    
    index_word_dict = dict(enumerate(column_names))
    index_tag_dict = {k: v for v, k in enumerate(all_known_tags)}
    viterbi = np.zeros([len(index_tag_dict),len(index_word_dict)], dtype = np.float128)
    
    # set start tag probability to 1 for initial start state
    viterbi[index_tag_dict["START_DUMMY"], 0] = 1
    
    prev_word_index = 0
    word_iterate_index = list(index_word_dict.keys()).copy()
    word_iterate_index.remove(0)
    
    backtracker = np.zeros([len(index_tag_dict),len(index_word_dict)], dtype=np.dtype('U50'))
    for word_index in word_iterate_index:
        max_val = -math.inf
        max_prev_tag = None
        current_tag_inventory = all_known_tags
        
        if index_word_dict[word_index] not in all_known_words:
            if len(number_tag)>0 and index_word_dict[word_index].isnumeric():
                current_tag_inventory = number_tag
            else:
                current_tag_inventory = open_class_candidates
        
        for tag in current_tag_inventory:
            key_word_tag = form_dict_key(index_word_dict[word_index], tag)
            if key_word_tag in emission_prob_dict and emission_prob_dict[key_word_tag] == 0:
                viterbi[index_tag_dict[tag], word_index] = 0
            else:
                for prev_tag in all_known_tags:
                    value = viterbi[index_tag_dict[prev_tag], prev_word_index]*transition_prob_dict[form_dict_key(prev_tag, tag)]
                    if value > max_val:
                        max_val = value
                        max_prev_tag = prev_tag  
                viterbi[index_tag_dict[tag], word_index] = viterbi[index_tag_dict[max_prev_tag], prev_word_index]*transition_prob_dict[form_dict_key(max_prev_tag, tag)]*emission_prob_dict.get(key_word_tag, 1)
            backtracker[index_tag_dict[tag], word_index] = max_prev_tag
        prev_word_index = word_index
    return viterbi, backtracker, index_word_dict, index_tag_dict
    
# based on backtracker array, find the most probable pos tag sequence for the sentence
def form_path(words, viterbi, backtracker, index_word_dict, index_tag_dict):
    inv_tag_dict = {v: k for k, v in index_tag_dict.items()}
    end_tag = inv_tag_dict[np.argmax(viterbi[:, -1], axis = 0)]
    path = []
    path.append(end_tag)
    column_length = backtracker.shape[1]
    backtrack_path_list = list(range(1, column_length))
    backtrack_path_list.reverse()
    for val in backtrack_path_list:
        if(val == 1):
            break
        end_tag = backtracker[index_tag_dict[end_tag], val]
        path.append(end_tag)
    path.reverse()
    path.pop()
    sentence = ""
    for word, tag in zip(words, path):
        sentence = sentence + word + "/" + tag + " "
    sentence = sentence.rstrip()
    return sentence

def main(file_path):
    viterbi = None
    backtracker = None
    resulting_sentences = []
    with open(file_path) as fp:
        sentences = fp.readlines()
        for sentence in sentences:
            sentence = sentence.rstrip()
            words = sentence.split(" ")
            viterbi, backtracker,index_word_dict, index_tag_dict = viterbi_decoding(words, all_known_words, all_known_tags)
            result_sentence = form_path(words, viterbi, backtracker, index_word_dict, index_tag_dict)
            resulting_sentences.append(result_sentence)

    with open("hmmoutput.txt", "w") as outfile:
        outfile.write("\n".join(resulting_sentences))
    
if __name__=="__main__":
    file_path = sys.argv[1]
    main(file_path)
