import sys
import json

# When storing model into file using json.dumps, having dict key as tuple throws error. Hence forming string
def form_dict_key(word_1, word_2):
    return str(word_1) + ":" + str(word_2)

# Retrieving back the dict key from string
def retrieve_word_tag(key):
    splits = key.rsplit(":", 1)
    return splits[0], splits[1]

unique_tags_count_dict = {}
unique_word_count_dict = {}
unique_word_tag_count_dict = {}
unique_tag_unique_word_count_dict = {}
tag_transition_count_dict = {}
transition_prob_dict = {}
emission_prob_dict = {}
number_tag_dict = {}
number_tag = []

# Keep counts to compute transition and emission probabilities
def update_count_dictionaries(word, tag, prev_tag):
    count = 0
    if word in unique_word_count_dict:
        count = unique_word_count_dict[word]
    unique_word_count_dict[word] = count + 1
    count = 0
    if tag in unique_tags_count_dict:
        count = unique_tags_count_dict[tag]
    unique_tags_count_dict[tag] = count + 1
    count = 0
    key_word_tag = form_dict_key(word, tag)
    if key_word_tag in unique_word_tag_count_dict:
        count = unique_word_tag_count_dict[key_word_tag]
    else:
        if tag not in unique_tag_unique_word_count_dict:
            unique_tag_unique_word_count_dict[tag] = set(word)
        else:
            unique_tag_unique_word_count_dict[tag].add(word)
    unique_word_tag_count_dict[key_word_tag] = count + 1 
    count = 0
    key_prev_tag_tag = form_dict_key(prev_tag, tag)
    if key_prev_tag_tag in tag_transition_count_dict:
        count = tag_transition_count_dict[key_prev_tag_tag]
    tag_transition_count_dict[key_prev_tag_tag] = count + 1
    if word.isnumeric():
        count = 0
        if tag in number_tag_dict:
            count = number_tag_dict[tag]
        number_tag_dict[tag] = count + 1

# compute open class tags based on tags associated with large number of unique words. Keep top 4 of them
def compute_open_class_tags():
    open_class_candidates = []
    for k in sorted(unique_tag_unique_word_count_dict, key=lambda k: len(unique_tag_unique_word_count_dict[k]), reverse=True):
        open_class_candidates.append(k)
    return open_class_candidates[0:4]

# Compute transition probabilities. Laplace Smoothing is used on top of it.
def compute_transition_probabilities():
    all_tags = set(unique_tags_count_dict.keys())
    for tag_t_minus_1 in all_tags:
        for tag_t in all_tags:
            key_tag_t_minus_1_tag_t = form_dict_key(tag_t_minus_1, tag_t)
            if key_tag_t_minus_1_tag_t not in transition_prob_dict:
                if key_tag_t_minus_1_tag_t in tag_transition_count_dict:
                    prob_tag_t_given_tag_t_minus_1 = (tag_transition_count_dict[key_tag_t_minus_1_tag_t] +1)/(unique_tags_count_dict[tag_t_minus_1] + len(all_tags))
                else:
                    prob_tag_t_given_tag_t_minus_1 = 1/(unique_tags_count_dict[tag_t_minus_1] + len(all_tags))
                transition_prob_dict[key_tag_t_minus_1_tag_t] = prob_tag_t_given_tag_t_minus_1

# Compute emission probabilities
def compute_emission_probabilities():
    all_words = set(unique_word_count_dict.keys())
    all_tags = set(unique_tags_count_dict.keys())
    for word in all_words:
        for tag in all_tags:
            key_word_tag = form_dict_key(word, tag)
            if key_word_tag not in emission_prob_dict:
                if key_word_tag in unique_word_tag_count_dict:
                    prob_word_given_tag = unique_word_tag_count_dict[key_word_tag]/unique_tags_count_dict[tag]
                else:
                    prob_word_given_tag = 0
                emission_prob_dict[key_word_tag] = prob_word_given_tag

# We introduce a special start state (START_DUMMY) and end state (END_DUMMY) tags 
# to better capture the probablities of each tag being at the start/end of sentence

# To make computations easy, we have END_WORD as the only word that END_DUMMY tag emits
# For words that are number, we limit the tags possible only to top most tag used for numbers
def main(file_path):
    number_of_sentences = 0
    with open(file_path) as fp:
        sentences = fp.readlines()
        number_of_sentences = len(sentences)
        for sentence in sentences:
            sentence = sentence.rstrip()
            prev_tag = "START_DUMMY"
            words = sentence.split(" ")
            first_word = True
            for word in words:
                data = word.rsplit('/', 1)
                update_count_dictionaries(data[0], data[1], prev_tag)
                if first_word: 
                    first_word = False
                prev_tag = data[1]
            update_count_dictionaries("END_WORD","END_DUMMY", prev_tag)
    unique_tags_count_dict["START_DUMMY"] = number_of_sentences
    unique_tags_count_dict["END_DUMMY"] = number_of_sentences
    compute_transition_probabilities()
    compute_emission_probabilities()
    open_class_candidates = compute_open_class_tags()
    number_tag = [max(number_tag_dict, key=number_tag_dict.get)]
    parameters_dict = {}
    parameters_dict["transition_probabilities"] = transition_prob_dict
    parameters_dict["emission_probabilities"] = emission_prob_dict
    parameters_dict["known_tags_count"] = unique_tags_count_dict
    parameters_dict["known_words_count"] = unique_word_count_dict
    parameters_dict["open_class"] = open_class_candidates
    parameters_dict["number_tag"] = number_tag


    with open('hmmmodel.txt', 'w', encoding='utf8') as file:
         file.write(json.dumps(parameters_dict, ensure_ascii=False))

if __name__=="__main__":
    file_path = sys.argv[1]
    main(file_path)
