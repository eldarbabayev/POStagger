#!/usr/bin/env python3.5
import operator
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import urllib
import re
import math

''' Utility '''

def check_tag_for_pipes(tag):
    if re.match(r"(\w+)\|(\w+)", tag):
        tag = re.sub(r"(\w+)\|(\w+)", r"\1", tag)
    if re.match(r"(\w+)\|(\w+)", tag):
        tag = re.sub(r"(\w+)\|(\w+)", r"\1", tag)    
    return tag

def get_tag_array(tag_dict):
    tag_array = []
    for key, value in tag_dict.items():
        tag_array.append(key)
    return tag_array

''' Load the data '''

# Load just one file to see results
def load_one_testing_file(path):
    fo = open(path, "r")
    stat = os.stat(path)
    content = fo.read(stat.st_size)
    fo.close()
    return content

# Load the data and split it into training and test data
def load_all_data(path, number_of_test):
    all_folders = [f for f in os.listdir(path)]
    length = len(all_folders)
    all_content = []

    for i in range(1, length):
        files = [f for f in os.listdir(path + '/' + all_folders[i])]
        for one_file in files:
            file_path = path + '/' + all_folders[i] + '/' + one_file
            all_content = np.concatenate((all_content, load_and_parse_one_file(file_path)), axis = 0)
    content_path = "/Users/eldarbabayev/Desktop/computationaLinguistics/computationalling/data/all_content.POS"
    fo = open(content_path, "w")
    stat = os.stat(content_path)
    for item in all_content:
        fo.write("%s" % item)
    fo.close()
    length_of_content = len(all_content)
    training_data = []
    test_data = []

    range_test_data_start = int(round(length_of_content * number_of_test * 0.1))
    range_test_data_end = range_test_data_start + int(round(length_of_content * 0.1)) - 4
    ninety_percent_of_data = int(round(length_of_content * 0.9))

    # load the training data
    for i in range(0, range_test_data_start):
        training_data.append(all_content[i])
    for i in range(range_test_data_end + 1, length_of_content):
        training_data.append(all_content[i])
    # load the test data
    for i in range(range_test_data_start, range_test_data_end):
        test_data.append(all_content[i])
    return training_data, test_data

    
''' Write to files '''

def write_to_testing_file(array):
    test_content_path = "/Users/eldarbabayev/Desktop/computationaLinguistics/computationalling/data/test_content.POS"
    fo = open(test_content_path, "w")
    stat = os.stat(test_content_path)
    for item in array:
        fo.write("%s" % item)
    fo.close()

''' Pre-processing of the data '''

# Section is divided by ====..
def divide_by_sections(content):
    content = re.sub(r"=+", "SECTION", content)
    array_of_sections = content.split("SECTION")
    return array_of_sections

def remove_unnecessary_new_lines(array_of_sections):
    for i in range(len(array_of_sections)-1,-1,-1):
        if re.match(r"^[\n]+$", array_of_sections[i]):
            array_of_sections.pop(i)
    return array_of_sections

# Divide the array by sentences
def divide_by_sentences(array):
    newarray = []
    for i in range(len(array)):
        if re.search(r"\./\.", array[i]) != None:
            elem = array[i]
            elem = elem.split("./.")
            newarray = newarray + elem
        else:
            newarray.append(array[i])
    return newarray

# Add START/STOP at beginning/end of sentences
def add_start_end(array):
    newarray = []
    for i in range(len(array)):
        elem = array[i]
        elem = "\n" + "START" + elem  +  "STOP"
        newarray.append(elem)
    write_to_testing_file(newarray)
    return newarray
        
# Remove quotes
def remove_quotes(array):
    newarray = []
    for i in range(len(array)):
        elem = array[i]
        elem = re.sub(r"''/''", r"", elem)
        elem = re.sub(r"``/``", r"", elem)
        elem = re.sub(r"'/''", r"", elem)
        elem = re.sub(r"`/``", r"", elem)
        newarray.append(elem)
    write_to_testing_file(newarray)
    return newarray

# Remove commas
def remove_commas(array):
    newarray = []
    for i in range(len(array)):
        elem = array[i]
        elem = re.sub(r",/,", r"", elem)
        elem = re.sub(r"2/,", r"", elem)
        newarray.append(elem)
    write_to_testing_file(newarray)
    return newarray

#remove colons and semicolons
def remove_colons_and_semicolons(array):
    newarray = []
    for i in range(len(array)):
        elem = array[i]
        elem = re.sub(r":/:", r"", elem)
        elem = re.sub(r";/:", r"", elem)
        newarray.append(elem)
    write_to_testing_file(newarray)
    return newarray

#remove bang and question mark
def remove_bang_question_hash(array):
    newarray = []
    for i in range(len(array)):
        elem = array[i]
        elem = re.sub(r"\?/\.", r"", elem)
        elem = re.sub(r"!/\.", r"", elem)
        elem = re.sub(r"#/#", r"", elem)
        newarray.append(elem)
    write_to_testing_file(newarray)
    return newarray

#remove --/:
def remove_double_dash_one_dash_triple_dot(array):
    newarray = []
    for i in range(len(array)):
        elem = array[i]
        elem = re.sub(r"--/:", r"", elem)
        elem = re.sub(r"\.\.\./:", r"", elem)
        elem = re.sub(r"-/:", r"", elem)
        newarray.append(elem)
    write_to_testing_file(newarray)
    return newarray

# Remove square brackets
def remove_square_brackets(array):
    newarray = []
    for i in range(len(array)):
        elem = array[i]
        elem = re.sub(r"\[", r"", elem)
        elem = re.sub(r"\]", r"", elem)
        newarray.append(elem)
    write_to_testing_file(newarray)
    return newarray

# Remove round brackets
def remove_round_brackets(array):
    newarray = []
    for i in range(len(array)):
        elem = array[i]
        elem = re.sub(r"\(/\(", r"", elem)
        elem = re.sub(r"{/\(", r"", elem)
        elem = re.sub(r"}/\)", r"", elem)
        elem = re.sub(r"\)/\)", r"", elem)
        newarray.append(elem)
    write_to_testing_file(newarray)
    return newarray        

# Remove new lines within sentences
def remove_new_lines_withing_sentences(array):
    newarray = []
    for i in range(len(array)):
        elem = array[i]
        elem = re.sub(r"[\n]+", r" ", elem)
        newarray.append(elem)
    write_to_testing_file(newarray)
    return newarray

# Only keep one whitespace between word-tags
def remove_blank_spaces_but_one_between_wordtags(array):
    newarray = []
    for i in range(len(array)):
        elem = array[i]
        elem = re.sub(r"[\s]+", r"\n", elem)
        newarray.append(elem)
    write_to_testing_file(newarray)
    return newarray    

# Remove \nSTART\nSTOP element from array
def remove_empty_sentence_from_array(array):
    newarray = []
    for i in range(len(array)):
        if re.match(r"\nSTART\nSTOP", array[i]):
            # Do nothing
            newarray = newarray
        else:
            newarray.append(array[i])
    write_to_testing_file(newarray)
    return newarray

# Remove newline on the first line of file
def remove_newline_from_beginning_of_each_sentence(array):
    newarray = []
    for i in range(len(array)):
        elem = array[i]
        elem = re.sub("^\n", "", elem)
        newarray.append(elem)
    return newarray

# Create array of word/tag sequences
def merge_all_sentences_into_array(array):
    newarray = []
    for i in range(len(array)):
        if re.search(r"\n", array[i]) != None:
            elem = array[i]
            elem = elem.split("\n")
            newarray = newarray + elem
        else:
            newarray = newarray + array[i]
    return newarray    

# convert 3\/4/CD to 34/CD
def convert_ratio_to_number(array):
    newarray = []
    for i in range(len(array)):
        if re.search(r"([0-9]+)\\/([0-9]+)(\/CD)", array[i]) != None:
            elem = array[i]
            elem = re.sub(r"([0-9]+)\\/([0-9]+)(\/CD)", r"\1\2\3", elem)
            newarray.append(elem)
        else:
            newarray.append(array[i])
    return newarray

# convert word1\/word2/tag to word1/tag
def convert_two_words_into_one(array):
    newarray = []
    for i in range(len(array)):
        if re.search(r"([\w-]+)\\/([\w-]+)(\/[A-Z$]+)", array[i]) != None:
            elem = array[i]
            elem = re.sub(r"([\w-]+)\\/([\w-]+)(\/[A-Z$]+)", r"\1\3", elem)
            newarray.append(elem)
        else:
            newarray.append(array[i])
    return newarray    

# convert S*/NNP&P/NN to S&P/NNP
def convert_word_and_word_into_one(array):
    newarray = []
    for i in range(len(array)):
        if re.search(r"([\w]+)\*/([A-Z]+)&([\w]+)/([A-Z]+)", array[i]) != None:
            elem = array[i]
            elem = re.sub(r"([\w]+)\*/([A-Z]+)&([\w]+)/([A-Z]+)", r"\1&\3/\2", elem)
            newarray.append(elem)
        else:
            newarray.append(array[i])
    return newarray

''' Load and parse one file '''

def load_and_parse_one_file(path):
    return convert_two_words_into_one(
           convert_word_and_word_into_one(
           convert_two_words_into_one(
           convert_ratio_to_number(
           merge_all_sentences_into_array(
           remove_newline_from_beginning_of_each_sentence(
           remove_empty_sentence_from_array(
           remove_blank_spaces_but_one_between_wordtags(
           remove_new_lines_withing_sentences(
           remove_double_dash_one_dash_triple_dot(
           remove_colons_and_semicolons(
           remove_bang_question_hash(
           remove_commas(
           remove_quotes(
           remove_round_brackets(
           remove_square_brackets(
           add_start_end(
           divide_by_sentences(
           remove_unnecessary_new_lines(
           divide_by_sections(
           load_one_testing_file(path)))))))))))))))))))))

''' Create appropriate dictionaries '''

def create_dictionary_of_tags_and_word_and_tags(training_data):
    tag_dict = {}
    word_tag_dict = {}
    startkey = 'START'
    stopkey = 'STOP'
    count = 0
    for elem in training_data:
        count = count + 1
        if (elem == stopkey):
            number_of_stops = tag_dict.get(stopkey, 0)
            tag_dict[stopkey] = number_of_stops + 1
        elif (elem == startkey):
            number_of_starts = tag_dict.get(startkey, 0)
            tag_dict[startkey] = number_of_starts + 1
        else:
            [word, tag] = elem.split("/")
            tag = check_tag_for_pipes(tag)
            key = word.lower() + "/" + tag
            tag_dict[tag] = tag_dict.get(tag, 0) + 1
            word_tag_dict[key] = word_tag_dict.get(key, 0) + 1
    return tag_dict, word_tag_dict

def create_consequent_tags_dictionary(training_data):
    startkey = 'START'
    stopkey = 'STOP'
    cons_tag_dict = {}
    for i in range(0, len(training_data) - 1):
        if (training_data[i] == startkey):
            [word, next_tag] = training_data[i+1].split("/")
            next_tag = check_tag_for_pipes(next_tag)
            key = startkey + "/" + next_tag
            cons_tag_dict[key] = cons_tag_dict.get(key, 0) + 1
        elif (training_data[i] == stopkey):
            if (training_data[i+1] == startkey):
                key = stopkey + "/" + startkey
                cons_tag_dict[key] = cons_tag_dict.get(key, 0) + 1
            else:
                [word, next_tag] = training_data[i+1].split("/")
                next_tag = check_tag_for_pipes(next_tag)
                key = stopkey + "/" + next_tag
                cons_tag_dict[key] = cons_tag_dict.get(key, 0) + 1
        else:
            if (training_data[i+1] == stopkey):
                [word, prev_tag] = training_data[i].split("/")
                prev_tag = check_tag_for_pipes(prev_tag)
                key = prev_tag + "/" + stopkey
                cons_tag_dict[key] = cons_tag_dict.get(key, 0) + 1
            else:
                [word, prev_tag] = training_data[i].split("/")
                [word, next_tag] = training_data[i+1].split("/")
                prev_tag = check_tag_for_pipes(prev_tag)
                next_tag = check_tag_for_pipes(next_tag)
                key = prev_tag + "/" + next_tag
                cons_tag_dict[key] = cons_tag_dict.get(key, 0) + 1
    return cons_tag_dict

''' Create log probabilities dictionary '''

def compute_probabilities(word_tag_dict, just_tag_dict, consequent_tags_dict):
    word_tag_prob = {}
    word_tag_prob_log = {}
    consequent_tags_prob = {}
    consequent_tags_prob_log = {}
    
    for key, value in word_tag_dict.items():
        [word, tag] = key.split("/")
        count_of_tag = just_tag_dict[tag]
        prob = value / count_of_tag
        word_tag_prob[key] = prob
        word_tag_prob_log[key] = math.log(prob)
        
    for key, value in consequent_tags_dict.items():
        [prev_tag, next_tag] = key.split("/")
        count_of_tag = just_tag_dict[prev_tag]
        prob = value / count_of_tag
        consequent_tags_prob[key] = prob
        consequent_tags_prob_log[key] = math.log(prob)

    return word_tag_prob_log, consequent_tags_prob_log

''' Viterbi Algorithm '''

def viterbi(test_data, cons_tags_prob_log, word_tag_prob_log, tag_dict):
    tag_array = get_tag_array(tag_dict)
    number_of_tags = len(tag_dict)

    test_data_size = len(test_data)

    # Initialise score and backpointer arrays
    score = [[0 for x in range(test_data_size)] for x in range(number_of_tags)]
    backpointer = [[0 for x in range(test_data_size)] for x in range(number_of_tags)]

    # Initialise the array of predictions
    best_tagging = [0 for x in range(test_data_size)]

    # Initialise
    for i in range(0, number_of_tags):
        tag = tag_array[i]
        word_and_tag = test_data[0] + "/" + tag
        start_and_tag = "START" + "/" + tag
        if word_and_tag in word_tag_prob_log:
            if start_and_tag in cons_tags_prob_log:
                score[i][0] = word_tag_prob_log[word_and_tag] + cons_tags_prob_log[start_and_tag]
            else:
                cons_tags_prob_log[start_and_tag] = math.log(1/(len(cons_tags_prob_log)))
                score[i][0] = word_tag_prob_log[word_and_tag] + cons_tags_prob_log[start_and_tag]
        else:
            if start_and_tag in cons_tags_prob_log:
                # Do laplacian smoothing, add word_and_tag to dict and compute probs
                word_tag_prob_log[word_and_tag] = math.log(1/(len(word_tag_prob_log)))
                score[i][0] = word_tag_prob_log[word_and_tag] + cons_tags_prob_log[start_and_tag]
            else:
                cons_tags_prob_log[start_and_tag] = math.log(1/(len(cons_tags_prob_log)))
                word_tag_prob_log[word_and_tag] = math.log(1/(len(word_tag_prob_log)))
                score[i][0] = word_tag_prob_log[word_and_tag] + cons_tags_prob_log[start_and_tag]
                
    # Induction
    for j in range(1, test_data_size):
        for i in range(0, number_of_tags):
            # find max prev score
            max_val = float("-inf")
            max_k = 0
            for k in range(0, number_of_tags):
                tag = tag_array[k] + "/" + tag_array[i]
                word_and_tag = test_data[j] + "/" + tag_array[i]
                if tag in cons_tags_prob_log:
                    if word_and_tag in word_tag_prob_log:
                        current = score[k][j-1] + cons_tags_prob_log[tag] + word_tag_prob_log[word_and_tag]
                    else:
                        # Do laplacian smoothing, add word_and_tag to dict and compute probs
                        word_tag_prob_log[word_and_tag] = math.log(1/(len(word_tag_prob_log)))
                        current = score[k][j-1] + word_tag_prob_log[word_and_tag] + cons_tags_prob_log[tag]

                else:
                    if word_and_tag in word_tag_prob_log:
                        # Do laplacian smoothing, add tag to dict and compute probs
                        cons_tags_prob_log[tag] = math.log(1/(len(cons_tags_prob_log)))
                        current = score[k][j-1] + word_tag_prob_log[word_and_tag] + cons_tags_prob_log[tag]
                    else:
                        cons_tags_prob_log[tag] = math.log(1/(len(cons_tags_prob_log)))
                        word_tag_prob_log[word_and_tag] = math.log(1/(len(word_tag_prob_log)))
                        current = score[k][j-1] + word_tag_prob_log[word_and_tag] + cons_tags_prob_log[tag]

                if (current > max_val):
                    # update max value and index
                    max_val = current
                    max_k = k
            # update score and backpointer arrays
            score[i][j] = max_val
            backpointer[i][j] = max_k

    # Back tracing the best tagging
    max_tn = float("-inf")
    max_i = 0
    for i in range(0, number_of_tags):
        if (max_tn < score[i][test_data_size-1]):
            max_tn = score[i][test_data_size-1]
            max_i = i
            
    best_tagging[test_data_size-1] = max_i
    
    for j in range(test_data_size-2,0, -1):
        best_tagging[j] = backpointer[best_tagging[j+1]][j+1]

    prediction = []
    for i in range(len(best_tagging)):
        prediction.append(tag_array[best_tagging[i]])

    return prediction
    
# convert word/tag to just word
def get_test_data_and_tags_from_ground_truth(ground_truth):
    test_data = []
    tags = []
    for elem in ground_truth:
        if elem == 'STOP':
            #do nothing..
            test_data = test_data
        elif elem == 'START':
            #do nothing..
            test_data = test_data
        else:
            [word, tag] = elem.split("/")
            test_data.append(word)
            tags.append(tag)
    return test_data, tags

# compute classification accuracy
def compute_classification_accuracy(ground_truth, prediction):
    accuracy = 0
    print(prediction)
    print(ground_truth)
    ground_truth = [x for x in ground_truth if x != 'STOP']
    ground_truth = [x for x in ground_truth if x != 'START']
    for i in range(len(ground_truth)):
        if (ground_truth[i] == prediction[i]):
            accuracy = accuracy + 1
    average = (accuracy/len(ground_truth))*100
    return average

def do_cross_validation(path):
    accuracy_list = []
    #    for i in range(0, 10):
    for i in range(0, 10):
        print(i)
        training_data, ground_truth = load_all_data(path, i)
        test_data, tags = get_test_data_and_tags_from_ground_truth(ground_truth)
        tag_dict, word_tag_dict = create_dictionary_of_tags_and_word_and_tags(training_data)
        cons_tag_dict = create_consequent_tags_dictionary(training_data)
        word_tag_prob_log, cons_tag_prob_log = compute_probabilities(word_tag_dict, tag_dict, cons_tag_dict)
        prediction = viterbi(test_data, cons_tag_prob_log, word_tag_prob_log, tag_dict)
        accuracy_list.append(compute_classification_accuracy(tags, prediction))
        
    final_accuracy = sum(accuracy_list) / len(accuracy_list)
    return final_accuracy

''' Running the program '''

# Main function, start of program
if __name__ == "__main__":
    # Testing on one sentence
    path = "/Users/eldarbabayev/Desktop/computationaLinguistics/computationalling/WSJ-2-12/05/WSJ_0515.POS"
    load_and_parse_one_file(path)

    # Testing on 90 percent of data
    path = "/Users/eldarbabayev/Desktop/computationaLinguistics/computationalling\
/WSJ-2-12new"
    print(do_cross_validation(path))




    
