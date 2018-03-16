# calculate tfidf and output to json
import os
import nltk as nltk
import json
import math

postinglist = {}

def  computeTfidf(postinglist, len_files):
    N = len_files
    for key in postinglist:
        postinglist[key]['tfidf'] = postinglist[key]['term_freq'] * math.log(N/postinglist[key]['doc_freq'])


def preprocess(content):

    tokens = []

    sentences = nltk.sent_tokenize(content) # split content into sentences

    for sentence in sentences:

        tokens = tokens + nltk.word_tokenize(sentence) # merge two list

    return tokens

def updateDF(tokens, filename):

    docid = filename

    curr_doc_tf = {}

    unique_set = set()

    for token in tokens:

        unique_set.add(token)

        if token not in curr_doc_tf:

            curr_doc_tf[token]=0

        curr_doc_tf[token] = curr_doc_tf[token] + 1 # obtain current document term freq 

    for unique_term in unique_set: # only unique items will be added to postinglist

        if unique_term not in postinglist: # if that unique term have not added to postinglist

            postinglist[unique_term] = {'doc_freq':0,'term_freq':0,'posting':{}}

        postinglist[unique_term]['doc_freq'] = postinglist[unique_term]['doc_freq'] + 1 # every document will only add once
        
        postinglist[unique_term]['term_freq'] = postinglist[unique_term]['term_freq'] + curr_doc_tf[unique_term] # total term freq

        postinglist[unique_term]['posting'][docid] = curr_doc_tf[unique_term]

    return postinglist

def run():

    dataDir = os.getcwd() + '/data_2' # data directory

    len_files = len(os.listdir(dataDir))
    
    for filename in os.listdir(dataDir): # open the directory

        curr_file_dir = dataDir+'/'+filename # complete file dir of the data

        curr_file = open(curr_file_dir, 'r')  # read the file

        content = curr_file.read()

        tokens = preprocess(content) # preprocess 
        
        postinglist = updateDF(tokens,filename) # this will update the postinglist


    computeTfidf(postinglist,len_files)
    with open('result.json', 'w') as outfile:
        json.dump(postinglist, outfile)


run()
