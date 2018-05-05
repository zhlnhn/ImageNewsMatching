import pke
import os
import glob
import json
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
#Normalize the weight of each candidates
def normalize(keyphrase):
    total = 0
    for k,v in keyphrase:
        total += v
    for i in range(len(keyphrase)):
        v = float(keyphrase[i][1])/total
        tmp = (keyphrase[i][0],v)
        keyphrase[i] = tmp
    return keyphrase
'''
Use the keyphrase method to extract the keywords
Use both TF-IDF and Topic Clustering method
Then combine this two method to get keyphrases
'''
def getKeyphrases(file_name,N):
    extr = pke.TopicRank(input_file=file_name)
    extr_2 = pke.TfIdf(input_file=file_name)
    extr.read_document(format='raw')
    extr.candidate_selection()
    extr.candidate_weighting()
    extr_2.read_document(format='raw')
    extr_2.candidate_selection()
    extr_2.candidate_weighting()
    keyphrase = extr.get_n_best(n=2*N)
    keyphrase = normalize(keyphrase)
    keyphrases2 = extr_2.get_n_best(n=2*N)
    keyphrases2 = normalize(keyphrases2)
    keyphrase += keyphrases2
    sorted(keyphrase,key = min)
    return keyphrase[0:N]


def getKeyPhrase(fPath="."):
    os.chdir(fPath)
    #create a dictionary to store keyphrases for each news
    keyphrase_lst = {}
    #set N = 20, to choose best 20 keyphrases to describe the news
    N = 20
    #Open file, read each news document and extract keyphrases
    for i,file_name in enumerate(glob.glob('*.txt')):
        name = "news"
        keyphrase = getKeyphrases(file_name,N)
        keydic = {}
        name += file_name[4:5]
        for word in keyphrase:
            keydic[word[0]] = word[1]
        keyphrase_lst[name] = keydic
        #print(keyphrase_lst)
        lst=[v for v,_ in keyphrase]
        print name+' : '+' '.join(lst)
    return keyphrase_lst

    #write the result into a json file

if __name__=="__main__":
    currpath=os.getcwd()
    keyphrase_lst=getKeyPhrase("test_data2")
    os.chdir(currpath)
    #print(keyphrase_lst)
    with open('keywords.json','w') as outfile:
        json.dump(keyphrase_lst, outfile)
