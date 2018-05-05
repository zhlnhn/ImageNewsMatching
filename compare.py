import copy
import json
#import getKeyphrase

def comp(captions,keywords,events):
    correct = 0
    result={}
    for image,caption in captions.items():
        max=0
        max_id=0
        #print(caption)

        for news,keywords_ in keywords.items():
            count=0
            for keyword,confid in keywords_.items():
                for k in keyword.split():
                    if k in caption:
                        count+=confid
            score=count/len(caption.split())
            if score>max:
                max=score
                max_id=news
        print(image,max_id)
        result[image]=max_id
        if image[5:] == max_id[4:]:
            correct +=1
    print "Correct number:",correct
    return result

if __name__=="__main__":
    f1=open("output.json")
    f2=open("keywords.json")
    captions=json.load(f1)
    keywords=json.load(f2)
    comp(captions,keywords,events)
