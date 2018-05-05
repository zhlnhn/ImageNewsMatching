import os
from eval import predict_captions
import sys
import json
from compare import comp as match_results
from getKeyphrase import getKeyPhrase
import argparse
from readxml import events
import shutil
def final_evaluation(folder='test_data'):
    print("=========start generate image captions=========")
    image_captions=predict_captions(folder)
    #print image_captions
    print("=========start generate news keyphrases=========")
    keyphrases=getKeyPhrase(folder)
    #print keyphrases
    #events=
    res=match_results(image_captions,keyphrases,events)
    return res
'''
def generate_folder(dir='final_matching_results',res,folder='test_data'):
    if not os.path.exists(dir):
        os.makedirs(dir)
    os.chdir(dir)
    for i in range(len(res)):
        new_dir="match"+str(i)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        os.chdir('../'+folder)
        shutil.copy(, '../'+dir+'/match'+str(i))
'''


if __name__=="__main__":
    #os.chdir("ImageCaptionNoCUDA")
    parser=argparse.ArgumentParser()
    parser.add_argument('--new_dataset',type=str,default='0',
                        help='1 represents new dataset, 0 represents provided dataset')
    parser.add_argument('--image_folder', type=str, default='',
                    help='If this is nonempty then will predict on the images in this folder path')
    args=parser.parse_args()
    if(args.new_dataset=='0'):
        final_evaluation()
    elif(args.new_dataset=='1'):
        res=final_evaluation(args.image_folder)
