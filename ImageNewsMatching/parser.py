import json
f=open("final_caption.txt")
res={}
for line in f:
    kv=line.split(':')
    res[kv[0]]=kv[1]

with open('output.json','w') as outfile:
    json.dump(res, outfile)
