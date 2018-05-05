from lxml import etree
import xml.dom.minidom as minidom
import glob

'''
Read triggers and events from xml file for future matching system
'''
events = {}
for filename in glob.glob("*.xml"):
    doc = etree.parse(filename)
    memoryElem = doc.find('document')
    text = memoryElem.text        # element text
    entity_mention = memoryElem.get('entity_mention')
    trigger_lst = []
    if entity_mention == None:
        continue
    for argument in entity_mention:
        triggers = argument[text][0][1]
        keyname = filename[4:]
        keyname = keyname.replace(".xml","")
        trigger_lst(triggers)
    events[keyname] = trigger_lst
