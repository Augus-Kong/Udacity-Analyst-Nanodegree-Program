
#使用代码构造样本
import xml.etree.cElementTree as ET
from collections import defaultdict
import re

osm_file = open("chicago.osm", "r") #先设定文件变量

street_type_re = re.compile(r'\S+\.?$', re.IGNORECASE) #创建匹配格式 pattern
street_types = defaultdict(int)  #?????

def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name) 
	if m:
		street_type = m.group()
		street_types[street_type] += 1 #?????

def print_sorted_dict(d):
	keys = d.keys()  #列出字典中所有的key值
	keys = sorted(keys, key=lambda s: s.lower()) 
    for k in keys:
		v = d[k]
		print "%s: %d" % (k, v) 

def is_street_name(elem):
	return (elem.tag == "tag") and (elem.attrib['k'] == "addr:street")

def audit():
	for event, elem in ET.iterparse(osm_file):
		if is_street_name(elem):
		audit_street_type(street_types, elem.attrib['v'])    
	print_sorted_dict(street_types)    

if __name__ == '__main__':
	audit()



#mapparser.py

import xml.etree.cElementTree as ET #导入库函数 ET
import pprint

def count_tags(filename):
	elem_dict = {}.fromkeys(('bounds','member','nd','node','osm','relation','tag','way')，0) 
	#创建空白字典，包含key值 且value值等于0
	for _, elem in ET.iterparse(filename, events=("start",)): #遍历数据文件
		if elem.tag in elem_dict: #通过筛选是否存在空字典中key值对应的标签
			elem_dict[elem.tag] += 1 #改变字典的value值
		else:
			elem_dict[elem.tag] = 1 #此处的[]使用 elem.tag 而没有使用 '' 符号    ????
	return elem_dict

def test():

	tags = count_tags('example.osm')
	pprint.pprint(tags)
	assert tags =={'bounds': 1,
	'member': 3,
'nd': 4,
'node': 20,
'osm': 1,
'relation': 1,
'tag': 7,
'way': 1}

if __name__ =="__main__":
	test()

# Iterating through Ways Tags

import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint


#练习： 标签类型

import xml.etree.cElementTree as ET
import pprint
import re
"""
Your task is to explore the data a bit more.
Before you process the data and add it into your database, you should check the
"k" value for each "<tag>" and see if there are any potential problems.

We have provided you with 3 regular expressions to check for certain patterns
in the tags. As we saw in the quiz earlier, we would like to change the data
model and expand the "addr:street" type of keys to a dictionary like this:
{"address": {"street": "Some value"}}
So, we have to see if we have such tags, and if we have any tags with
problematic characters.

Please complete the function 'key_type', such that we have a count of each of
four tag categories in a dictionary:
  "lower", for tags that contain only lowercase letters and are valid,
  "lower_colon", for otherwise valid tags with a colon in their names,
  "problemchars", for tags with problematic characters, and
  "other", for other tags that do not fall into the other three categories.
See the 'process_map' and 'test' functions for examples of the expected format.
"""


lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')


def key_type(element, keys):
    
    if element.tag == "tag":
        if lower.match(element.attrib['k']):
            keys["lower"] +=1
        elif lower_colon.match(element.attrib['k']):
            keys["lower_colon"] +=1
        elif problemchars.search(element.attrib['k']):
            keys["problemchars"] +=1
        else:
            keys["other"] +=1
    return keys



def process_map(filename):
    keys = {"lower": 0, "lower_colon": 0, "problemchars": 0, "other": 0}
    for _, element in ET.iterparse(filename):
        keys = key_type(element, keys)

    return keys

#练习：探索用户

import xml.etree.cElementTree as ET
import pprint
import re
"""
Your task is to explore the data a bit more.
The first task is a fun one - find out how many unique users
have contributed to the map in this particular area!

The function process_map should return a set of unique user IDs ("uid")
"""

def get_user(element):
    if 'uid' in element.attrib:
        return element.attrib['uid']


def process_map(filename):
    users = set()
    for _, element in ET.iterparse(filename):
        if get_user(element):
            users.add(get_user(element))

    return users


def test():

    users = process_map('example.osm')
    pprint.pprint(users)
    assert len(users) == 6



if __name__ == "__main__":
    test()



def test():
    # You can use another testfile 'map.osm' to look at your solution
    # Note that the assertion below will be incorrect then.
    # Note as well that the test function here is only used in the Test Run;
    # when you submit, your code will be checked against a different dataset.
    keys = process_map('example.osm')
    pprint.pprint(keys)
    assert keys == {'lower': 5, 'lower_colon': 0, 'other': 1, 'problemchars': 1}


if __name__ == "__main__":
    test()