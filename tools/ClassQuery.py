# coding:utf-8
import xml.etree.cElementTree as ET
import os
from collections import Counter


def count(pathdir):
    category = []
    path = pathdir
    for index, xml in enumerate(os.listdir(path)):
        print(str(index) + ' xml: ' + xml)
        root = ET.parse(os.path.join(path, xml))
        objects = root.findall('object')
        category += [ob.find('name').text for ob in objects]
    print(Counter(category))
    total_num = sum([value for key, value in Counter(category).items()])
    print('total_num:', total_num)


if __name__ == '__main__':
    pathdir = '/media/zrway/8T/xmj/project/DataSet/kitti/Annotations'
    count(pathdir)