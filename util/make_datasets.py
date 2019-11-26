import os
import json

'''
    labelme的数据整合方法：
    1、rename json数据中image_name成图片文件数据，将None和较大的image_data改成'-1'
    2、整合单个json到一个json文件中
'''


def rewrite_json_file(filepath, json_data):
    with open(filepath, 'w') as f:
        json.dump(json_data, f)
    f.close()


def rename_img_path(file_dir):
    for f in os.listdir(file_dir):
        if f.endswith('json'):
            with open(file_dir+'/'+f, 'rb') as f1:
                fileJson = json.load(f1)
                fileJson["imagePath"] = f.split('.')[0] + '.jpg'
                fileJson["imageData"] = '-1'
                # fileJson['shapes']["fill_color"] = '-1'
                # fileJson["line_color"] = '-1'
                rewrite_json_file('/Users/joash/PycharmProjects/Mask_RCNN/tv/'+f, fileJson)
        else:
            continue

def merge_json(file_dir):

    content = '{'

    for f in os.listdir(file_dir):
        print(f)
        if f.endswith('json'):
            with open(file_dir+'/'+f, 'rb') as f1:
                fileJson = json.load(f1)
                print(type(fileJson))
                print(f.split('.')[0])
                content+=f.split('.')[0]+':'+str(fileJson)+','
    content+='}'
    return content


file_dir = '/Users/joash/PycharmProjects/Mask_RCNN/tv/'
# rename_img_path(file_dir)

content = merge_json(file_dir)
print(content)

# f = open("/Users/joash/PycharmProjects/Mask_RCNN/images/ceiling/train/train_ceiling.json","r+")   #设置文件对象
# str1 = f.read()
# str2 = str1.replace("'", '"')
# print(str2)
# f.write(str2)