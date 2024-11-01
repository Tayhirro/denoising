import os

#获取目录
pathroot=os.getcwd()


 
#列出内容
contents = os.listdir('/path/to/directory')



#path
exists = os.path.exists('file_or_directory')

basename = os.path.basename(path)
dirname = os.path.dirname(path)

full_path = os.path.join('/path/to', 'file.txt')

#遍历
for root, dirs, files in os.walk('/path/to/directory'): #root是当前目录，dirs是当前目录下的子目录，files是当前目录下的文件
    for file in files:
        print(os.path.join(root, file))