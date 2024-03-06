from PIL import Image
import  os

path = os.getcwd() #获取当前路径
file_list = os.listdir()
for file in file_list:
    filename = os.path.splitext(file)[0]
    filexten = os.path.splitext(file)[1]
    if filexten == '.png' or '.jpg':
        I = Image.open(file)
        L = I.convert('L')
        L.save('huidu-' + file)