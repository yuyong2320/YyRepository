import os

train_folders ='D:/视频人群计数数据集/mall_dataset/mall_testdata'
train_all_img_lists = []
for root, dirs, files in os.walk(train_folders):
    for file_name in files:
        if file_name.endswith('.jpg'):
            train_all_img_lists.append(os.path.join(root, file_name))
print(len(train_all_img_lists))
print(train_all_img_lists[0])
# print(train_all_img_lists)
for path in train_all_img_lists:
    # index = int(img_name.split('.')[0])


    print(path[-8:-4])  # 截取倒数第三位到结尾
    # a = int(os.path.basename(path).split('.')[0]) - 4898000060
    a = int(path[-8:-4])-1

    os.path.join(os.path.dirname(path),str(a)+'.jpg')
    #print(path)
    # print(os.path.join(os.path.dirname(path),str(a)+'.jpg'))
    os.rename(path,os.path.join(os.path.dirname(path),str(a)+'.jpg'))
    print("已经将"+path+"重命名成"+os.path.join(os.path.dirname(path),str(a)+'.jpg'))


