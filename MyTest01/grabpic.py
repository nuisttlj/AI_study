import os
import urllib.request as request
import urllib.parse as parse
import re

header = \
    {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
        "referer": "https://image.baidu.com"
    }
url = "https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord={word}&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=0&word={word}&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&cg=girl&pn={pageNum}&rn=30&gsm=1e00000000001e&1490169411926="
keyword = input("请输入搜索关键字：")

keyword = parse.quote(keyword, "utf-8")

n = 0
j = 1
start_num = j

img_dir = r"C:\man ear"
if os.path.isdir(img_dir) != True:
    os.makedirs(img_dir)
if os.listdir(img_dir) != []:
    name_list = os.listdir(img_dir)
    j = max([int(i.replace(".jpg", '')) for i in name_list]) + 1
    start_num = j
while (n < 3000):
    error = 0
    n += 30
    url1 = url.format(word=keyword, pageNum=str(n))
    req = request.Request(url1, headers=header)
    req = request.urlopen(req)

    try:
        html = req.read().decode("utf-8")
    except:
        print("error!")
        error = 1
        print("error page: " + str(n))
    if error == 1:
        continue

    p = re.compile(r"thumbURL.*?\.jpg")
    s = p.findall(html)

    # with open("bgpic_test1.txt", "a") as f:
    for i in s:
        i = i.replace('thumbURL":"', '')
        print(i)
        # f.write(i)
        # f.write("\n")
        store_name = os.path.join(img_dir, "{num}.jpg".format(num=j))
        request.urlretrieve(i, store_name)
        j += 1
        # f.close()
print("total grab_pic is: " + str(j - start_num + 1))
