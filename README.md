# YOLO
YOLO物件辨識
Python 上的 YOLO (darkflow) 基本安裝

我電腦是windows系統，直接使用Anaconda來進行Tensorflow跟python的安裝吧!!! 如果不知道怎麼使用Anaconda來安裝tensorflow跟python的，請參考這篇

P.S 如果有GPU的話要安裝Tensorflow-gpu，然後還要安裝CUDA，可以參考這篇 (Tensorflow-gpu的安裝當初花了我不少功夫…windows真的很不好用)

安裝好tensorflow跟python後，YOLO還有一個最重要的東西要安裝，那就是OpenCV了，如果要在python上安裝OpenCV，現在pypi上有可以直接的安裝包了，直接用pip 安裝 opencv-contrib-python 就可以了

pip install opencv-contrib-python
安裝好Tensorflow、Python、OpenCV後就可以開始安裝YOLO了

這裡我們安裝的是darkflow，這個是for python使用的(只支持YOLOv2)，我們先把darkflow從github clone下來

git clone https://github.com/thtrieu/darkflow

然後我們到clone下的資料夾進行pip 安裝，會出現error無法安裝

cd darkflow
pip install -e .

我們可以從error看出沒有Cython這個套件，因此我們用pip 來安裝這個套件

pip install Cython

安裝好Cython後，再一次安裝darkflow，便會成功安裝了

pip install -e .

接下來我們下載已經事先訓練好的weight(官方訓練的)放在darkflow下的bin目錄來測試yolo的安裝是否成功，下載好後輸入以下code來測試darkflow。

python flow --model cfg/yolo.cfg --load bin/yolov2.weights --imgdir sample_img/
如果沒有問題的話就會在sample_img裡面會有一個out資料夾，裡面圖片便會有物件偵測了，到這裡，Python上的YOLO就安裝完成啦!!! YO HO~~


YOLO object detection
也可以把yolo利用在影像或電腦上的攝影機

影像

python flow --model cfg/yolo.cfg --load bin/yolov2.weights --demo [audiofile] --saveVideo --gpu [0~1]
攝影機直接demo (把audiofile改成camera)

python flow --model cfg/yolo.cfg --load bin/yolov2.weights --demo camera --saveVideo --gpu [0~1]
自行利用自己data訓練YOLO

接下來就教大家怎麼使用darkflow來訓練自己的物件偵測吧~

首先 我們先來講怎麼準備訓練資料的部份


我們要準備很多想要訓練的圖片，這些圖片裡面必須要有妳要偵測的物件(這不是廢話嗎!!!
然後我們要將bounding box給標出來，也就是要將圖片裡面你要偵測的物件框框標出來，並且將bounding box標上標籤，也就是這張圖片的Annotation(註釋)，這樣的Dataset有分成二種，而這二種在YOLO都可以拿來Training，VOC Dataset跟COCO Dataset(如果不知道VOC跟COCO是什麼的可以點連結看介紹)，這裡我是用VOC來進行Training的，接下來我來介紹個很棒的tool，建立VOC 會非常的方便：labelimg
Labelimg

先附上大神的github：https://github.com/tzutalin/labelImg

接下來我來介紹怎麼使用Anaconda去安裝labelimg

首先，要先安裝pyqt，我有試過用pip intall去找過，但沒有這個套件，但是anaconda好像可以找得到這個套件，所以我直接用conda install ，輸入以下的code安裝pyqt

conda install pyqt=5

開始安裝pyqt需要的套件了
安裝好後把labelimg 下載回來(可能會花一點時間，沒有當掉不要緊張 哈哈)，然後直接執行labelimg，然後會發現沒有resources這個module

git clone https://github.com/tzutalin/labelImg
cd lableimg
python lableimg.py

直拉用pip安裝resources，後面會發現有很多套件還沒有安裝，我直接把要安裝的套件都列出來：

pip install resources
pip install requests
pip install staty
pip install lxml
把這些都安裝成功後，執行labelimg就會出現這樣的畫面了


接下來就很直觀的，左邊都有按扭可以直接選，有幾個小步驟可以分享給大家讓大家更快的標記訓練資料

先Change Save Dir，把標記資料的xml檔要放的path給選好
使用快捷鍵
W==>建立Bounding box

A==>上一張

D==>下一張

Ctril+S==>存XML檔(如果有做第1步的話，這裡就會直接存檔，不然每一次存檔都會跳出視窗問你要存到那裡)


我當初自行標記了300張左右的訓練資料只花了15分鐘，熟悉了其實會快非常多。

開始訓練

訓練資料準備好了，接下來就開始訓練自己的yolo model啦!!

1.把訓練資料放在對應的path上


2.再來就是修改cfg跟label.txt

為什麼要修改這個呢，因為如果現在我們要label的只有二種的話，那網路的架構的output就要有所改變，原本training的cfg是有80種label的，所以我們要去改config，這裡我用的是tiny-yolo，tiny-yolo就是比較少層跟比較少參數的network，這樣訓練相對會花比較少時間，打開cfg，有幾個要注意並更改的部份，在最後一層跟 region的部分要改二個地方(直接拉到最後就看得到了)：


改完後在darkflow裡面有個labels.txt也要改，把裡面的標籤改成妳要的Label名稱


這樣就完成事前設定啦!!!

接下來直接輸入以下的code就可以直接進行訓練了

python --model [model.cfg] --train --dataset [image path] --annotation [annotation path]

開始training
default的epoch數量是1000，然後data每經過2000個data會把目前的參數存成ckpt檔在ckpt的資料夾，epch跟一些參數設定可以在darkflow\darkflow\default.py裡面更改 (在cfg上更改不會有任何效果)


default setting
如果要在上次的記憶點重新訓練的話，只需要在中間加入load就可以了 (-1就是default的最後一個check point)

python flow --model [model.cfg] --load -1 --train --dataset [image path] --annotation [annotation path]
如果要看自己的model train的有沒有問題，就跟上面介紹的demo跟imgdir就可以了，如果成功偵測了，那就代表妳的YOLO訓練出來啦! 恭喜 YOHO~

結論

我最後的dataset數量是這樣 (有些照片是二個人一起拍的)

Total 250

Robert 168

Candy 211

然後train了約150個epoch左右(我有用GPU，但因為我的GPU是很舊的840M，記憶不夠大，所以batch size只能到4，train了一個晚上左右，train了20k左右的iteration)，loss從106降到1.2，然後就會開始有些震盪了， 資料量雖然不多，但是train出來的效果比我想像中的還好，基本的一些正面照跟低頭照也都可以偵測出來，影片也沒什麼問題，不過我一開始把bounding box 標得太大了，把很多一些不必要的feature都標進去了，所以一開始train出來的model在demo的時候完全偵測不出東西來，即使我train了1天(汗…)，後來重新再框一次bounding box，主要標記臉的部份，盡量少框一些不必要的feature，後來就成功偵測出來了，而這次也只train了一個晚上就可以成功detect了。

各位如果要在python上做YOLO的一些應用也是ok的，以下code很直觀的就可以直接在python上做一些應用了，要做predata process什麼的都可以，之後想要試著把YOLO放進手機裡面，如果有機會的話再寫一篇怎麼放吧(我自己應該也要花一些時間研究，畢竟我沒寫過APP，哈哈XD)

from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}

tfnet = TFNet(options)

imgcv = cv2.imread("./sample_img/sample_dog.jpg")
result = tfnet.return_predict(imgcv)
print(result)
參考資料

大神darkflow github

樓上的大陸翻譯 from CSDN 風吳痕

YOLO官網
