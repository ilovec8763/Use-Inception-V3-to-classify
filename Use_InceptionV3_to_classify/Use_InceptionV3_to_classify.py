import cv2
from tkinter import messagebox
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input

cap = cv2.VideoCapture(0) #0表示使用defult的鏡頭

pre_trained_model = InceptionV3(weights="imagenet")
while True:
 
  ret, frame = cap.read() # ret : capture 有沒有成功 , frame : 單張影像

  if ~ret is True:
    messagebox.showinfo("Error","No video was capture.")
    break
  
  frame = cv2.resize(frame, (299,299))
  width = int(frame.shape[0])
  height = int(frame.shape[1])
  
  frame = np.expand_dims(frame, axis = 0) # (None, (299,299,3))
  frame = preprocess_input(frame)

  prediction = pre_trained_model.predict(frame) # 使用pretrained model 做作預測
  
  object_predict = imagenet_utils.decode_predictions(prediction) # 使用預測從image net的標籤中提取對應的文字
  print('object_name', object_predict[0][0][1])
  print('Accuracy = ', object_predict[0][0][2])
  
  frame = np.squeeze(frame) #把批次(大小為1)的維度去掉
  font = cv2.FONT_HERSHEY_SIMPLEX #設定字體
  
  # putText的參數依序為 (圖片，顯示字串，字串顯示的位置，字體，文字縮放比例，顏色(用 藍-綠-紅)，thickness，線條樣式)
  frame = cv2.putText(frame, object_predict[0][0][1] , (width - width//3, height - height//3), font, 0.6, (0,150,0), 1, cv2.LINE_AA)
  frame = cv2.putText(frame, 'Accuracy = ' + str(round(object_predict[0][0][2] * 100, 2)) + "%" , (width - width//2 , height - height//4), font, 0.4, (0,150,0), 1, cv2.LINE_AA)
  
    
  frame = cv2.resize(frame, (500,500)) # 放大顯示
  cv2.imshow("frame",frame)
 
  #按Q退出
  if cv2.waitKey(1) == ord('q'): # 1 是指1毫秒 # ord('q')， 表示"按下Q鍵"的動作 
    break

cap.release() # 把照相機的控制權要回來
cv2.destroyAllWindows() # 消除所有視窗
