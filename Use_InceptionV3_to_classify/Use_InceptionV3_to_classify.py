import cv2
from tkinter import messagebox
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input

cap = cv2.VideoCapture(0) #0��ܨϥ�defult�����Y

pre_trained_model = InceptionV3(weights="imagenet")
while True:
 
  ret, frame = cap.read() # ret : capture ���S�����\ , frame : ��i�v��

  if ~ret is True:
    messagebox.showinfo("Error","No video was capture.")
    break
  
  frame = cv2.resize(frame, (299,299))
  width = int(frame.shape[0])
  height = int(frame.shape[1])
  
  frame = np.expand_dims(frame, axis = 0) # (None, (299,299,3))
  frame = preprocess_input(frame)

  prediction = pre_trained_model.predict(frame) # �ϥ�pretrained model ���@�w��
  
  object_predict = imagenet_utils.decode_predictions(prediction) # �ϥιw���qimage net�����Ҥ�������������r
  print('object_name', object_predict[0][0][1])
  print('Accuracy = ', object_predict[0][0][2])
  
  frame = np.squeeze(frame) #��妸(�j�p��1)�����ץh��
  font = cv2.FONT_HERSHEY_SIMPLEX #�]�w�r��
  
  # putText���Ѽƨ̧Ǭ� (�Ϥ��A��ܦr��A�r����ܪ���m�A�r��A��r�Y���ҡA�C��(�� ��-��-��)�Athickness�A�u���˦�)
  frame = cv2.putText(frame, object_predict[0][0][1] , (width - width//3, height - height//3), font, 0.6, (0,150,0), 1, cv2.LINE_AA)
  frame = cv2.putText(frame, 'Accuracy = ' + str(round(object_predict[0][0][2] * 100, 2)) + "%" , (width - width//2 , height - height//4), font, 0.4, (0,150,0), 1, cv2.LINE_AA)
  
    
  frame = cv2.resize(frame, (500,500)) # ��j���
  cv2.imshow("frame",frame)
 
  #��Q�h�X
  if cv2.waitKey(1) == ord('q'): # 1 �O��1�@�� # ord('q')�A ���"���UQ��"���ʧ@ 
    break

cap.release() # ��Ӭ۾��������v�n�^��
cv2.destroyAllWindows() # �����Ҧ�����
