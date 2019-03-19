import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):                                               # pytorch에 정의된 Dataset을 상속
    def __init__(self, img_path, gt_path, file_names, size_x, size_y):  # MyDataset class 생성자
        self.img_path = img_path                                        # 이미지 폴더 위치
        self.gt_path = gt_path                                          # ground true인 pix segmentation 폴더 위치
        self.file_names =file_names                                     # 파일 이름 리스트
        self.size_x = size_x                                            # resize할 크기
        self.size_y = size_y                                            # resize할 크기
    def __getitem__(self, index): # C++의 오버로딩?과 동일한 개념, index번째의 파일을 요청했을 때 return값
        img = cv2.imread(self.img_path+self.file_names[index][:-1])     # 코드의 부족함으로 '\n' 을 빼주기 위해 -1이 들어감
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                     # gray scale로 변환
        img = cv2.resize(img, (self.size_x, self.size_y), interpolation=cv2.INTER_NEAREST) # resize함수
        img = ~img                                                      # 색상 반전
        img = np.asarray(img, dtype=float)                              # numpy float형 배열로 변경
        img = img / 255                                                 # 0~1 사이 값을 가지도록 255로 나눗셈
        img = np.expand_dims(img, axis=0)                               # (1, H, W)의 형태가 되도록 차원 확장
        img_tensor = torch.FloatTensor(img)                             # FloatTensor타입으로 변경
        
        gt = cv2.imread(self.gt_path+self.file_names[index][:-1])       # gt 데이터 불러오기
        gt = cv2.resize(gt, (self.size_x, self.size_y), interpolation=cv2.INTER_NEAREST) # 입력과 같은 크기로 resize (주의 : 보간법은 NEAREST사용해야함)
        gt = gt[:,:,0]                                                  # 한 레이어만 있어도 됨
        gt_tensor = torch.LongTensor(gt)                                # pytorch cross-entropy loss를 사용하려면 정답이 long타입이 되어야함
        
        return img_tensor, gt_tensor
    def __len__(self):  # 크기를 반환하는 함수
        return len(self.file_names)
