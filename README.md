# serving 

### 요구사항
- onnx 모델을 관리한다
- 다수의 onnx 모델을 선택하여 prediction을 실행한다
- 전처리/ 후처리 관리 스키마를 정의한다


---

#### 인공지능 모델

#### env
- base python : 3.9

#### command
- preprocess
```
python main.py --data_dir ./_source/data --train_data train.csv --preprocess
```
- scale
```
python main.py --data_dir ./_source/data --train_data train.csv --scale
```
- train
```
python main.py --data_dir ./_source/data --train_data train.csv --train --epochs 100
```
- test
```
python main.py --data_dir ./_source/data --test_data test.csv --test
```
- predict
```
python main.py --data_dir ./_source/data --predict_data predict.csv --predict
```
- optimize
```
python main.py --data_dir ./_source/data --train_data train.csv --test_data test.csv --optimize
```