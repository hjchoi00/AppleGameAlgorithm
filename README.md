# 사과 게임 이미지 → 숫자 행렬 변환기

사과 게임 스크린샷을 자동으로 분석하여 숫자 행렬로 변환하는 프로그램입니다.

## 설치 방법

### 1. 필요한 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 2. OCR 엔진 선택

#### 방법 A: EasyOCR (권장)

- 설치 쉬움, 정확도 높음
- `apple_game_ocr.py` 사용
- 추가 설정 불필요

#### 방법 B: Tesseract OCR

- 가볍고 빠름
- `simple_ocr_example.py` 사용
- Tesseract 별도 설치 필요: https://github.com/UB-Mannheim/tesseract/wiki

```bash
pip install pytesseract
```

## 사용 방법

### 1. 이미지 준비

사과 게임 스크린샷을 프로젝트 폴더에 저장하세요.

### 2. 코드 실행

**EasyOCR 버전:**

```python
from apple_game_ocr import AppleGameOCR

ocr = AppleGameOCR()
matrix = ocr.image_to_matrix("your_screenshot.png")
ocr.print_matrix(matrix)
```

**Tesseract 버전:**

```python
from simple_ocr_example import image_to_matrix

matrix = image_to_matrix("your_screenshot.png")
print(matrix)
```

## 작동 원리

1. **이미지 로드**: OpenCV로 이미지 읽기
2. **색상 필터링**: HSV 색상 공간에서 빨간색 영역 추출
3. **객체 감지**: 윤곽선 검출로 각 사과 위치 파악
4. **격자 정렬**: Y좌표로 행 그룹화, X좌표로 열 정렬
5. **숫자 인식**: 각 사과 중심부를 OCR로 숫자 추출
6. **행렬 생성**: 2D 리스트로 결과 반환

## 매개변수 조정

### 사과 감지 정확도 조정

`apple_game_ocr.py`의 다음 값들을 수정:

```python
# 최소 사과 크기 (픽셀)
if area > 200:  # 이 값을 조정

# 행 정렬 허용 오차
y_tolerance = 15  # 이 값을 조정

# 빨간색 범위 (HSV)
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
```

## 결과 활용

```python
# NumPy 배열로 변환
import numpy as np
np_matrix = np.array(matrix)

# 특정 위치 값 접근
value = matrix[row][col]

# 파일로 저장
ocr.save_matrix(matrix, "output.txt")
```

## 문제 해결

### 숫자 인식이 정확하지 않을 때

1. 이미지 해상도를 높여서 캡처
2. `padding` 값 조정 (ROI 크기)
3. 임계값(`threshold`) 조정

### 사과가 제대로 감지되지 않을 때

1. HSV 색상 범위 조정
2. 최소 크기(`area > 200`) 조정
3. `y_tolerance` 값 조정

## 라이선스

MIT License
