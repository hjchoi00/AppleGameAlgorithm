# 🍎 사과게임 최적 알고리즘 솔버

사과게임에서 **최고 점수를 달성하는 최적의 알고리즘**을 찾는 프로젝트입니다.

## 개요

사과게임은 숫자가 적힌 사과들 중 합이 10이 되는 직사각형 영역을 선택하여 제거하는 게임입니다. 이 프로젝트는 다양한 전략 알고리즘을 구현하고 비교하여 가장 높은 점수를 얻는 방법을 탐구합니다.

## 설치 방법

```bash
pip install -r requirements.txt
```

## 프로젝트 구조

```
├── main.py          # 전략 알고리즘 구현 및 비교
├── apple_ocr.py     # 게임 스크린샷 → 숫자 행렬 변환 (OCR)
├── board_mat/       # 테스트용 보드 행렬 파일
└── board_img/       # 게임 스크린샷 이미지
```

## 구현된 전략

| 전략 | 설명 |
|------|------|
| **Center-Out** | 맵 중앙에서 가까운 것부터 제거 |
| **Density-Seed** | 짝꿍(합 10)이 많은 위치를 시드로 확장 |
| **Pair-First** | 2개짜리 조합을 먼저 제거하여 더 큰 조합 기회 확보 |
| **Depth-1 Lookahead** | 1수 앞을 예측하여 다음 기회가 많은 수 선택 |
| **Depth-2 Lookahead** | 2수 앞까지 예측하여 최적의 수 선택 |
| **Full Rollout** | 각 수에 대해 끝까지 시뮬레이션하여 최종 점수가 높은 수 선택 |

## 사용 방법

### 1. 보드 행렬 파일로 실행

```python
from main import compare_all_strategies, read_matrix

# 행렬 파일 로드
mat = read_matrix("board_mat/board1.txt")

# 모든 전략 비교
compare_all_strategies(mat)
```

### 2. 게임 스크린샷에서 자동 변환

```python
from apple_ocr import AppleGameOCR

ocr = AppleGameOCR()
matrix = ocr.image_to_matrix("screenshot.png")
```

### 3. 개별 전략 실행

```python
from main import solve_full_rollout, read_matrix

mat = read_matrix("board_mat/board1.txt")
final_board, score, moves = solve_full_rollout(mat, top_k=50)

print(f"최종 점수: {score}")
print(f"이동 횟수: {len(moves)}")
```

## 실행 예시

```bash
python main.py
```

출력:
```
🍎 사과게임 전략 비교
============================================================

▶ Center-Out 전략 실행 중...
  → 점수: 142, 횟수: 45, 시간: 0.12s
▶ Density-Seed 전략 실행 중...
  → 점수: 145, 횟수: 47, 시간: 0.11s
...

📊 최종 결과 (점수 순)
============================================================
순위 전략                 점수     횟수     시간
------------------------------------------------------------
1    Full-Rollout         156      52       2.34s
2    Depth-2 Lookahead    153      50       0.89s
...
🏆 최고 전략: Full-Rollout (점수: 156)
```

## 알고리즘 원리

### 핵심 최적화
- **Integral Image**: O(1) 시간에 직사각형 영역의 합 계산
- **Numba JIT**: 후보 탐색 및 이동 적용 함수 최적화
- **가지치기**: Lookahead 전략에서 상위 k개 후보만 평가

### 후보 조건
- 직사각형 내 숫자의 합 = 10
- 0이 아닌 셀이 2개 이상

## 라이선스

MIT License
