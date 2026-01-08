"""
사과 게임 이미지를 YOLO 모델로 숫자 행렬로 변환하는 프로그램
(OneShotVision 기반 - apple_ocr.py와 동일한 인터페이스)
"""
import cv2
import numpy as np
import json
import sys
import os
from pathlib import Path
from ultralytics import YOLO

# === [설정] ===
CONFIG_FILE = "grid_config.json"
MODEL_PATH = "best.pt"
ROWS = 10
COLS = 17


class AppleGameYOLO:
    """YOLO 모델을 사용한 사과 게임 숫자 인식기"""
    
    def __init__(self):
        # 현재 스크립트 위치 기준으로 파일 경로 설정
        self.script_dir = Path(__file__).parent
        self.config_path = self.script_dir / CONFIG_FILE
        self.model_path = self.script_dir / MODEL_PATH
        
        # 설정 파일 로드
        if not self.config_path.exists():
            raise Exception(f"❌ {CONFIG_FILE}이 없습니다!")
        
        with open(self.config_path, 'r') as f:
            self.cfg = json.load(f)
        
        # 모델 로드
        if not self.model_path.exists():
            raise Exception(f"❌ 모델({MODEL_PATH})이 없습니다!")
        
        print("🧠 YOLO 모델 로딩 중...")
        self.model = YOLO(str(self.model_path))
        print("✅ 모델 로딩 완료!")
        
        # 이미지 저장 변수
        self.image = None
        self.original = None
    
    def load_image(self, image_path):
        """이미지 로드 (한글 경로 지원)"""
        try:
            img_array = np.fromfile(image_path, dtype=np.uint8)
            self.image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if self.image is not None:
                self.original = self.image.copy()
                return True
            return False
        except Exception as e:
            print(f"이미지 로드 오류: {e}")
            return False
    
    def detect_game_board(self, img_bgr):
        """게임판 영역 찾기 (여러 방법 시도)"""
        h, w = img_bgr.shape[:2]
        
        # 방법 1: 초록색 배경 찾기
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([30, 40, 40]), np.array([90, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            max_cnt = max(contours, key=cv2.contourArea)
            gx, gy, gw, gh = cv2.boundingRect(max_cnt)
            # 게임판이 충분히 크면 사용 (이미지의 30% 이상)
            if gw * gh > w * h * 0.3:
                print("[게임판 감지] 초록색 배경으로 감지")
                return (gx, gy, gw, gh)
        
        # 방법 2: 빨간 사과 영역으로 추정
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 모든 빨간 영역을 포함하는 bounding box
            all_points = np.vstack(contours)
            gx, gy, gw, gh = cv2.boundingRect(all_points)
            # 약간의 패딩 추가
            padding = 20
            gx = max(0, gx - padding)
            gy = max(0, gy - padding)
            gw = min(w - gx, gw + padding * 2)
            gh = min(h - gy, gh + padding * 2)
            print("[게임판 감지] 빨간 사과 영역으로 추정")
            return (gx, gy, gw, gh)
        
        # 방법 3: 이미지 전체 사용
        print("[게임판 감지] 이미지 전체 사용")
        return (0, 0, w, h)
    
    def image_to_matrix(self, image_path):
        """이미지를 숫자 행렬로 변환하는 메인 함수"""
        print("이미지 로드 중...")
        if not self.load_image(image_path):
            print("이미지를 로드할 수 없습니다.")
            return None
        
        img_bgr = self.image
        
        # 게임판 영역 찾기
        print("게임판 영역 감지 중...")
        board_rect = self.detect_game_board(img_bgr)
        
        if board_rect is None:
            print("❌ 게임판을 찾을 수 없습니다.")
            return None
        
        gx, gy, gw, gh = board_rect
        print(f"게임판 영역: x={gx}, y={gy}, w={gw}, h={gh}")
        
        # 게임판 이미지 추출 (패딩 없이)
        board_img = img_bgr[gy:gy+gh, gx:gx+gw]
        cur_h, cur_w = board_img.shape[:2]
        
        # YOLO 모델 추론
        print("YOLO 모델로 숫자 인식 중...")
        results = self.model(board_img, conf=0.5, iou=0.5, verbose=False)
        
        # 격자 초기화 (0으로 채움)
        grid = [[0] * COLS for _ in range(ROWS)]
        
        # 감지된 박스가 없으면 종료
        if not results[0].boxes or len(results[0].boxes) == 0:
            print("[디버그] YOLO 감지 박스 수: 0")
            print("⚠️ 숫자가 감지되지 않았습니다.")
            return grid
        
        # 모든 감지된 박스 수집
        detections = []
        for box in results[0].boxes:
            bx, by, bw, bh = box.xywh[0].cpu().numpy()
            cls = int(box.cls[0]) + 1  # 클래스 번호 (1~9)
            detections.append({'x': bx, 'y': by, 'w': bw, 'h': bh, 'cls': cls})
        
        print(f"[디버그] YOLO 감지 박스 수: {len(detections)}")
        
        # 감지된 사과들의 위치를 기반으로 격자 계산
        # X, Y 좌표 수집
        xs = sorted(set(d['x'] for d in detections))
        ys = sorted(set(d['y'] for d in detections))
        
        # 평균 셀 크기 추정
        avg_w = np.mean([d['w'] for d in detections])
        avg_h = np.mean([d['h'] for d in detections])
        
        print(f"[디버그] board_img 크기: {cur_w} x {cur_h}")
        print(f"[디버그] 평균 사과 크기: {avg_w:.1f} x {avg_h:.1f}")
        
        # 격자 시작점과 셀 크기 계산
        # 방법: 이미지 크기를 격자 수로 나눔
        cell_w = cur_w / COLS
        cell_h = cur_h / ROWS
        
        print(f"[디버그] 계산된 셀 크기: {cell_w:.1f} x {cell_h:.1f}")
        
        detected_count = 0
        out_of_range_count = 0
        
        for i, d in enumerate(detections):
            # 격자 인덱스 계산 (이미지 좌표를 격자 인덱스로)
            col_idx = int(d['x'] / cell_w)
            row_idx = int(d['y'] / cell_h)
            
            # 처음 5개만 디버그 출력
            if i < 5:
                print(f"[디버그] box[{i}]: pos=({d['x']:.1f}, {d['y']:.1f}), cls={d['cls']}, idx=({row_idx}, {col_idx})")
            
            if 0 <= row_idx < ROWS and 0 <= col_idx < COLS:
                grid[row_idx][col_idx] = d['cls']
                detected_count += 1
            else:
                out_of_range_count += 1
        
        print(f"\n✅ 격자에 배치된 숫자: {detected_count}")
        if out_of_range_count > 0:
            print(f"⚠️ 범위 밖 감지: {out_of_range_count}")
        return grid
    
    def print_matrix(self, matrix):
        """행렬을 보기 좋게 출력"""
        if not matrix:
            print("행렬이 비어있습니다.")
            return
        
        print("\n=== 변환된 숫자 행렬 ===")
        for row in matrix:
            print(' '.join(f"{num:>2}" for num in row))
        print()
    
    def save_matrix(self, matrix, output_path):
        """행렬을 파일로 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for row in matrix:
                f.write(' '.join(str(num) for num in row) + '\n')
        print(f"행렬이 {output_path}에 저장되었습니다.")


def main():
    import re
    
    # 현재 스크립트 폴더 기준 경로 설정
    script_dir = Path(__file__).parent
    board_img_dir = script_dir / "board_img"
    board_mat_dir = script_dir / "board_mat"
    
    # board_mat 폴더가 없으면 생성
    board_mat_dir.mkdir(exist_ok=True)
    
    # 명령줄 인자 처리
    if len(sys.argv) < 2:
        print("=" * 50)
        print("🍎 YOLO 기반 사과 게임 숫자 추출기")
        print("=" * 50)
        print("\n사용법: python apple_yolo.py <이미지파일명>")
        print("예시: python apple_yolo.py image1.png")
        print("      → board_img/image1.png 에서 읽어서")
        print("      → board_mat/board1.txt 로 저장")
        sys.exit(1)
    
    # 이미지 파일명
    image_name = sys.argv[1]
    image_path = board_img_dir / image_name
    
    # 파일 존재 확인
    if not image_path.exists():
        print(f"❌ 오류: '{image_path}' 파일을 찾을 수 없습니다.")
        print(f"\nboard_img 폴더 내용:")
        if board_img_dir.exists():
            files = [f for f in os.listdir(board_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                for f in files:
                    print(f"  - {f}")
            else:
                print("  (이미지 파일 없음)")
        else:
            print(f"  '{board_img_dir}' 폴더가 존재하지 않습니다.")
        sys.exit(1)
    
    print(f"🎯 이미지 처리 시작: {image_path}")
    print("=" * 50)
    
    # YOLO 객체 생성
    yolo = AppleGameYOLO()
    
    # 이미지를 행렬로 변환
    matrix = yolo.image_to_matrix(str(image_path))
    
    if matrix:
        print("\n" + "=" * 50)
        # 결과 출력
        yolo.print_matrix(matrix)
        
        # 출력 파일명 생성: image1.png -> board1.txt
        base_name = os.path.splitext(image_name)[0]  # 확장자 제거
        # 숫자 추출 (예: image1 -> 1, capture123 -> 123)
        numbers = re.findall(r'\d+', base_name)
        if numbers:
            output_name = f"board{numbers[-1]}.txt"  # 마지막 숫자 사용
        else:
            output_name = f"board_{base_name}.txt"  # 숫자가 없으면 원본 이름 사용
        
        output_path = board_mat_dir / output_name
        yolo.save_matrix(matrix, str(output_path))
        
        # NumPy 배열로 변환 (추가 처리용)
        np_matrix = np.array(matrix)
        print(f"✅ 행렬 크기: {np_matrix.shape}")
        print(f"✅ 결과 저장: {output_path}")
    else:
        print("❌ 이미지를 행렬로 변환하는데 실패했습니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()
