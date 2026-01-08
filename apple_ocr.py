"""
사과 게임 이미지를 숫자 행렬로 변환하는 프로그램
"""
import cv2
import numpy as np
from PIL import Image
import easyocr
from collections import defaultdict
import sys
import os

class AppleGameOCR:
    def __init__(self):
        # EasyOCR 리더 초기화 (한글과 영어 숫자 인식)
        self.reader = easyocr.Reader(['en'], gpu=True)
        
    def load_image(self, image_path):
        """이미지 로드 (한글 경로 지원)"""
        # 한글 경로 지원을 위해 numpy.fromfile + cv2.imdecode 사용
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
    
    def detect_apples(self):
        """빨간 사과 영역 감지"""
        # BGR을 HSV로 변환
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # 빨간색 범위 정의 (HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # 빨간색 마스크 생성
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 사과 정보 저장 (중심점 좌표와 영역)
        apples = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # 일정 크기 이상의 윤곽선만 사과로 인식
            if area > 200:  # 최소 크기 조정 가능
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    x, y, w, h = cv2.boundingRect(contour)
                    apples.append({
                        'center': (cx, cy),
                        'bbox': (x, y, w, h),
                        'contour': contour
                    })
        
        return apples
    
    def organize_grid(self, apples):
        """사과들을 격자 구조로 정렬"""
        if not apples:
            return []
        
        # Y 좌표로 행 그룹화 (허용 오차 범위 내)
        rows = defaultdict(list)
        y_tolerance = 15  # Y 좌표 허용 오차
        
        for apple in apples:
            cy = apple['center'][1]
            # 기존 행과 비교
            found_row = False
            for row_y in list(rows.keys()):
                if abs(cy - row_y) < y_tolerance:
                    rows[row_y].append(apple)
                    found_row = True
                    break
            if not found_row:
                rows[cy] = [apple]
        
        # 각 행을 X 좌표로 정렬
        grid = []
        for row_y in sorted(rows.keys()):
            row = sorted(rows[row_y], key=lambda a: a['center'][0])
            grid.append(row)
        
        return grid
    
    def extract_number(self, apple_bbox, debug_idx=None):
        """사과 영역에서 숫자 추출 (개선된 버전)"""
        x, y, w, h = apple_bbox
        
        # 사과 전체 영역 추출
        padding = int(w * 0.15)
        roi = self.original[
            max(0, y + padding):min(self.original.shape[0], y + h - padding),
            max(0, x + padding):min(self.original.shape[1], x + w - padding)
        ]
        
        if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
            return '?'
        
        # 리사이즈로 크기 확대 (OCR 정확도 향상)
        scale = 4
        roi_resized = cv2.resize(roi, (roi.shape[1] * scale, roi.shape[0] * scale), 
                                interpolation=cv2.INTER_CUBIC)
        
        # BGR에서 각 채널 분리
        b, g, r = cv2.split(roi_resized)
        
        # 전처리: 그레이스케일 변환
        gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        
        # 여러 방법으로 전처리 시도
        processed_images = []
        
        # 방법 1: 밝은 영역 추출 (흰색 숫자)
        _, binary1 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        processed_images.append(('binary_200', binary1))
        
        # 방법 2: 더 낮은 임계값
        _, binary2 = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        processed_images.append(('binary_170', binary2))
        
        # 방법 3: B 채널 기반 (빨간 배경에서 흰색 추출에 유리)
        _, binary3 = cv2.threshold(b, 180, 255, cv2.THRESH_BINARY)
        processed_images.append(('b_channel', binary3))
        
        # 방법 4: G 채널 기반
        _, binary4 = cv2.threshold(g, 180, 255, cv2.THRESH_BINARY)
        processed_images.append(('g_channel', binary4))
        
        # 방법 5: 색상 차이 기반 (빨간색과 흰색의 차이)
        red_white_diff = cv2.absdiff(r, cv2.min(b, g))
        _, binary5 = cv2.threshold(red_white_diff, 100, 255, cv2.THRESH_BINARY_INV)
        processed_images.append(('color_diff', binary5))
        
        # 방법 6: 적응형 이진화
        binary6 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 15, 2)
        processed_images.append(('adaptive', binary6))
        
        # 방법 7: Otsu's 이진화
        _, binary7 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(('otsu', binary7))
        
        # 디버그: 이미지 저장 (비활성화)
        # if debug_idx is not None and debug_idx < 10:
        #     cv2.imwrite(f"debug_roi_{debug_idx}.png", roi_resized)
        #     for name, img in processed_images:
        #         cv2.imwrite(f"debug_{name}_{debug_idx}.png", img)
        
        # 각 전처리 방법으로 OCR 시도
        best_result = None
        best_confidence = 0
        best_method = None
        
        for method_name, proc_img in processed_images:
            try:
                results = self.reader.readtext(proc_img, allowlist='0123456789',
                                             detail=1, paragraph=False,
                                             width_ths=0.5, height_ths=0.5)
                for bbox, text, confidence in results:
                    # 숫자만 추출
                    number = ''.join(filter(str.isdigit, text))
                    if number and len(number) == 1 and confidence > best_confidence:
                        best_confidence = confidence
                        best_result = number
                        best_method = method_name
            except Exception as e:
                continue
        
        # 디버그 출력 (비활성화)
        # if debug_idx is not None and best_result:
        #     print(f"  → [{debug_idx}] 인식: {best_result} (신뢰도: {best_confidence:.2f}, 방법: {best_method})")
        
        # 최소 신뢰도 체크
        if best_result and best_confidence > 0.05:
            return best_result
        
        return '?'
    
    def image_to_matrix(self, image_path):
        """이미지를 숫자 행렬로 변환하는 메인 함수"""
        print("이미지 로드 중...")
        if not self.load_image(image_path):
            print("이미지를 로드할 수 없습니다.")
            return None
        
        print("사과 영역 감지 중...")
        apples = self.detect_apples()
        print(f"감지된 사과 수: {len(apples)}")
        
        print("격자 구조 정렬 중...")
        grid = self.organize_grid(apples)
        print(f"행 수: {len(grid)}")
        
        print("숫자 인식 중...")
        matrix = []
        total_apples = sum(len(row) for row in grid)
        current = 0
        
        for row_idx, row in enumerate(grid):
            row_numbers = []
            for col_idx, apple in enumerate(row):
                current += 1
                # 디버그 모드 비활성화
                number = self.extract_number(apple['bbox'], debug_idx=None)
                row_numbers.append(number)
                
                # 진행률 표시
                if current % 20 == 0 or current == total_apples:
                    print(f"진행: {current}/{total_apples} ({100*current//total_apples}%)")
            matrix.append(row_numbers)
        
        return matrix
    
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
    from pathlib import Path
    
    # 현재 스크립트 폴더 기준 경로 설정
    script_dir = Path(__file__).parent
    board_img_dir = script_dir / "board_img"
    board_mat_dir = script_dir / "board_mat"
    
    # board_mat 폴더가 없으면 생성
    board_mat_dir.mkdir(exist_ok=True)
    
    # 명령줄 인자 처리
    if len(sys.argv) < 2:
        print("=" * 50)
        print("🍎 OCR 기반 사과 게임 숫자 추출기")
        print("=" * 50)
        print("\n사용법: python apple_ocr.py <이미지파일명>")
        print("예시: python apple_ocr.py image1.png")
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
    
    # OCR 객체 생성
    ocr = AppleGameOCR()
    
    # 이미지를 행렬로 변환
    matrix = ocr.image_to_matrix(str(image_path))
    
    if matrix:
        print("\n" + "=" * 50)
        # 결과 출력
        ocr.print_matrix(matrix)
        
        # 출력 파일명 생성: image1.png -> board1.txt
        base_name = os.path.splitext(image_name)[0]  # 확장자 제거
        # 숫자 추출 (예: image1 -> 1, capture123 -> 123)
        numbers = re.findall(r'\d+', base_name)
        if numbers:
            output_name = f"board{numbers[-1]}.txt"  # 마지막 숫자 사용
        else:
            output_name = f"board_{base_name}.txt"  # 숫자가 없으면 원본 이름 사용
        
        output_path = board_mat_dir / output_name
        ocr.save_matrix(matrix, str(output_path))
        
        # NumPy 배열로 변환 (추가 처리용)
        # ?를 0으로 변환
        matrix_clean = [[0 if x == '?' else int(x) for x in row] for row in matrix]
        np_matrix = np.array(matrix_clean)
        print(f"✅ 행렬 크기: {np_matrix.shape}")
        print(f"✅ 결과 저장: {output_path}")
    else:
        print("❌ 이미지를 행렬로 변환하는데 실패했습니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()
