"""
main6.py - 다양한 전략을 비교할 수 있는 사과게임 솔버

전략 목록:
1. Center-Out: 중앙에서 가까운 것부터 제거
2. Density-Seed: 짝꿍이 많은 곳(Seed)부터 확장
3. Pair-First: 2개짜리를 먼저 제거
4. Depth-1 Lookahead: 1수 앞 예측 (다음 기회 최대화)
5. Depth-2 Lookahead: 2수 앞 예측
6. Full Rollout: 끝까지 시뮬레이션 (닥터 스트레인지)
"""

import numpy as np
from numba import njit
import time

def read_matrix(path):
    return np.loadtxt(path, dtype=int)

# =========================================================
# Numba 최적화: Integral image + 후보 탐색 + move 적용
# =========================================================

@njit
def get_integral_images(board):
    H, W = board.shape
    P_sum = np.zeros((H + 1, W + 1), dtype=np.int32)
    P_cnt = np.zeros((H + 1, W + 1), dtype=np.int32)
    for r in range(H):
        for c in range(W):
            val = board[r, c]
            is_nonzero = 1 if val > 0 else 0
            P_sum[r + 1, c + 1] = P_sum[r, c + 1] + P_sum[r + 1, c] - P_sum[r, c] + val
            P_cnt[r + 1, c + 1] = P_cnt[r, c + 1] + P_cnt[r + 1, c] - P_cnt[r, c] + is_nonzero
    return P_sum, P_cnt

@njit
def get_rect_stat(r1, c1, r2, c2, P_sum, P_cnt):
    s = P_sum[r2 + 1, c2 + 1] - P_sum[r1, c2 + 1] - P_sum[r2 + 1, c1] + P_sum[r1, c1]
    cnt = P_cnt[r2 + 1, c2 + 1] - P_cnt[r1, c2 + 1] - P_cnt[r2 + 1, c1] + P_cnt[r1, c1]
    return s, cnt

@njit
def find_candidates_fast(board):
    """
    후보 반환: (r1, c1, r2, c2, cells, area)
    cells: 0이 아닌 셀 개수
    area: 직사각형 면적
    """
    H, W = board.shape
    P_sum, P_cnt = get_integral_images(board)
    cands = []
    for r1 in range(H):
        for c1 in range(W):
            for r2 in range(r1, H):
                for c2 in range(c1, W):
                    s, cells = get_rect_stat(r1, c1, r2, c2, P_sum, P_cnt)
                    if s == 10:
                        # 0이 아닌 셀이 2개 이상이어야 함
                        if cells >= 2:
                            area = (r2 - r1 + 1) * (c2 - c1 + 1)
                            cands.append((r1, c1, r2, c2, cells, area))
                    elif s > 10:
                        break
    return cands

@njit
def apply_move_fast(board, r1, c1, r2, c2):
    board[r1:r2 + 1, c1:c2 + 1] = 0

# =========================================================
# 전략 1: 중앙 집중형 (Center-Out)
# =========================================================

def solve_center_out(matrix, verbose=True):
    """
    맵의 중앙에서 가까운 것부터 제거.
    가운데를 먼저 비워서 공간 확보.
    """
    board = matrix.copy()
    H, W = board.shape
    center_r, center_c = H / 2, W / 2
    
    total_score = 0
    moves = []
    
    while True:
        cands = find_candidates_fast(board)
        if not cands:
            break
        
        # 각 후보의 중심점과 맵 중앙과의 거리 계산
        scored = []
        for (r1, c1, r2, c2, cells, area) in cands:
            drag_center_r = (r1 + r2) / 2
            drag_center_c = (c1 + c2) / 2
            dist = ((drag_center_r - center_r)**2 + (drag_center_c - center_c)**2)**0.5
            scored.append((dist, area, r1, c1, r2, c2, cells))
        
        # 1순위: 거리(가까운), 2순위: 면적(작은)
        scored.sort(key=lambda x: (x[0], x[1]))
        
        _, _, r1, c1, r2, c2, cells = scored[0]
        apply_move_fast(board, r1, c1, r2, c2)
        total_score += cells
        moves.append((r1, c1, r2, c2, cells))
    
    if verbose:
        print(f"[Center-Out] 총점: {total_score}, 횟수: {len(moves)}")
    
    return board, total_score, moves

# =========================================================
# 전략 2: 밀도 기반 시드 확장 (Density-Based Expansion)
# =========================================================

def solve_density_seed(matrix, verbose=True):
    """
    인접한 짝(합이 10)이 가장 많은 위치(Seed)를 찾고,
    그 주변부터 확장하며 제거.
    """
    board = matrix.copy()
    H, W = board.shape
    
    # 1. Seed 찾기: 상하좌우 인접 셀과 합이 10이 되는 경우가 많은 곳
    density_map = np.zeros((H, W), dtype=np.int32)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for r in range(H):
        for c in range(W):
            if board[r, c] == 0:
                continue
            val = board[r, c]
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if board[nr, nc] != 0 and (val + board[nr, nc] == 10):
                        density_map[r, c] += 1
    
    # 가장 밀도 높은 위치를 Seed로
    max_idx = np.argmax(density_map)
    seed_r, seed_c = max_idx // W, max_idx % W
    
    if verbose:
        print(f"📍 Seed 위치: ({seed_r}, {seed_c}), 밀도: {density_map[seed_r, seed_c]}")
    
    total_score = 0
    moves = []
    
    # 2. Seed 중심으로 가까운 것부터 제거
    while True:
        cands = find_candidates_fast(board)
        if not cands:
            break
        
        scored = []
        for (r1, c1, r2, c2, cells, area) in cands:
            drag_center_r = (r1 + r2) / 2
            drag_center_c = (c1 + c2) / 2
            dist = ((drag_center_r - seed_r)**2 + (drag_center_c - seed_c)**2)**0.5
            scored.append((dist, area, r1, c1, r2, c2, cells))
        
        scored.sort(key=lambda x: (x[0], x[1]))
        
        _, _, r1, c1, r2, c2, cells = scored[0]
        apply_move_fast(board, r1, c1, r2, c2)
        total_score += cells
        moves.append((r1, c1, r2, c2, cells))
    
    if verbose:
        print(f"[Density-Seed] 총점: {total_score}, 횟수: {len(moves)}")
    
    return board, total_score, moves

# =========================================================
# 전략 3: 2개짜리 짝 우선 (Pair-First)
# =========================================================

def solve_pair_first(matrix, verbose=True):
    """
    사과 2개짜리(짝)를 먼저 제거.
    작은 조합을 먼저 처리해서 더 큰 조합 가능성을 열어둠.
    """
    board = matrix.copy()
    H, W = board.shape
    center_r, center_c = H / 2, W / 2
    
    total_score = 0
    moves = []
    
    while True:
        cands = find_candidates_fast(board)
        if not cands:
            break
        
        scored = []
        for (r1, c1, r2, c2, cells, area) in cands:
            drag_center_r = (r1 + r2) / 2
            drag_center_c = (c1 + c2) / 2
            dist = ((drag_center_r - center_r)**2 + (drag_center_c - center_c)**2)**0.5
            # 1순위: cells(작은), 2순위: dist(가까운), 3순위: area(작은)
            scored.append((cells, dist, area, r1, c1, r2, c2))
        
        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        
        cells, _, _, r1, c1, r2, c2 = scored[0]
        apply_move_fast(board, r1, c1, r2, c2)
        total_score += cells
        moves.append((r1, c1, r2, c2, cells))
    
    if verbose:
        print(f"[Pair-First] 총점: {total_score}, 횟수: {len(moves)}")
    
    return board, total_score, moves

# =========================================================
# 전략 4: 1수 앞 예측 (Depth-1 Lookahead)
# =========================================================

def solve_depth1_lookahead(matrix, verbose=True):
    """
    각 후보를 선택했을 때, 다음에 가능한 후보 수가 많은 것 선택.
    기회를 최대한 남기는 전략.
    """
    board = matrix.copy()
    
    total_score = 0
    moves = []
    
    while True:
        cands = find_candidates_fast(board)
        if not cands:
            break
        
        # 후보가 하나면 바로 실행
        if len(cands) == 1:
            r1, c1, r2, c2, cells, area = cands[0]
            apply_move_fast(board, r1, c1, r2, c2)
            total_score += cells
            moves.append((r1, c1, r2, c2, cells))
            continue
        
        best_move = None
        max_opportunities = -1
        
        for (r1, c1, r2, c2, cells, area) in cands:
            # 가상으로 적용
            sim = board.copy()
            apply_move_fast(sim, r1, c1, r2, c2)
            
            # 다음 단계 후보 수
            next_cands = find_candidates_fast(sim)
            opp = len(next_cands)
            
            # 더 많은 기회를 남기는 것 선택
            if opp > max_opportunities:
                max_opportunities = opp
                best_move = (r1, c1, r2, c2, cells, area)
            elif opp == max_opportunities and best_move is not None:
                # 동점: cells 작은 것 > area 작은 것 우선
                if cells < best_move[4] or (cells == best_move[4] and area < best_move[5]):
                    best_move = (r1, c1, r2, c2, cells, area)
        
        if best_move is None:
            break
        
        r1, c1, r2, c2, cells, area = best_move
        apply_move_fast(board, r1, c1, r2, c2)
        total_score += cells
        moves.append((r1, c1, r2, c2, cells))
    
    if verbose:
        print(f"[Depth-1] 총점: {total_score}, 횟수: {len(moves)}")
    
    return board, total_score, moves

# =========================================================
# 전략 5: 2수 앞 예측 (Depth-2 Lookahead)
# =========================================================

def evaluate_future_depth2(board, depth, max_depth=1, top_k=5):
    """재귀적으로 미래 예측 (가지치기 적용)"""
    cands = find_candidates_fast(board)
    
    if not cands:
        return 0
    
    # 목표 깊이에 도달하면 남은 후보 수 반환
    if depth >= max_depth:
        return len(cands)
    
    # 상위 k개만 평가 (속도 최적화)
    scored = [(c[4], c[5], c) for c in cands]  # (cells, area, cand)
    scored.sort(key=lambda x: (x[0], x[1]))
    top_cands = [s[2] for s in scored[:top_k]]
    
    max_score = 0
    for (r1, c1, r2, c2, cells, area) in top_cands:
        sim = board.copy()
        apply_move_fast(sim, r1, c1, r2, c2)
        sub_score = 1 + evaluate_future_depth2(sim, depth + 1, max_depth, top_k)
        if sub_score > max_score:
            max_score = sub_score
    
    return max_score

def solve_depth2_lookahead(matrix, verbose=True):
    """
    2수 앞까지 예측하여 최적의 수 선택.
    각 후보에 대해 재귀적으로 미래 점수 평가.
    """
    board = matrix.copy()
    
    total_score = 0
    moves = []
    
    while True:
        cands = find_candidates_fast(board)
        if not cands:
            break
        
        if len(cands) == 1:
            r1, c1, r2, c2, cells, area = cands[0]
            apply_move_fast(board, r1, c1, r2, c2)
            total_score += cells
            moves.append((r1, c1, r2, c2, cells))
            continue
        
        best_move = None
        max_future = -1
        
        for (r1, c1, r2, c2, cells, area) in cands:
            sim = board.copy()
            apply_move_fast(sim, r1, c1, r2, c2)
            
            future = evaluate_future_depth2(sim, depth=0, max_depth=1, top_k=5)
            
            if future > max_future:
                max_future = future
                best_move = (r1, c1, r2, c2, cells, area)
            elif future == max_future and best_move is not None:
                if cells < best_move[4] or (cells == best_move[4] and area < best_move[5]):
                    best_move = (r1, c1, r2, c2, cells, area)
        
        if best_move is None:
            break
        
        r1, c1, r2, c2, cells, area = best_move
        apply_move_fast(board, r1, c1, r2, c2)
        total_score += cells
        moves.append((r1, c1, r2, c2, cells))
    
    if verbose:
        print(f"[Depth-2] 총점: {total_score}, 횟수: {len(moves)}")
    
    return board, total_score, moves

# =========================================================
# 전략 6: Full Rollout (닥터 스트레인지)
# =========================================================

def greedy_rollout(board):
    """
    탐욕적으로 끝까지 플레이하여 최종 점수 반환.
    (2개짜리 우선 > 면적 작은 것 우선)
    """
    sim = board.copy()
    score = 0
    
    while True:
        cands = find_candidates_fast(sim)
        if not cands:
            break
        
        # 2개짜리 > 면적 작은 것 우선
        scored = [(c[4], c[5], c) for c in cands]
        scored.sort(key=lambda x: (x[0], x[1]))
        
        _, _, (r1, c1, r2, c2, cells, area) = scored[0]
        apply_move_fast(sim, r1, c1, r2, c2)
        score += 1  # 횟수 카운트
    
    return score

def solve_full_rollout(matrix, top_k=30, verbose=True):
    """
    각 후보에 대해 끝까지 시뮬레이션하여
    최종 점수가 가장 높은 것 선택.
    """
    board = matrix.copy()
    
    total_score = 0
    moves = []
    
    while True:
        cands = find_candidates_fast(board)
        if not cands:
            break
        
        # 후보가 너무 많으면 상위 top_k개만 평가
        scored_cands = [(c[4], c) for c in cands]  # (cells, cand)
        scored_cands.sort(key=lambda x: -x[0])  # cells 내림차순
        eval_cands = [c for _, c in scored_cands[:top_k]]
        
        best_move = None
        best_total = -1
        
        for (r1, c1, r2, c2, cells, area) in eval_cands:
            sim = board.copy()
            apply_move_fast(sim, r1, c1, r2, c2)
            
            # 끝까지 시뮬레이션
            future = greedy_rollout(sim)
            total = cells + future
            
            if total > best_total:
                best_total = total
                best_move = (r1, c1, r2, c2, cells, area)
            elif total == best_total and best_move is not None:
                # 동점: cells 큰 것 > area 작은 것 우선
                if cells > best_move[4] or (cells == best_move[4] and area < best_move[5]):
                    best_move = (r1, c1, r2, c2, cells, area)
        
        if best_move is None:
            break
        
        r1, c1, r2, c2, cells, area = best_move
        apply_move_fast(board, r1, c1, r2, c2)
        total_score += cells
        moves.append((r1, c1, r2, c2, cells))
    
    if verbose:
        print(f"[Full-Rollout] 총점: {total_score}, 횟수: {len(moves)}")
    
    return board, total_score, moves

# =========================================================
# 모든 전략 비교
# =========================================================

def compare_all_strategies(matrix):
    """모든 전략을 실행하고 결과 비교"""
    print("=" * 60)
    print("🍎 사과게임 전략 비교")
    print("=" * 60)
    
    strategies = [
        ("Center-Out", solve_center_out),
        ("Density-Seed", solve_density_seed),
        ("Pair-First", solve_pair_first),
        ("Depth-1 Lookahead", solve_depth1_lookahead),
        ("Depth-2 Lookahead", solve_depth2_lookahead),
        ("Full-Rollout", solve_full_rollout),
    ]
    
    results = []
    
    for name, solver in strategies:
        print(f"\n▶ {name} 전략 실행 중...")
        t0 = time.time()
        
        if name == "Full-Rollout":
            _, score, moves = solver(matrix.copy(), top_k=30, verbose=False)
        else:
            _, score, moves = solver(matrix.copy(), verbose=False)
        
        elapsed = time.time() - t0
        results.append((name, score, len(moves), elapsed))
        print(f"  → 점수: {score}, 횟수: {len(moves)}, 시간: {elapsed:.2f}s")
    
    # 결과 정렬 (점수 내림차순)
    results.sort(key=lambda x: -x[1])
    
    print("\n" + "=" * 60)
    print("📊 최종 결과 (점수 순)")
    print("=" * 60)
    print(f"{'순위':<4} {'전략':<20} {'점수':<8} {'횟수':<8} {'시간':<10}")
    print("-" * 60)
    
    for i, (name, score, moves_count, elapsed) in enumerate(results, 1):
        print(f"{i:<4} {name:<20} {score:<8} {moves_count:<8} {elapsed:.2f}s")
    
    print("-" * 60)
    best_name, best_score, _, _ = results[0]
    print(f"🏆 최고 전략: {best_name} (점수: {best_score})")
    
    return results

# =========================================================
# 실행부
# =========================================================

if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) < 2:
        print("사용법: python main.py <보드파일명>")
        print("예시: python main.py board1")
        sys.exit(1)
    
    board_name = sys.argv[1]
    
    # .txt 확장자가 없으면 추가
    if not board_name.endswith(".txt"):
        board_name += ".txt"
    
    # board_mat/ 경로가 없으면 추가
    if not os.path.dirname(board_name):
        board_path = os.path.join("board_mat", board_name)
    else:
        board_path = board_name
    
    if not os.path.exists(board_path):
        print(f"오류: 파일을 찾을 수 없습니다: {board_path}")
        sys.exit(1)
    
    mat = read_matrix(board_path)
    print(f"보드 파일: {board_path}")
    print(f"보드 크기: {mat.shape}")
    print(f"보드:\n{mat}\n")
    
    # 모든 전략 비교
    compare_all_strategies(mat)
