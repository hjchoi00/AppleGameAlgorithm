"""
benchmark.py - 모든 보드에 대해 전략별 평균 성능 측정
"""

import os
import time
import numpy as np
from main import (
    read_matrix,
    solve_center_out,
    solve_density_seed,
    solve_pair_first,
    solve_depth1_lookahead,
    solve_depth2_lookahead,
    solve_full_rollout,
    solve_my_depth2
)


def benchmark_all():
    """board_mat 폴더의 모든 보드에 대해 전략별 평균 성능 측정"""
    
    board_dir = "board_mat"
    board_files = sorted([f for f in os.listdir(board_dir) if f.endswith(".txt")])
    
    if not board_files:
        print("보드 파일이 없습니다.")
        return
    
    print("=" * 70)
    print("🍎 사과게임 전략 벤치마크")
    print("=" * 70)
    print(f"테스트 보드: {len(board_files)}개")
    print(f"보드 목록: {', '.join(board_files)}")
    print()
    
    strategies = [
        ("Center-Out", solve_center_out, {}),
        ("Density-Seed", solve_density_seed, {}),
        ("Pair-First", solve_pair_first, {}),
        ("Depth-1 Lookahead", solve_depth1_lookahead, {}),
        ("Depth-2 Lookahead", solve_depth2_lookahead, {}),
        ("Full-Rollout", solve_full_rollout, {"top_k": 30}),
        ("My Depth-2", solve_my_depth2, {}),
    ]
    
    # 결과 저장: {전략명: {"scores": [], "moves": [], "times": []}}
    results = {name: {"scores": [], "moves": [], "times": []} for name, _, _ in strategies}
    
    # 각 보드에 대해 모든 전략 실행
    for board_file in board_files:
        board_path = os.path.join(board_dir, board_file)
        mat = read_matrix(board_path)
        print(f"▶ {board_file} 테스트 중... (크기: {mat.shape})")
        
        for name, solver, kwargs in strategies:
            t0 = time.time()
            _, score, moves = solver(mat.copy(), verbose=False, **kwargs)
            elapsed = time.time() - t0
            
            results[name]["scores"].append(score)
            results[name]["moves"].append(len(moves))
            results[name]["times"].append(elapsed)
        
        print(f"   완료!")
    
    summary = []
    for name, _, _ in strategies:
        avg_score = np.mean(results[name]["scores"])
        avg_moves = np.mean(results[name]["moves"])
        avg_time = np.mean(results[name]["times"])
        summary.append((name, avg_score, avg_moves, avg_time))
    
    # 점수 순으로 정렬
    summary.sort(key=lambda x: -x[1])
    
    print("\n" + "=" * 70)
    print("🏆 최종 순위 (평균 점수 기준)")
    print("=" * 70)
    print(f"{'순위':<3} {'전략':<17} {'평균점수':>10} {'평균횟수':>7} {'평균시간':>6}")
    print("-" * 70)
    
    for i, (name, avg_score, avg_moves, avg_time) in enumerate(summary, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{medal}{i:<3} {name:<22} {avg_score:>10.2f} {avg_moves:>10.2f} {avg_time:>10.3f}s")
    
    print("-" * 70)
    print(f"\n✨ 최고 전략: {summary[0][0]} (평균 {summary[0][1]:.2f}점)")
    
    # 보드별 상세 결과 출력
    # print("\n" + "=" * 70)
    # print("📋 보드별 상세 결과")
    # print("=" * 70)
    
    # for i, board_file in enumerate(board_files):
    #     print(f"\n[{board_file}]")
    #     board_results = []
    #     for name, _, _ in strategies:
    #         score = results[name]["scores"][i]
    #         moves = results[name]["moves"][i]
    #         board_results.append((name, score, moves))
        
    #     board_results.sort(key=lambda x: -x[1])
    #     for name, score, moves in board_results:
    #         print(f"  {name:<22}: {score:>4}점, {moves:>3}회")


if __name__ == "__main__":
    benchmark_all()
