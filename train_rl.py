"""
train_rl.py - 사과게임 강화학습 훈련 스크립트

MaskablePPO를 사용하여 최적 전략 학습
"""

import os
import numpy as np
import torch
from datetime import datetime

# Stable Baselines 3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# MaskablePPO (Action Masking 지원)
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    MASKABLE_AVAILABLE = True
except ImportError:
    MASKABLE_AVAILABLE = False
    print("⚠️ sb3-contrib가 설치되지 않았습니다. pip install sb3-contrib")

# 환경
from apple_env import AppleGameEnv, AppleGameEnvWithMask, AppleGameEnvTopK, AppleGameEnvTopKFlat

# 기존 알고리즘 비교용
from main import (
    read_matrix, find_candidates_fast, apply_move_fast,
    solve_pair_first, solve_full_rollout
)


class LoggingCallback(BaseCallback):
    """학습 중 로그 출력 콜백"""
    
    def __init__(self, log_freq=1000, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_scores = []
        
    def _on_step(self) -> bool:
        # 에피소드 종료 시 기록
        if self.locals.get("dones") is not None:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals["infos"][i]
                    if "total_score" in info:
                        self.episode_scores.append(info["total_score"])
        
        # 주기적으로 출력
        if self.n_calls % self.log_freq == 0 and self.episode_scores:
            avg_score = np.mean(self.episode_scores[-100:])
            max_score = np.max(self.episode_scores[-100:]) if self.episode_scores else 0
            print(f"[Step {self.n_calls}] 최근 100 에피소드 - 평균: {avg_score:.1f}, 최고: {max_score}")
        
        return True


def mask_fn(env):
    """환경에서 action mask를 가져오는 함수"""
    return env.get_action_mask()


def make_env(board_dir="board_mat", rank=0, use_mask=False):
    """환경 생성 함수"""
    def _init():
        if use_mask:
            env = AppleGameEnvWithMask(board_dir=board_dir)
            # ActionMasker wrapper 적용 (마스킹 명시적 활성화)
            env = ActionMasker(env, mask_fn)
        else:
            env = AppleGameEnv(board_dir=board_dir)
        env = Monitor(env)
        return env
    return _init


def make_env_topk(board_dir="board_mat", rank=0, top_k=20):
    """Top-K 환경 생성 함수 (Flat 버전 - MLP Policy 호환)"""
    def _init():
        env = AppleGameEnvTopKFlat(board_dir=board_dir, top_k=top_k)
        # ActionMasker wrapper 적용
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env
    return _init


def train_ppo(
    total_timesteps=100000,
    learning_rate=3e-4,
    n_steps=4096,
    batch_size=256,
    n_epochs=10,
    n_envs=4,
    save_path="models/ppo_apple"
):
    """MaskablePPO 학습 (Action Masking 적용)"""
    print("=" * 60)
    print("🧠 MaskablePPO 학습 시작 (Action Masking 적용)")
    print("=" * 60)
    
    if not MASKABLE_AVAILABLE:
        print("❌ sb3-contrib가 필요합니다: pip install sb3-contrib")
        return None
    
    # 병렬 환경 생성 (Mask 지원 환경)
    env = DummyVecEnv([make_env(rank=i, use_mask=True) for i in range(n_envs)])
    
    # GPU 사용 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 사용 디바이스: {device}")
    print(f"🎭 Action Masking: 활성화 (유효한 후보만 선택 가능)")
    
    # MaskablePPO 모델 생성
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.995,
        gae_lambda=0.97,
        clip_range=0.2,
        verbose=1,
        device=device
    )
    
    # 콜백 설정
    callback = LoggingCallback(log_freq=5000)
    
    # 학습
    print(f"총 {total_timesteps:,} 스텝 학습 예정 (환경 {n_envs}개 병렬)")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback
    )
    
    # 모델 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"✅ 모델 저장: {save_path}")
    
    return model


def train_ppo_topk(
    total_timesteps=100000,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    n_envs=4,
    top_k=20,
    save_path="models/ppo_topk_apple"
):
    """MaskablePPO + Top-K 학습
    
    휴리스틱으로 상위 K개 후보만 선택지로 제공.
    Action space가 작아져서 학습 효율 증가.
    """
    print("=" * 60)
    print(f"🧠 MaskablePPO + Top-{top_k} 학습 시작")
    print("=" * 60)
    
    if not MASKABLE_AVAILABLE:
        print("❌ sb3-contrib가 필요합니다: pip install sb3-contrib")
        return None
    
    # 병렬 환경 생성 (Top-K 환경)
    env = DummyVecEnv([make_env_topk(rank=i, top_k=top_k) for i in range(n_envs)])
    
    # GPU 사용 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 사용 디바이스: {device}")
    print(f"🎯 Top-K: {top_k} (휴리스틱으로 선별된 상위 {top_k}개 후보만 선택 가능)")
    print(f"🎭 Action Masking: 활성화")
    print(f"📊 단순 보상: cells + (종료 시 -remaining)")
    print(f"👁️ 관측: 보드 + Top-K 후보 특징(9차원) + 마스크")
    
    # MaskablePPO 모델 생성
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.995,
        gae_lambda=0.97,
        clip_range=0.2,
        verbose=1,
        device=device
    )
    
    # 콜백 설정
    callback = LoggingCallback(log_freq=5000)
    
    # 학습
    print(f"총 {total_timesteps:,} 스텝 학습 예정 (환경 {n_envs}개 병렬)")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback
    )
    
    # 모델 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"✅ 모델 저장: {save_path}")
    
    return model, top_k


def evaluate_model(model, env, n_episodes=10):
    """학습된 모델 평가 (env는 make_env_topk로 생성된 wrapped env)"""
    scores = []
    steps_list = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        
        while True:
            # ActionMasker가 이미 적용되어 있으므로 action_masks 불필요
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                scores.append(info["total_score"])
                steps_list.append(info["steps"])
                break
    
    return {
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "max_score": np.max(scores),
        "min_score": np.min(scores),
        "mean_steps": np.mean(steps_list)
    }


def compare_with_heuristics(model, board_paths, verbose=True, top_k=20):
    """학습된 모델과 기존 휴리스틱 비교 (make_env_topk와 동일한 환경 사용)"""
    print("\n" + "=" * 60)
    print("📊 RL 모델 vs 휴리스틱 비교")
    print("=" * 60)
    
    results = {
        "RL Model": [],
        "Pair-First": [],
        "Full-Rollout": []
    }
    
    for board_path in board_paths:
        mat = read_matrix(board_path)
        board_name = os.path.basename(board_path)
        
        # RL 모델: make_env_topk와 동일한 wrapped 환경 사용
        env = make_env_topk(top_k=top_k)()
        
        # 먼저 reset() 호출하여 Monitor 상태 초기화
        env.reset()
        
        # 그 후 보드를 원하는 보드로 교체 (Monitor -> ActionMasker -> AppleGameEnvTopKFlat)
        unwrapped = env.unwrapped
        unwrapped.board = mat.copy().astype(np.int32)
        unwrapped.all_candidates = list(find_candidates_fast(unwrapped.board))
        unwrapped.top_candidates = unwrapped._select_top_k(unwrapped.all_candidates)
        unwrapped.prev_num_candidates = len(unwrapped.all_candidates)
        unwrapped.total_score = 0
        unwrapped.steps = 0
        obs = unwrapped._get_obs()
        
        while unwrapped.top_candidates:
            # ActionMasker가 적용되어 있으므로 action_masks 불필요
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        results["RL Model"].append(info["total_score"])
        
        # Pair-First
        _, score_pf, _ = solve_pair_first(mat.copy(), verbose=False)
        results["Pair-First"].append(score_pf)
        
        # Full-Rollout
        _, score_fr, _ = solve_full_rollout(mat.copy(), top_k=30, verbose=False)
        results["Full-Rollout"].append(score_fr)
        
        if verbose:
            print(f"\n[{board_name}]")
            print(f"  RL Model:     {results['RL Model'][-1]:>4}")
            print(f"  Pair-First:   {results['Pair-First'][-1]:>4}")
            print(f"  Full-Rollout: {results['Full-Rollout'][-1]:>4}")
    
    # 평균 출력
    print("\n" + "-" * 60)
    print("📈 평균 점수:")
    for name, scores in results.items():
        print(f"  {name:<15}: {np.mean(scores):.1f} (±{np.std(scores):.1f})")
    
    return results


def play_with_model(model, board_path=None, render=True, top_k=20):
    """학습된 모델로 게임 플레이 (make_env_topk와 동일한 환경 사용)"""
    # make_env_topk와 동일한 wrapped 환경 사용
    env = make_env_topk(top_k=top_k)()
    unwrapped = env.unwrapped
    unwrapped.render_mode = "human" if render else None
    
    # 먼저 reset() 호출하여 Monitor 상태 초기화
    obs, _ = env.reset()
    
    if board_path:
        # 보드를 원하는 보드로 교체
        unwrapped.board = read_matrix(board_path).astype(np.int32)
        unwrapped.all_candidates = list(find_candidates_fast(unwrapped.board))
        unwrapped.top_candidates = unwrapped._select_top_k(unwrapped.all_candidates)
        unwrapped.prev_num_candidates = len(unwrapped.all_candidates)
        unwrapped.total_score = 0
        unwrapped.steps = 0
        obs = unwrapped._get_obs()
    
    if render:
        unwrapped.render()
    
    while True:
        # ActionMasker가 적용되어 있으므로 action_masks 불필요
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if render:
            unwrapped.render()
        
        if terminated or truncated:
            break
    
    print(f"\n🏁 게임 종료! 최종 점수: {info['total_score']}, 스텝: {info['steps']}")
    return info


def randomized_search(
    n_trials=20,
    timesteps_per_trial=50000,
    eval_episodes=30,
    save_best=True
):
    """
    랜덤 서치로 최적 하이퍼파라미터 탐색
    
    탐색 범위:
    - top_k: [10, 20, 30, 50]
    - learning_rate: [1e-4, 3e-4, 1e-3]
    - n_steps: [1024, 2048, 4096]
    - batch_size: [32, 64, 128, 256]
    - n_epochs: [5, 10, 15, 20]
    - gamma: [0.99, 0.995, 0.999]
    - gae_lambda: [0.9, 0.95, 0.97, 0.99]
    """
    import random
    import json
    from datetime import datetime
    
    print("=" * 70)
    print("🔍 Randomized Search for Hyperparameter Optimization")
    print("=" * 70)
    
    # 탐색 공간 정의
    search_space = {
        "top_k": [10, 20, 30, 50],
        "learning_rate": [1e-4, 3e-4, 5e-4, 1e-3],
        "n_steps": [1024, 2048, 4096],
        "batch_size": [32, 64, 128, 256],
        "n_epochs": [5, 10, 15, 20],
        "gamma": [0.99, 0.995, 0.999],
        "gae_lambda": [0.9, 0.95, 0.97, 0.99],
        "n_envs": [2, 4, 8]
    }
    
    print(f"📋 탐색 공간:")
    for key, values in search_space.items():
        print(f"   {key}: {values}")
    print(f"\n🎲 총 {n_trials}회 시도, 각 {timesteps_per_trial:,} 스텝")
    print("-" * 70)
    
    # 결과 저장
    results = []
    best_score = -float('inf')
    best_params = None
    best_model = None
    
    # 고정 보드 파일로 평가
    board_files = [f"board_mat/board{i}.txt" for i in range(1, 9)]
    board_files = [f for f in board_files if os.path.exists(f)]
    
    for trial in range(n_trials):
        print(f"\n{'='*70}")
        print(f"🎯 Trial {trial + 1}/{n_trials}")
        print("=" * 70)
        
        # 랜덤 파라미터 샘플링
        params = {key: random.choice(values) for key, values in search_space.items()}
        
        # batch_size가 n_steps * n_envs보다 크면 조정
        total_batch = params["n_steps"] * params["n_envs"]
        if params["batch_size"] > total_batch:
            params["batch_size"] = total_batch
        
        print(f"📊 파라미터: {params}")
        
        try:
            # 환경 생성
            env = DummyVecEnv([
                make_env_topk(rank=i, top_k=params["top_k"]) 
                for i in range(params["n_envs"])
            ])
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # 모델 생성 및 학습
            model = MaskablePPO(
                "MlpPolicy",
                env,
                learning_rate=params["learning_rate"],
                n_steps=params["n_steps"],
                batch_size=params["batch_size"],
                n_epochs=params["n_epochs"],
                gamma=params["gamma"],
                gae_lambda=params["gae_lambda"],
                clip_range=0.2,
                verbose=0,
                device=device
            )
            
            model.learn(total_timesteps=timesteps_per_trial)
            
            # 평가: 고정 보드에서 점수 측정
            eval_scores = []
            for board_path in board_files:
                mat = read_matrix(board_path)
                eval_env = make_env_topk(top_k=params["top_k"])()
                
                # reset 후 보드 교체
                eval_env.reset()
                unwrapped = eval_env.unwrapped
                unwrapped.board = mat.copy().astype(np.int32)
                unwrapped.all_candidates = list(find_candidates_fast(unwrapped.board))
                unwrapped.top_candidates = unwrapped._select_top_k(unwrapped.all_candidates)
                unwrapped.prev_num_candidates = len(unwrapped.all_candidates)
                unwrapped.total_score = 0
                unwrapped.steps = 0
                obs = unwrapped._get_obs()
                
                while unwrapped.top_candidates:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    if terminated:
                        break
                eval_scores.append(info["total_score"])
            
            # 랜덤 보드에서도 평가
            random_env = make_env_topk(top_k=params["top_k"])()
            random_scores = []
            for _ in range(eval_episodes):
                obs, _ = random_env.reset()
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = random_env.step(action)
                    if terminated or truncated:
                        random_scores.append(info["total_score"])
                        break
            
            # 점수 계산 (고정 보드 + 랜덤 보드 평균)
            fixed_mean = np.mean(eval_scores)
            random_mean = np.mean(random_scores)
            combined_score = 0.6 * fixed_mean + 0.4 * random_mean  # 고정 보드 중시
            
            result = {
                "trial": trial + 1,
                "params": params,
                "fixed_board_mean": fixed_mean,
                "fixed_board_scores": eval_scores,
                "random_board_mean": random_mean,
                "combined_score": combined_score
            }
            results.append(result)
            
            print(f"✅ 고정 보드 평균: {fixed_mean:.1f}")
            print(f"✅ 랜덤 보드 평균: {random_mean:.1f}")
            print(f"✅ 종합 점수: {combined_score:.1f}")
            
            # 최고 점수 갱신
            if combined_score > best_score:
                best_score = combined_score
                best_params = params.copy()
                best_model = model
                print(f"🏆 새로운 최고 점수! ({combined_score:.1f})")
                
        except Exception as e:
            print(f"❌ Trial {trial + 1} 실패: {e}")
            results.append({
                "trial": trial + 1,
                "params": params,
                "error": str(e)
            })
    
    # 결과 출력
    print("\n" + "=" * 70)
    print("📊 Randomized Search 결과")
    print("=" * 70)
    
    # 성공한 결과만 정렬
    successful = [r for r in results if "combined_score" in r]
    successful.sort(key=lambda x: x["combined_score"], reverse=True)
    
    print(f"\n🏅 Top 5 결과:")
    for i, r in enumerate(successful[:5], 1):
        print(f"\n{i}. 종합 점수: {r['combined_score']:.1f}")
        print(f"   고정 보드: {r['fixed_board_mean']:.1f}, 랜덤 보드: {r['random_board_mean']:.1f}")
        print(f"   파라미터: {r['params']}")
    
    print(f"\n🏆 최적 파라미터:")
    print(f"   {best_params}")
    print(f"   최고 점수: {best_score:.1f}")
    
    # 결과 저장
    os.makedirs("search_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON 저장
    results_file = f"search_results/search_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "best_params": best_params,
            "best_score": best_score,
            "all_results": results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n📁 결과 저장: {results_file}")
    
    # 최고 모델 저장
    if save_best and best_model:
        model_path = f"models/best_search_{timestamp}"
        best_model.save(model_path)
        print(f"📁 최고 모델 저장: {model_path}")
    
    return best_params, best_score, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="사과게임 강화학습")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "play", "compare", "search"])
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--topk", type=int, default=20, help="Top-K 후보 수 (기본: 20)")
    parser.add_argument("--model", type=str, default=None, help="평가/플레이할 모델 경로")
    parser.add_argument("--board", type=str, default=None, help="특정 보드로 플레이")
    parser.add_argument("--n_trials", type=int, default=20, help="랜덤 서치 시도 횟수")
    parser.add_argument("--trial_timesteps", type=int, default=50000, help="시도당 학습 스텝")
    args = parser.parse_args()
    
    if args.mode == "train":
        # 학습 (MaskablePPO + Top-K)
        top_k_value = args.topk
        model, top_k_value = train_ppo_topk(total_timesteps=args.timesteps, top_k=args.topk)
        
        # 학습 후 평가: make_env_topk와 동일한 환경 사용
        print("\n📊 학습된 모델 평가 중...")
        eval_env = make_env_topk(top_k=top_k_value)()
        results = evaluate_model(model, eval_env, n_episodes=20)
        print(f"평균 점수: {results['mean_score']:.1f} (±{results['std_score']:.1f})")
        print(f"최고 점수: {results['max_score']}")
        
        # 휴리스틱과 비교
        board_files = [f"board_mat/board{i}.txt" for i in range(1, 9)]
        board_files = [f for f in board_files if os.path.exists(f)]
        compare_with_heuristics(model, board_files, top_k=top_k_value)
        
    elif args.mode == "eval":
        # 평가만
        model_path = args.model or "models/ppo_topk_apple"
        top_k_value = args.topk
        
        if MASKABLE_AVAILABLE:
            model = MaskablePPO.load(model_path)
        else:
            model = PPO.load(model_path)
        
        eval_env = make_env_topk(top_k=top_k_value)()
        results = evaluate_model(model, eval_env, n_episodes=50)
        print(f"평균 점수: {results['mean_score']:.1f} (±{results['std_score']:.1f})")
        print(f"최고/최저: {results['max_score']} / {results['min_score']}")
        
    elif args.mode == "play":
        # 플레이
        model_path = args.model or "models/ppo_topk_apple"
        top_k_value = args.topk
        
        if MASKABLE_AVAILABLE:
            model = MaskablePPO.load(model_path)
        else:
            model = PPO.load(model_path)
        
        board_path = f"board_mat/{args.board}.txt" if args.board else None
        play_with_model(model, board_path, render=True, top_k=top_k_value)
        
    elif args.mode == "compare":
        # 비교
        model_path = args.model or "models/ppo_topk_apple"
        top_k_value = args.topk
        
        if MASKABLE_AVAILABLE:
            model = MaskablePPO.load(model_path)
        else:
            model = PPO.load(model_path)
        
        board_files = [f"board_mat/board{i}.txt" for i in range(1, 9)]
        board_files = [f for f in board_files if os.path.exists(f)]
        compare_with_heuristics(model, board_files, top_k=top_k_value)
    
    elif args.mode == "search":
        # 랜덤 서치
        print("🔍 하이퍼파라미터 랜덤 서치 시작...")
        best_params, best_score, results = randomized_search(
            n_trials=args.n_trials,
            timesteps_per_trial=args.trial_timesteps,
            eval_episodes=30,
            save_best=True
        )
        print(f"\n🏆 최적 파라미터: {best_params}")
        print(f"🏆 최고 점수: {best_score:.1f}")
