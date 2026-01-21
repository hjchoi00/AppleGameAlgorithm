"""
train_rl.py - 사과게임 강화학습 훈련 스크립트

MaskablePPO를 사용하여 최적 전략 학습
"""

import os
import random
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt


def set_seed(seed=42):
    """재현성을 위한 seed 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🎲 Seed 고정: {seed}")


# Stable Baselines 3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# MaskablePPO (Action Masking 지원)
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    MASKABLE_AVAILABLE = True
except ImportError:
    MASKABLE_AVAILABLE = False
    print("⚠️ sb3-contrib가 설치되지 않았습니다. pip install sb3-contrib")

# 환경
from apple_env import AppleGameEnv, AppleGameEnvWithMask

# 기존 알고리즘 비교용
from main import (
    read_matrix, find_candidates_fast, apply_move_fast,
    solve_pair_first, solve_full_rollout
)


class LoggingCallback(BaseCallback):
    """학습 중 로그 출력, Train/Val 평가 및 Best 모델 저장 콜백"""
    
    def __init__(self, log_freq=10000, eval_freq=5000, 
                 train_board_dir="board_mat/train",
                 val_board_dir="board_mat/val",
                 save_path="models/ppo_apple",
                 verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.episode_scores = []
        
        # timestep 기준 카운터
        self._next_log_timestep = log_freq
        self._next_eval_timestep = eval_freq
        
        # 그래프용 데이터
        self.timesteps_history = []
        self.train_avg_history = []  # Train 보드 평균 점수
        self.val_avg_history = []    # Val 보드 평균 점수
        
        # Best 모델 추적 (Val 기준)
        self.best_val_score = -float('inf')
        self.best_timestep = 0
        
        # Train 보드 파일 로드
        self.train_board_files = sorted([
            os.path.join(train_board_dir, f) 
            for f in os.listdir(train_board_dir) 
            if f.endswith(".txt")
        ])
        
        # Val 보드 파일 로드
        self.val_board_files = sorted([
            os.path.join(val_board_dir, f) 
            for f in os.listdir(val_board_dir) 
            if f.endswith(".txt")
        ])
        
        print(f"📋 Train 보드: {len(self.train_board_files)}개")
        print(f"📋 Val 보드: {len(self.val_board_files)}개")
        
    def _on_step(self) -> bool:
        # 에피소드 종료 시 기록 (로그용)
        if self.locals.get("dones") is not None:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals["infos"][i]
                    if "total_score" in info:
                        self.episode_scores.append(info["total_score"])
        
        # 주기적으로 로그 출력 (num_timesteps 기준)
        if self.num_timesteps >= self._next_log_timestep and self.episode_scores:
            avg_score = np.mean(self.episode_scores[-100:])
            print(f"[Step {self.num_timesteps:,}] 최근 100 에피소드 평균: {avg_score:.1f}")
            self._next_log_timestep += self.log_freq
        
        # Train/Val 보드 평가 (num_timesteps 기준)
        if self.num_timesteps >= self._next_eval_timestep:
            train_avg = self._evaluate_on_boards(self.train_board_files)
            val_avg = self._evaluate_on_boards(self.val_board_files)
            
            self.timesteps_history.append(self.num_timesteps)
            self.train_avg_history.append(train_avg)
            self.val_avg_history.append(val_avg)
            
            # Best 모델 저장 (Val 기준)
            is_best = val_avg > self.best_val_score
            if is_best:
                self.best_val_score = val_avg
                self.best_timestep = self.num_timesteps
                best_path = self.save_path + "_best"
                self.model.save(best_path)
                print(f"[Step {self.num_timesteps:,}] 🏆 Train: {train_avg:.1f} | Val: {val_avg:.1f} ⭐ NEW BEST! 저장됨")
            else:
                print(f"[Step {self.num_timesteps:,}] 📊 Train: {train_avg:.1f} | Val: {val_avg:.1f} (best: {self.best_val_score:.1f} @ {self.best_timestep:,})")
            
            self._next_eval_timestep += self.eval_freq
        
        return True
    
    def _evaluate_on_boards(self, board_files):
        """지정된 보드들에서 현재 모델 평가"""
        scores = []
        
        for board_path in board_files:
            mat = read_matrix(board_path)
            
            # 평가용 환경 생성 (seed=None으로 학습 RNG에 영향 없음)
            env = make_env(use_mask=True)()
            env.reset()  # Monitor가 step 허용하도록 reset 필요
            
            # 보드 교체 (reset 직후 덮어쓰기)
            unwrapped = env.unwrapped
            unwrapped.board = mat.copy().astype(np.int32)
            unwrapped.candidates = list(find_candidates_fast(unwrapped.board))
            unwrapped.total_score = 0
            unwrapped.steps = 0
            unwrapped._compute_next_candidates_cache()
            obs = unwrapped._get_obs()
            
            # 플레이
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            
            scores.append(info["total_score"])
        
        return np.mean(scores)
    
    def plot_learning_curve(self, save_path="learning_curve.png", show=True):
        """학습 곡선 그래프 생성 (Train/Val 비교)"""
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        # 한글 폰트 설정 (Windows)
        mpl.rcParams["font.family"] = "Malgun Gothic"
        mpl.rcParams["axes.unicode_minus"] = False
        
        if not self.timesteps_history:
            print("⚠️ 기록된 데이터가 없습니다.")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Train 평균 점수
        plt.plot(self.timesteps_history, self.train_avg_history, 
                 'b-o', linewidth=2, markersize=4, label=f'Train ({len(self.train_board_files)}개)')
        
        # Val 평균 점수
        plt.plot(self.timesteps_history, self.val_avg_history, 
                 'r-s', linewidth=2, markersize=4, label=f'Val ({len(self.val_board_files)}개)')
        
        # Best 지점 표시
        if self.best_timestep > 0:
            best_idx = self.timesteps_history.index(self.best_timestep) if self.best_timestep in self.timesteps_history else -1
            if best_idx >= 0:
                plt.axvline(x=self.best_timestep, color='g', linestyle='--', alpha=0.7, label=f'Best @ {self.best_timestep:,}')
                plt.scatter([self.best_timestep], [self.val_avg_history[best_idx]], 
                           color='g', s=100, zorder=5, marker='*')
        
        plt.xlabel('Timesteps', fontsize=12)
        plt.ylabel('Average Score', fontsize=12)
        plt.title('학습 곡선 (Train vs Val)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Y축 범위 조정
        all_scores = self.train_avg_history + self.val_avg_history
        if all_scores:
            y_min = min(all_scores) - 5
            y_max = max(all_scores) + 5
            plt.ylim(y_min, y_max)
        
        plt.tight_layout()
        
        # 저장
        plt.savefig(save_path, dpi=150)
        print(f"📊 학습 곡선 저장: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


def mask_fn(env):
    """환경에서 action mask를 가져오는 함수"""
    return env.get_action_mask()


def make_env(board_dir="board_mat/train", rank=0, use_mask=False, seed=None):
    """환경 생성 함수
    
    seed가 주어지면 base_seed로 설정되어
    매 에피소드 reset마다 base_seed + episode_idx로 일관된 seed 사용
    """
    def _init():
        # base_seed 계산: seed가 있으면 seed + rank
        base_seed = (seed + rank) if seed is not None else None
        
        if use_mask:
            env = AppleGameEnvWithMask(board_dir=board_dir, base_seed=base_seed)
            env = ActionMasker(env, mask_fn)
        else:
            env = AppleGameEnv(board_dir=board_dir, base_seed=base_seed)
        
        return Monitor(env)
    return _init


def train_ppo(
    total_timesteps=100000,
    learning_rate=5e-5,
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    n_envs=2,
    save_path="models/ppo_apple",
    train_board_dir="board_mat/train",
    val_board_dir="board_mat/val",
    test_board_dir="board_mat/test"
):
    """MaskablePPO 학습 (Action Masking 적용) + Train/Val/Test 분리"""
    print("=" * 60)
    print("🧠 MaskablePPO 학습 시작 (Action Masking 적용)")
    print("=" * 60)
    
    if not MASKABLE_AVAILABLE:
        print("❌ sb3-contrib가 필요합니다: pip install sb3-contrib")
        return None
    
    # 병렬 환경 생성 (Train 데이터로 학습)
    env = DummyVecEnv([make_env(board_dir=train_board_dir, rank=i, use_mask=True, seed=42) for i in range(n_envs)])
    
    # GPU 사용 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 사용 디바이스: {device}")
    print(f"🎭 Action Masking: 활성화 (유효한 후보만 선택 가능)")
    
    # MaskablePPO 모델 생성 (Dict observation → MultiInputPolicy)
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.999,
        gae_lambda=0.99,
        clip_range=0.2,
        verbose=1,
        device=device,
        seed=42  # 재현성을 위한 시드 고정
    )
    
    # 콜백 설정 (Train/Val 평가)
    eval_freq = 5000
    callback = LoggingCallback(
        log_freq=10000, 
        eval_freq=eval_freq,
        train_board_dir=train_board_dir,
        val_board_dir=val_board_dir,
        save_path=save_path
    )
    
    # 학습
    print(f"총 {total_timesteps:,} 스텝 학습 예정 (환경 {n_envs}개 병렬)")
    print(f"📊 Train/Val 평가 주기: {eval_freq:,} 스텝마다")
    print(f"📁 Train: {train_board_dir}")
    print(f"📁 Val: {val_board_dir}")
    print(f"📁 Test: {test_board_dir}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback
    )
    
    # 최종 모델 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\n✅ 최종 모델 저장: {save_path}")
    print(f"🏆 Best 모델 저장: {save_path}_best (Val {callback.best_val_score:.1f} @ step {callback.best_timestep:,})")
    
    # 학습 곡선 그래프 생성
    callback.plot_learning_curve(
        save_path=save_path.replace(".zip", "") + "_learning_curve.png",
        show=True
    )
    
    # ========== Test 평가 (Best 모델 사용) ==========
    print("\n" + "=" * 60)
    print("🧪 Test 데이터 평가 (Best 모델 사용)")
    print("=" * 60)
    
    best_model = MaskablePPO.load(save_path + "_best")
    compare_with_heuristics(best_model, board_dir=test_board_dir, verbose=True)
    
    return model, callback


def evaluate_model_on_all_boards(model, env_factory, board_dir="board_mat", verbose=False):
    """
    board_mat의 모든 보드에서 평가 (고정 보드 전체 평가)
    
    Args:
        model: 학습된 모델
        env_factory: 환경 생성 함수
        board_dir: 보드 파일 디렉토리
        verbose: 각 보드별 점수 출력 여부
    """
    # 모든 보드 파일 로드
    board_files = sorted([
        os.path.join(board_dir, f) 
        for f in os.listdir(board_dir) 
        if f.endswith(".txt")
    ])
    
    if not board_files:
        print(f"⚠️ {board_dir}에 보드 파일이 없습니다.")
        return None
    
    scores = []
    steps_list = []
    
    for board_path in board_files:
        mat = read_matrix(board_path)
        board_name = os.path.basename(board_path)
        
        # 환경 생성 및 초기화
        env = env_factory()
        env.reset()
        
        # 보드 교체
        unwrapped = env.unwrapped
        unwrapped.board = mat.copy().astype(np.int32)
        unwrapped.candidates = list(find_candidates_fast(unwrapped.board))
        unwrapped.total_score = 0
        unwrapped.steps = 0
        obs = unwrapped._get_obs()
        
        # 플레이
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        scores.append(info["total_score"])
        steps_list.append(info["steps"])
        
        if verbose:
            print(f"  {board_name}: {info['total_score']} 점")
    
    return {
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "max_score": np.max(scores),
        "min_score": np.min(scores),
        "mean_steps": np.mean(steps_list),
        "n_boards": len(board_files),
        "scores": scores
    }


def compare_with_heuristics(model, board_dir="board_mat", verbose=True):
    """학습된 모델과 기존 휴리스틱 비교"""
    print("\n" + "=" * 60)
    print("📊 RL 모델 vs 휴리스틱 비교")
    print("=" * 60)
    
    board_files = sorted([
        os.path.join(board_dir, f) 
        for f in os.listdir(board_dir) 
        if f.endswith(".txt")
    ])
    
    results = {
        "RL Model": [],
        "Pair-First": [],
        "Full-Rollout": []
    }
    
    for board_path in board_files:
        mat = read_matrix(board_path)
        board_name = os.path.basename(board_path)
        
        # RL 모델
        env = make_env(use_mask=True)()
        env.reset()
        unwrapped = env.unwrapped
        unwrapped.board = mat.copy().astype(np.int32)
        unwrapped.candidates = list(find_candidates_fast(unwrapped.board))
        unwrapped.total_score = 0
        unwrapped.steps = 0
        obs = unwrapped._get_obs()
        
        while unwrapped.candidates:
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


def play_with_model(model, board_path=None, render=True):
    """학습된 모델로 게임 플레이"""
    env = make_env(use_mask=True)()
    unwrapped = env.unwrapped
    unwrapped.render_mode = "human" if render else None
    
    obs, _ = env.reset()
    
    if board_path:
        unwrapped.board = read_matrix(board_path).astype(np.int32)
        unwrapped.candidates = list(find_candidates_fast(unwrapped.board))
        unwrapped.total_score = 0
        unwrapped.steps = 0
        obs = unwrapped._get_obs()
    
    if render:
        unwrapped.render()
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if render:
            unwrapped.render()
        
        if terminated or truncated:
            break
    
    print(f"\n🏁 게임 종료! 최종 점수: {info['total_score']}, 스텝: {info['steps']}")
    return info


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="사과게임 강화학습")
    parser.add_argument("--mode", type=str, default="train", 
                        choices=["train", "eval", "play", "compare"])
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--model", type=str, default=None, help="평가/플레이할 모델 경로")
    parser.add_argument("--board", type=str, default=None, help="특정 보드로 플레이")
    parser.add_argument("--seed", type=int, default=42, help="재현성을 위한 랜덤 시드 (기본: 42, -1이면 랜덤)")
    parser.add_argument("--test-dir", type=str, default="board_mat/test", help="테스트 데이터 경로")
    args = parser.parse_args()
    
    # Seed 고정 (-1이면 랜덤)
    if args.seed >= 0:
        set_seed(args.seed)
    else:
        print("🎲 Seed: 랜덤 (고정 안함)")
    
    if args.mode == "train":
        # 학습 (MaskablePPO) - Train/Val/Test 분리
        print("=" * 60)
        print("🧠 MaskablePPO 학습 (Train/Val/Test 분리)")
        print("=" * 60)
        model, callback = train_ppo(total_timesteps=args.timesteps)
        # train_ppo 내에서 Test 평가까지 수행
        
    elif args.mode == "eval":
        # 평가만 (Test 데이터)
        model_path = args.model or "models/ppo_apple_best"
        
        if MASKABLE_AVAILABLE:
            model = MaskablePPO.load(model_path)
        else:
            model = PPO.load(model_path)
        
        print(f"\n📊 모델 평가 중 ({args.test_dir})...")
        compare_with_heuristics(model, board_dir=args.test_dir, verbose=True)
        
    elif args.mode == "play":
        # 플레이
        model_path = args.model or "models/ppo_apple_best"
        
        if MASKABLE_AVAILABLE:
            model = MaskablePPO.load(model_path)
        else:
            model = PPO.load(model_path)
        
        board_path = f"board_mat/test/{args.board}.txt" if args.board else None
        play_with_model(model, board_path, render=True)
        
    elif args.mode == "compare":
        # 비교 (Test 데이터)
        model_path = args.model or "models/ppo_apple_best"
        
        if MASKABLE_AVAILABLE:
            model = MaskablePPO.load(model_path)
        else:
            model = PPO.load(model_path)
        
        compare_with_heuristics(model, board_dir=args.test_dir)
