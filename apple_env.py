"""
apple_env.py - 사과게임 강화학습 환경 (Gymnasium 호환)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import random
from main import find_candidates_fast, apply_move_fast, read_matrix


class AppleGameEnv(gym.Env):
    """
    사과게임 강화학습 환경
    
    State: 10x17 보드 (0~9 숫자)
    Action: 후보 리스트에서 인덱스 선택
    Reward: 제거한 사과 개수
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(self, board_dir="board_mat", max_candidates=500, render_mode=None):
        super().__init__()
        
        self.board_dir = board_dir
        self.max_candidates = max_candidates
        self.render_mode = render_mode
        
        # 보드 파일 목록 로드
        self.board_files = [
            os.path.join(board_dir, f) 
            for f in os.listdir(board_dir) 
            if f.endswith(".txt")
        ]
        
        if not self.board_files:
            raise ValueError(f"No board files found in {board_dir}")
        
        # 보드 크기 확인 (첫 번째 파일 기준)
        sample_board = read_matrix(self.board_files[0])
        self.board_height, self.board_width = sample_board.shape
        
        # Observation space: 보드 상태 (0~9 정규화)
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(self.board_height, self.board_width), 
            dtype=np.float32
        )
        
        # Action space: 후보 인덱스 (최대 max_candidates개)
        self.action_space = spaces.Discrete(max_candidates)
        
        # 현재 상태
        self.board = None
        self.candidates = []
        self.total_score = 0
        self.steps = 0
        
    def reset(self, seed=None, options=None):
        """새 에피소드 시작"""
        super().reset(seed=seed)
        
        # 랜덤 보드 선택 또는 랜덤 생성
        if random.random() < 0.5 and self.board_files:
            # 기존 보드 파일에서 선택
            board_path = random.choice(self.board_files)
            self.board = read_matrix(board_path).astype(np.int32)
        else:
            # 랜덤 보드 생성 (1~9)
            self.board = np.random.randint(1, 10, size=(self.board_height, self.board_width), dtype=np.int32)
        
        self.candidates = list(find_candidates_fast(self.board))
        self.total_score = 0
        self.steps = 0
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        """액션 실행"""
        # 유효한 액션인지 확인
        if not self.candidates:
            # 게임 종료
            return self._get_obs(), 0, True, False, self._get_info()
        
        # 액션 인덱스를 유효 범위로 클리핑
        action = action % len(self.candidates)
        
        # 선택한 후보 실행
        r1, c1, r2, c2, cells, area = self.candidates[action]
        apply_move_fast(self.board, r1, c1, r2, c2)
        
        # 보상 계산
        reward = cells  # 제거한 사과 개수
        self.total_score += cells
        self.steps += 1
        
        # 새로운 후보 탐색
        self.candidates = list(find_candidates_fast(self.board))
        
        # 종료 조건
        terminated = len(self.candidates) == 0
        truncated = False
        
        # 게임 종료 시 보너스
        if terminated:
            # 남은 사과가 적을수록 보너스
            remaining = np.sum(self.board > 0)
            reward += max(0, 50 - remaining)  # 최대 50점 보너스
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _get_obs(self):
        """관측값 반환 (정규화된 보드)"""
        return (self.board / 9.0).astype(np.float32)
    
    def _get_info(self):
        """추가 정보 반환"""
        return {
            "total_score": self.total_score,
            "steps": self.steps,
            "candidates": len(self.candidates),
            "remaining": np.sum(self.board > 0)
        }
    
    def get_valid_actions(self):
        """유효한 액션 인덱스 리스트 반환"""
        return list(range(len(self.candidates)))
    
    def get_action_mask(self):
        """유효한 액션 마스크 반환 (True = 유효)"""
        mask = np.zeros(self.max_candidates, dtype=bool)
        n_valid = len(self.candidates)
        mask[:n_valid] = True
        
        # 디버그: mask 검증 (주석 해제하여 사용)
        # assert mask.shape == (self.action_space.n,), f"Mask shape mismatch: {mask.shape} vs {self.action_space.n}"
        # assert mask.dtype == bool, f"Mask dtype mismatch: {mask.dtype}"
        # print(f"[DEBUG] candidates: {n_valid}, mask true: {int(mask.sum())}")
        
        return mask
    
    def render(self):
        """보드 시각화"""
        if self.render_mode == "human" or self.render_mode == "ansi":
            print(f"\n=== Step {self.steps} | Score: {self.total_score} | Candidates: {len(self.candidates)} ===")
            print(self.board)
            return None
    
    def close(self):
        pass


class AppleGameEnvWithMask(AppleGameEnv):
    """
    액션 마스킹을 지원하는 환경
    (Maskable PPO 등에서 사용)
    """
    
    def action_masks(self):
        """sb3-contrib의 MaskablePPO용 마스크"""
        return self.get_action_mask()


class AppleGameEnvTopK(gym.Env):
    """
    Top-K 방식의 사과게임 환경 (개선된 버전)
    
    개선사항:
    1. 보상 재설계: cells + 후보 증가 보상 + 남은 사과 패널티 + 강화된 종료 보너스
    2. 관측 개선: 보드 텐서 + Top-K 후보 특징 행렬
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    # 후보 특징 개수: cells, area, zeros, height, width, r1, c1, r2, c2 (정규화)
    N_FEATURES = 9
    
    def __init__(self, board_dir="board_mat", top_k=20, render_mode=None,
                 alpha=0.01, beta=0.1, gamma=2.0):
        """
        Args:
            top_k: 상위 K개 후보만 선택
            alpha: 남은 사과 패널티 계수
            beta: 후보 증가 보상 계수
            gamma: 종료 보너스 계수
        """
        super().__init__()
        
        self.board_dir = board_dir
        self.top_k = top_k
        self.render_mode = render_mode
        
        # 보상 계수
        self.alpha = alpha  # 남은 사과 패널티
        self.beta = beta    # 후보 증가 보상
        self.gamma = gamma  # 종료 보너스
        
        # 보드 파일 목록 로드
        self.board_files = [
            os.path.join(board_dir, f) 
            for f in os.listdir(board_dir) 
            if f.endswith(".txt")
        ]
        
        if not self.board_files:
            raise ValueError(f"No board files found in {board_dir}")
        
        # 보드 크기 확인
        sample_board = read_matrix(self.board_files[0])
        self.board_height, self.board_width = sample_board.shape
        
        # Observation space: Dict (보드 + 후보 특징)
        self.observation_space = spaces.Dict({
            "board": spaces.Box(
                low=0, high=1, 
                shape=(self.board_height, self.board_width), 
                dtype=np.float32
            ),
            "candidates": spaces.Box(
                low=0, high=1,
                shape=(top_k, self.N_FEATURES),
                dtype=np.float32
            ),
            "valid_mask": spaces.Box(
                low=0, high=1,
                shape=(top_k,),
                dtype=np.float32
            )
        })
        
        # Action space: Top-K 중에서 선택 (0 ~ K-1)
        self.action_space = spaces.Discrete(top_k)
        
        # 현재 상태
        self.board = None
        self.all_candidates = []
        self.top_candidates = []
        self.total_score = 0
        self.steps = 0
        self.prev_num_candidates = 0
    
    def _select_top_k(self, candidates):
        """휴리스틱으로 상위 K개 후보 선택"""
        if not candidates:
            return []
        
        # 휴리스틱: cells 작은 것 > area 작은 것 (2개짜리 우선)
        scored = [(c[4], c[5], c) for c in candidates]
        scored.sort(key=lambda x: (x[0], x[1]))
        
        # 상위 K개 선택
        top = [s[2] for s in scored[:self.top_k]]
        return top
    
    def _get_candidate_features(self):
        """Top-K 후보들의 특징 행렬 반환 (K, N_FEATURES)"""
        features = np.zeros((self.top_k, self.N_FEATURES), dtype=np.float32)
        
        max_cells = 10  # 최대 cells (합이 10이므로)
        max_area = self.board_height * self.board_width
        
        for i, cand in enumerate(self.top_candidates):
            r1, c1, r2, c2, cells, area = cand
            height = r2 - r1 + 1
            width = c2 - c1 + 1
            zeros = area - cells  # 영역 내 0의 개수
            
            # 정규화된 특징
            features[i] = [
                cells / max_cells,                    # 0: cells (정규화)
                area / max_area,                      # 1: area (정규화)
                zeros / max_area,                     # 2: zeros (정규화)
                height / self.board_height,           # 3: height (정규화)
                width / self.board_width,             # 4: width (정규화)
                r1 / (self.board_height - 1),         # 5: r1 (정규화)
                c1 / (self.board_width - 1),          # 6: c1 (정규화)
                r2 / (self.board_height - 1),         # 7: r2 (정규화)
                c2 / (self.board_width - 1),          # 8: c2 (정규화)
            ]
        
        return features
        
    def reset(self, seed=None, options=None):
        """새 에피소드 시작"""
        super().reset(seed=seed)
        
        # 랜덤 보드 선택 또는 랜덤 생성
        if random.random() < 0.5 and self.board_files:
            board_path = random.choice(self.board_files)
            self.board = read_matrix(board_path).astype(np.int32)
        else:
            self.board = np.random.randint(1, 10, size=(self.board_height, self.board_width), dtype=np.int32)
        
        self.all_candidates = list(find_candidates_fast(self.board))
        self.top_candidates = self._select_top_k(self.all_candidates)
        self.total_score = 0
        self.steps = 0
        self.prev_num_candidates = len(self.all_candidates)
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        """액션 실행"""
        if not self.top_candidates:
            return self._get_obs(), 0, True, False, self._get_info()
        
        # 액션 인덱스를 유효 범위로 클리핑
        action = action % len(self.top_candidates)
        
        # 현재 상태 저장
        prev_remaining = np.sum(self.board > 0)
        prev_candidates = self.prev_num_candidates
        
        # 선택한 후보 실행
        r1, c1, r2, c2, cells, area = self.top_candidates[action]
        apply_move_fast(self.board, r1, c1, r2, c2)
        
        self.total_score += cells
        self.steps += 1
        
        # 새로운 후보 탐색
        self.all_candidates = list(find_candidates_fast(self.board))
        self.top_candidates = self._select_top_k(self.all_candidates)
        
        # 현재 상태
        curr_remaining = np.sum(self.board > 0)
        curr_candidates = len(self.all_candidates)
        
        # ========== 단순 보상 설계 (점수와 1:1 매칭) ==========
        # 기본 보상: 제거한 사과 수 (= 실제 점수 획득량)
        reward = cells
        
        # 종료 조건
        terminated = len(self.top_candidates) == 0
        truncated = False
        
        # 종료 시 남은 사과 패널티 (점수와 정합적)
        if terminated:
            reward -= curr_remaining
        
        # 다음 스텝을 위해 저장
        self.prev_num_candidates = curr_candidates
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _get_obs(self):
        """관측값 반환 (Dict: 보드 + 후보 특징 + 마스크)"""
        return {
            "board": (self.board / 9.0).astype(np.float32),
            "candidates": self._get_candidate_features(),
            "valid_mask": self.get_action_mask().astype(np.float32)
        }
    
    def _get_info(self):
        """추가 정보 반환"""
        return {
            "total_score": self.total_score,
            "steps": self.steps,
            "candidates": len(self.all_candidates),
            "top_candidates": len(self.top_candidates),
            "remaining": np.sum(self.board > 0)
        }
    
    def get_action_mask(self):
        """유효한 액션 마스크 반환"""
        mask = np.zeros(self.top_k, dtype=bool)
        n_valid = len(self.top_candidates)
        mask[:n_valid] = True
        return mask
    
    def action_masks(self):
        """sb3-contrib용"""
        return self.get_action_mask()
    
    def render(self):
        if self.render_mode == "human" or self.render_mode == "ansi":
            print(f"\n=== Step {self.steps} | Score: {self.total_score} | Top-K: {len(self.top_candidates)}/{len(self.all_candidates)} ===")
            print(self.board)
    
    def close(self):
        pass


class AppleGameEnvTopKFlat(AppleGameEnvTopK):
    """
    Top-K 환경의 Flat 버전 (MLP Policy용)
    
    Dict observation을 1D 벡터로 평탄화하여 MlpPolicy에서 사용 가능하게 함.
    """
    
    def __init__(self, board_dir="board_mat", top_k=20, render_mode=None,
                 alpha=0.01, beta=0.1, gamma=2.0):
        super().__init__(board_dir, top_k, render_mode, alpha, beta, gamma)
        
        # Observation space: Flat 벡터
        board_size = self.board_height * self.board_width
        candidate_size = top_k * self.N_FEATURES
        mask_size = top_k
        total_size = board_size + candidate_size + mask_size
        
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(total_size,),
            dtype=np.float32
        )
    
    def _get_obs(self):
        """관측값 반환 (1D 벡터)"""
        board_flat = (self.board / 9.0).astype(np.float32).flatten()
        candidates_flat = self._get_candidate_features().flatten()
        mask_flat = self.get_action_mask().astype(np.float32)
        
        return np.concatenate([board_flat, candidates_flat, mask_flat])


# 테스트용 코드
if __name__ == "__main__":
    print("🎮 사과게임 환경 테스트")
    print("=" * 50)
    
    env = AppleGameEnv(render_mode="human")
    obs, info = env.reset()
    
    print(f"보드 크기: {env.board_height}x{env.board_width}")
    print(f"초기 후보 수: {info['candidates']}")
    
    # 랜덤 에이전트로 한 에피소드 플레이
    total_reward = 0
    while True:
        # 유효한 액션 중 랜덤 선택
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break
        action = random.choice(valid_actions)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"\n🏁 게임 종료!")
    print(f"총 점수: {info['total_score']}")
    print(f"총 스텝: {info['steps']}")
    print(f"남은 사과: {info['remaining']}")
    print(f"총 보상: {total_reward}")
