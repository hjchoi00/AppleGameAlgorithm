"""
apple_env.py - 사과게임 강화학습 환경 (Gymnasium 호환)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import random
import torch
import torch.nn as nn
from main import find_candidates_fast, apply_move_fast, read_matrix


# ============================================================
# 후보 점수화 네트워크 (Candidate Scorer)
# ============================================================

class CandidateScorer(nn.Module):
    """
    후보 점수화 네트워크
    
    각 후보의 특징을 입력받아 점수(scalar)를 출력.
    이 점수로 Top-K 후보를 선별.
    """
    
    def __init__(self, input_dim=6, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, candidate_features):
        """
        Args:
            candidate_features: (n_candidates, feature_dim) 또는 (batch, n_candidates, feature_dim)
        Returns:
            scores: (n_candidates,) 또는 (batch, n_candidates) - 각 후보의 점수
        """
        return self.net(candidate_features).squeeze(-1)
    
    def score_candidates(self, candidates, board, board_height, board_width):
        """
        후보 리스트를 받아서 점수를 계산
        
        Args:
            candidates: list of (r1, c1, r2, c2, cells, area)
            board: 현재 보드 상태 (다음 상태 후보 수 계산용)
            board_height, board_width: 보드 크기 (정규화용)
        Returns:
            scores: numpy array of shape (n_candidates,)
        """
        if not candidates:
            return np.array([])
        
        # 특징 추출 (보드 필요)
        features = self._extract_features(candidates, board, board_height, board_width)
        features_tensor = torch.FloatTensor(features)
        
        # 점수 계산
        with torch.no_grad():
            scores = self.forward(features_tensor).numpy()
        
        return scores
    
    def _extract_features(self, candidates, board, board_height, board_width):
        """후보들의 특징 추출 (6개 특징)"""
        n = len(candidates)
        features = np.zeros((n, 6), dtype=np.float32)
        
        max_cells = 10
        
        # 먼저 모든 후보의 다음 상태 후보 수 계산
        next_counts = []
        for cand in candidates:
            r1, c1, r2, c2, cells, area = cand
            temp_board = board.copy()
            apply_move_fast(temp_board, r1, c1, r2, c2)
            next_candidates = find_candidates_fast(temp_board)
            next_counts.append(len(list(next_candidates)))
        
        # 최대값으로 정규화 (0 방지)
        max_next_candidates = max(next_counts) if next_counts else 1
        max_next_candidates = max(max_next_candidates, 1)
        
        for i, cand in enumerate(candidates):
            r1, c1, r2, c2, cells, area = cand
            
            features[i] = [
                cells / max_cells,                                    # 0: cells (정규화)
                next_counts[i] / max_next_candidates,                 # 1: 다음 상태 후보 수 (상대 정규화)
                r1 / (board_height - 1) if board_height > 1 else 0,   # 2: r1 (정규화)
                c1 / (board_width - 1) if board_width > 1 else 0,     # 3: c1 (정규화)
                r2 / (board_height - 1) if board_height > 1 else 0,   # 4: r2 (정규화)
                c2 / (board_width - 1) if board_width > 1 else 0,     # 5: c2 (정규화)
            ]
        
        return features


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
        
        # 실제 보드 파일에서만 선택 (랜덤 보드 생성 제거)
        board_path = random.choice(self.board_files)
        self.board = read_matrix(board_path).astype(np.int32)
        
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
    
    # 후보 특징 개수: cells, next_candidates_ratio, r1, c1, r2, c2 (정규화)
    N_FEATURES = 6
    
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
        
        # 먼저 모든 top 후보의 다음 상태 후보 수 계산
        next_counts = []
        for cand in self.top_candidates:
            r1, c1, r2, c2, cells, area = cand
            temp_board = self.board.copy()
            apply_move_fast(temp_board, r1, c1, r2, c2)
            next_candidates = find_candidates_fast(temp_board)
            next_counts.append(len(list(next_candidates)))
        
        # 최대값으로 정규화 (0 방지)
        max_next_candidates = max(next_counts) if next_counts else 1
        max_next_candidates = max(max_next_candidates, 1)
        
        for i, cand in enumerate(self.top_candidates):
            r1, c1, r2, c2, cells, area = cand
            
            # 정규화된 특징 (6개)
            features[i] = [
                cells / max_cells,                                     # 0: cells (정규화)
                next_counts[i] / max_next_candidates,                  # 1: 다음 상태 후보 수 (상대 정규화)
                r1 / (self.board_height - 1),                          # 2: r1 (정규화)
                c1 / (self.board_width - 1),                           # 3: c1 (정규화)
                r2 / (self.board_height - 1),                          # 4: r2 (정규화)
                c2 / (self.board_width - 1),                           # 5: c2 (정규화)
            ]
        
        return features
        
    def reset(self, seed=None, options=None):
        """새 에피소드 시작"""
        super().reset(seed=seed)
        
        # 실제 보드 파일에서만 선택 (랜덤 보드 생성 제거)
        board_path = random.choice(self.board_files)
        self.board = read_matrix(board_path).astype(np.int32)
        
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


# ============================================================
# 학습된 Scorer로 Top-K 선별하는 환경
# ============================================================

class AppleGameEnvLearnedTopK(AppleGameEnvTopK):
    """
    학습된 CandidateScorer로 Top-K를 선별하는 환경
    
    기존 규칙 기반(_select_top_k) 대신 신경망이 후보 점수를 매겨서 Top-K 선별.
    Scorer는 환경과 함께 학습되거나, 사전 학습된 것을 로드해서 사용.
    """
    
    def __init__(self, board_dir="board_mat", top_k=20, render_mode=None,
                 scorer=None, train_scorer=True):
        """
        Args:
            scorer: CandidateScorer 인스턴스 (None이면 새로 생성)
            train_scorer: True면 학습 모드, False면 추론 모드
        """
        super().__init__(board_dir, top_k, render_mode)
        
        # Scorer 네트워크
        self.scorer = scorer if scorer is not None else CandidateScorer()
        self.train_scorer = train_scorer
        
        # Flat observation space로 변경
        board_size = self.board_height * self.board_width
        candidate_size = top_k * self.N_FEATURES
        mask_size = top_k
        total_size = board_size + candidate_size + mask_size
        
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(total_size,),
            dtype=np.float32
        )
        
        # Scorer 학습용 버퍼
        self.scorer_buffer = []  # (features, chosen_idx, reward) 튜플 저장
    
    def _select_top_k(self, candidates):
        """학습된 Scorer로 상위 K개 후보 선택"""
        if not candidates:
            return []
        
        # Scorer로 점수 계산 (board 전달)
        scores = self.scorer.score_candidates(
            candidates, self.board, self.board_height, self.board_width
        )
        
        # 점수 높은 순으로 정렬하여 Top-K 선택
        # (학습 초기에는 랜덤에 가깝고, 학습이 진행되면서 좋은 후보를 선별)
        sorted_indices = np.argsort(-scores)  # 내림차순
        top_indices = sorted_indices[:self.top_k]
        
        return [candidates[i] for i in top_indices]
    
    def _get_obs(self):
        """관측값 반환 (1D 벡터)"""
        board_flat = (self.board / 9.0).astype(np.float32).flatten()
        candidates_flat = self._get_candidate_features().flatten()
        mask_flat = self.get_action_mask().astype(np.float32)
        
        return np.concatenate([board_flat, candidates_flat, mask_flat])
    
    def step(self, action):
        """액션 실행 + Scorer 학습 데이터 수집"""
        # 현재 선택된 후보의 특징 저장 (학습용)
        if self.train_scorer and self.top_candidates and action < len(self.top_candidates):
            chosen_cand = self.top_candidates[action]
            features = self.scorer._extract_features(
                [chosen_cand], self.board_height, self.board_width
            )[0]
        else:
            features = None
        
        # 기존 step 실행
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Scorer 학습 데이터 저장
        if features is not None and self.train_scorer:
            self.scorer_buffer.append((features, reward, terminated))
        
        return obs, reward, terminated, truncated, info
    
    def get_scorer_training_data(self):
        """Scorer 학습용 데이터 반환 및 버퍼 클리어"""
        data = self.scorer_buffer.copy()
        self.scorer_buffer = []
        return data
    
    def set_scorer(self, scorer):
        """외부에서 학습된 Scorer 설정"""
        self.scorer = scorer


class AppleGameEnvLearnedTopKFlat(AppleGameEnvLearnedTopK):
    """
    학습된 Scorer + Flat observation (MLP Policy 호환)
    """
    pass  # 이미 부모 클래스에서 Flat obs 구현됨


# ============================================================
# 모든 후보를 제공하는 환경 (Scorer End-to-End 학습용)
# ============================================================

class AppleGameEnvAllCandidates(gym.Env):
    """
    모든 후보를 제공하는 환경 (최대 MAX_CANDIDATES개)
    
    Policy가 직접 모든 후보 중에서 선택.
    이 환경에서 학습한 policy의 action probability를
    후보 점수(Scorer)로 사용할 수 있음.
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    N_FEATURES = 9
    
    def __init__(self, board_dir="board_mat", max_candidates=100, render_mode=None):
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
        
        # 보드 크기 확인
        sample_board = read_matrix(self.board_files[0])
        self.board_height, self.board_width = sample_board.shape
        
        # Observation space: 보드 + 모든 후보 특징 + 마스크
        board_size = self.board_height * self.board_width
        candidate_size = max_candidates * self.N_FEATURES
        mask_size = max_candidates
        total_size = board_size + candidate_size + mask_size
        
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(total_size,),
            dtype=np.float32
        )
        
        # Action space: 모든 후보 중 선택
        self.action_space = spaces.Discrete(max_candidates)
        
        # 현재 상태
        self.board = None
        self.candidates = []
        self.total_score = 0
        self.steps = 0
        self.prev_num_candidates = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 실제 보드 파일에서만 선택 (랜덤 보드 생성 제거)
        board_path = random.choice(self.board_files)
        self.board = read_matrix(board_path).astype(np.int32)
        
        self.candidates = list(find_candidates_fast(self.board))
        # 후보가 너무 많으면 랜덤 샘플링
        if len(self.candidates) > self.max_candidates:
            self.candidates = random.sample(self.candidates, self.max_candidates)
        
        self.total_score = 0
        self.steps = 0
        self.prev_num_candidates = len(self.candidates)
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        if not self.candidates:
            return self._get_obs(), 0, True, False, self._get_info()
        
        action = action % len(self.candidates)
        
        r1, c1, r2, c2, cells, area = self.candidates[action]
        apply_move_fast(self.board, r1, c1, r2, c2)
        
        self.total_score += cells
        self.steps += 1
        
        # 새로운 후보 탐색
        self.candidates = list(find_candidates_fast(self.board))
        if len(self.candidates) > self.max_candidates:
            self.candidates = random.sample(self.candidates, self.max_candidates)
        
        curr_remaining = np.sum(self.board > 0)
        
        # 단순 보상
        reward = cells
        
        terminated = len(self.candidates) == 0
        truncated = False
        
        if terminated:
            reward -= curr_remaining
        
        self.prev_num_candidates = len(self.candidates)
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _get_candidate_features(self):
        """모든 후보들의 특징 행렬 반환"""
        features = np.zeros((self.max_candidates, self.N_FEATURES), dtype=np.float32)
        
        max_cells = 10
        max_area = self.board_height * self.board_width
        
        for i, cand in enumerate(self.candidates):
            r1, c1, r2, c2, cells, area = cand
            height = r2 - r1 + 1
            width = c2 - c1 + 1
            zeros = area - cells
            
            features[i] = [
                cells / max_cells,
                area / max_area,
                zeros / max_area,
                height / self.board_height,
                width / self.board_width,
                r1 / (self.board_height - 1) if self.board_height > 1 else 0,
                c1 / (self.board_width - 1) if self.board_width > 1 else 0,
                r2 / (self.board_height - 1) if self.board_height > 1 else 0,
                c2 / (self.board_width - 1) if self.board_width > 1 else 0,
            ]
        
        return features
    
    def _get_obs(self):
        board_flat = (self.board / 9.0).astype(np.float32).flatten()
        candidates_flat = self._get_candidate_features().flatten()
        mask_flat = self.get_action_mask().astype(np.float32)
        
        return np.concatenate([board_flat, candidates_flat, mask_flat])
    
    def _get_info(self):
        return {
            "total_score": self.total_score,
            "steps": self.steps,
            "candidates": len(self.candidates),
            "remaining": np.sum(self.board > 0)
        }
    
    def get_action_mask(self):
        mask = np.zeros(self.max_candidates, dtype=bool)
        n_valid = len(self.candidates)
        mask[:n_valid] = True
        return mask
    
    def action_masks(self):
        return self.get_action_mask()
    
    def render(self):
        if self.render_mode == "human" or self.render_mode == "ansi":
            print(f"\n=== Step {self.steps} | Score: {self.total_score} | Candidates: {len(self.candidates)} ===")
            print(self.board)
    
    def close(self):
        pass


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
