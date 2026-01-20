"""
apple_env.py - 사과게임 강화학습 환경 (Gymnasium 호환)

특징 벡터 기반 관측 공간:
- board: 10x17 보드 상태 (공간적 맥락)
- action_features: 각 후보의 특징 벡터 (max_candidates x 7)
  [r1, c1, r2, c2, cells, area, next_candidates_ratio]
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
from main import find_candidates_fast, apply_move_fast, read_matrix


class AppleGameEnv(gym.Env):
    """
    사과게임 강화학습 환경 (특징 벡터 기반)
    
    Observation (Dict):
        - board: 10x17 보드 상태 (0~1 정규화)
        - action_features: 후보별 특징 벡터 (max_candidates x 7)
    
    Action: 후보 리스트에서 인덱스 선택 (0 ~ max_candidates-1)
    Reward: 제거한 사과 개수 + 종료 시 페널티
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    # 특징 벡터 차원
    N_FEATURES = 7  # r1, c1, r2, c2, cells, area, next_candidates_ratio
    
    def __init__(self, board_dir="board_mat", max_candidates=256, render_mode=None, base_seed=None):
        super().__init__()
        
        self.board_dir = board_dir
        self.max_candidates = max_candidates
        self.render_mode = render_mode
        
        # 재현성을 위한 seed 관리
        self.base_seed = base_seed
        self.episode_idx = 0
        
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
        
        # Observation space: Dict (board + action_features)
        self.observation_space = spaces.Dict({
            # 1. 보드 상태 (공간적 맥락)
            "board": spaces.Box(
                low=0, high=1, 
                shape=(self.board_height, self.board_width), 
                dtype=np.float32
            ),
            # 2. 후보 액션들의 특징 벡터
            "action_features": spaces.Box(
                low=0, high=1,
                shape=(self.max_candidates, self.N_FEATURES),
                dtype=np.float32
            )
        })
        
        # Action space: 후보 인덱스 (최대 max_candidates개)
        self.action_space = spaces.Discrete(max_candidates)
        
        # 현재 상태
        self.board = None
        self.candidates = []
        self.total_score = 0
        self.steps = 0
        
        # 캐시: 각 후보의 다음 상태 후보 수 (step에서 재사용)
        self._next_candidates_cache = []
        
    def reset(self, seed=None, options=None):
        """새 에피소드 시작"""
        # seed가 없고 base_seed가 있으면 episode_idx 기반 seed 사용
        if seed is None and self.base_seed is not None:
            seed = self.base_seed + self.episode_idx
        
        super().reset(seed=seed)
        self.episode_idx += 1
        
        # 실제 보드 파일에서만 선택
        board_path = self.np_random.choice(self.board_files)
        self.board = read_matrix(board_path).astype(np.int32)
        
        self.candidates = list(find_candidates_fast(self.board))
        self.total_score = 0
        self.steps = 0
        
        # 후보별 다음 상태 후보 수 계산 (캐시)
        self._compute_next_candidates_cache()
        
        return self._get_obs(), self._get_info()
    
    def _compute_next_candidates_cache(self):
        """각 후보를 선택했을 때 다음 상태의 후보 수를 미리 계산"""
        self._next_candidates_cache = []
        for cand in self.candidates:
            r1, c1, r2, c2, cells, area = cand
            temp_board = self.board.copy()
            apply_move_fast(temp_board, r1, c1, r2, c2)
            next_count = len(list(find_candidates_fast(temp_board)))
            self._next_candidates_cache.append(next_count)
    
    def step(self, action):
        """액션 실행"""
        # 유효한 액션인지 확인
        if not self.candidates:
            # 게임 종료
            return self._get_obs(), float(self.total_score), True, False, self._get_info()
        
        # 액션 인덱스를 유효 범위로 클리핑
        action = action % len(self.candidates)
        
        # 선택한 후보 실행
        r1, c1, r2, c2, cells, area = self.candidates[action]
        apply_move_fast(self.board, r1, c1, r2, c2)
        
        # 보상 계산
        reward = 0.5 * cells  # 제거한 사과 개수
        self.total_score += cells
        self.steps += 1
        
        # 새로운 후보 탐색
        self.candidates = list(find_candidates_fast(self.board))
        
        # 종료 조건
        terminated = len(self.candidates) == 0
        truncated = False
        
        # 게임 종료 시 최종 보상
        if terminated:
            # remaining = np.sum(self.board > 0)
            reward = float(self.total_score)
        else:
            # 후보별 다음 상태 후보 수 계산 (캐시 업데이트)
            self._compute_next_candidates_cache()
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _get_obs(self):
        """
        관측값 반환 (Dict: board + action_features)
        
        action_features 각 열:
            0: r1 / board_height (시작 행)
            1: c1 / board_width (시작 열)
            2: r2 / board_height (끝 행)
            3: c2 / board_width (끝 열)
            4: cells / 10 (제거할 사과 개수)
            5: area / (board_height * board_width) (면적)
            6: next_candidates / max_next (다음 상태 후보 비율 - 핵심!)
        """
        # 1. 보드 정규화
        board_obs = (self.board / 9.0).astype(np.float32)
        
        # 2. 후보 특징 벡터 생성
        features = np.zeros((self.max_candidates, self.N_FEATURES), dtype=np.float32)
        
        if self.candidates:
            # 다음 상태 후보 수의 최대값 (정규화용)
            max_next = max(self._next_candidates_cache) if self._next_candidates_cache else 1
            max_next = max(max_next, 1)  # 0 방지
            
            max_area = self.board_height * self.board_width
            
            for i, cand in enumerate(self.candidates[:self.max_candidates]):
                r1, c1, r2, c2, cells, area = cand
                next_count = self._next_candidates_cache[i] if i < len(self._next_candidates_cache) else 0
                
                features[i] = [
                    r1 / (self.board_height - 1) if self.board_height > 1 else 0,  # 0: r1
                    c1 / (self.board_width - 1) if self.board_width > 1 else 0,    # 1: c1
                    r2 / (self.board_height - 1) if self.board_height > 1 else 0,  # 2: r2
                    c2 / (self.board_width - 1) if self.board_width > 1 else 0,    # 3: c2
                    cells / 10.0,                                                   # 4: cells
                    area / max_area,                                                # 5: area
                    next_count / max_next,                                          # 6: next_candidates_ratio ★핵심
                ]
        
        return {
            "board": board_obs,
            "action_features": features
        }
    
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
