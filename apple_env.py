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
        
        # 실제 보드 파일에서만 선택
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
            remaining = np.sum(self.board > 0)l,p,LookupError
            
            reward -= (remaining ** 1.2)
        
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
