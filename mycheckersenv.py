import functools
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector

class CheckersEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "checkers_6x6_v0"}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.possible_agents = ["player_0", "player_1"]
        self.agents = self.possible_agents[:]
        
        self.action_spaces = {agent: spaces.Discrete(1296) for agent in self.possible_agents}
        self.observation_spaces = {
            agent: spaces.Dict({
                "observation": spaces.Box(low=-2, high=2, shape=(6, 6), dtype=np.int8),
                "action_mask": spaces.Box(low=0, high=1, shape=(1296,), dtype=np.int8)
            }) for agent in self.possible_agents
        }
        
        self.board = np.zeros((6, 6), dtype=np.int8)
        self.num_moves = 0
        self.max_moves = 100

        # Tracks which piece is mid-chain-jump
        self.multi_jump_piece = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.multi_jump_piece = None
        
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.num_moves = 0

        self.board = np.zeros((6, 6), dtype=np.int8)
        # Player 1 (-1) on top 2 rows, Player 0 (1) on bottom 2 rows
        for r in range(2):
            for c in range(6):
                if (r + c) % 2 == 1:
                    self.board[r, c] = -1
        for r in range(4, 6):
            for c in range(6):
                if (r + c) % 2 == 1:
                    self.board[r, c] = 1

        # setup log file with every new game
        with open("board_log.txt", "w") as f:
            f.write("--- Game Start ---\n")

    def observe(self, agent):
        # flip the pieces values for player 1
        board = self.board.copy() # deep copy 
        if agent == "player_1":
            board = -board

        # build the action mask 
        mask = np.zeros(1296, dtype=np.int8)
        legal_moves = self._get_legal_moves(agent)
        for move in legal_moves:
            s_r, s_c, e_r, e_c = move
            action_idx = (s_r * 6 + s_c) * 36 + (e_r * 6 + e_c)
            mask[action_idx] = 1

        # Flip the coordinates in the action mask for player 1
        if agent == "player_1":
            flipped_mask = np.zeros(1296, dtype=np.int8)
            for idx in np.where(mask == 1)[0]:
                flipped_mask[self._flip_action_index(idx)] = 1
            mask = flipped_mask

        return {"observation": board, "action_mask": mask}

    def _get_legal_moves(self, agent):
        """
        Calculates legal moves, enforcing mandatory jumps and multi-jump chains.
        If self.multi_jump_piece is set, only jumps from that specific piece are returned.
        """
        is_p0 = (agent == "player_0")
        player_sign = 1 if is_p0 else -1

        # multi-jump: manadatory chain captures from the same piece
        if self.multi_jump_piece is not None:
            pr, pc = self.multi_jump_piece
            return self._get_jumps_from_piece(pr, pc, player_sign, is_p0)

        #  normal move
        moves = []
        jumps = []
        
        for r in range(6):
            for c in range(6):
                piece = self.board[r, c]
                if piece == 0 or np.sign(piece) != player_sign:
                    continue
                
                is_king = abs(piece) == 2
                directions = self._get_directions(is_p0, is_king)
                    
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    # Normal move
                    if 0 <= nr < 6 and 0 <= nc < 6 and self.board[nr, nc] == 0:
                        moves.append((r, c, nr, nc))
                    # Jump
                    jr, jc = r + 2 * dr, c + 2 * dc
                    if 0 <= jr < 6 and 0 <= jc < 6:
                        mid_piece = self.board[nr, nc]
                        if mid_piece != 0 and np.sign(mid_piece) != player_sign and self.board[jr, jc] == 0:
                            jumps.append((r, c, jr, jc))

        # mandatory jump
        if jumps:
            return jumps
        return moves

    def _get_jumps_from_piece(self, r, c, player_sign, is_p0):
        """Returns only jump moves available from a single piece at (r, c)."""
        piece = self.board[r, c]
        is_king = abs(piece) == 2
        directions = self._get_directions(is_p0, is_king)
        jumps = []

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            jr, jc = r + 2 * dr, c + 2 * dc
            if 0 <= jr < 6 and 0 <= jc < 6:
                mid_piece = self.board[nr, nc]
                if mid_piece != 0 and np.sign(mid_piece) != player_sign and self.board[jr, jc] == 0:
                    jumps.append((r, c, jr, jc))
        return jumps

    def _get_directions(self, is_p0, is_king):
        """Returns possible movements for a piece."""
        directions = []
        if is_p0 or is_king:
            directions.append((-1, -1))   
            directions.append((-1, 1))   
        if not is_p0 or is_king:
            directions.append((1, -1))   
            directions.append((1, 1))        
        return directions


    def _flip_action_index(self, action_idx):
        """
        Converts an index for an action between real-board and flipped-board coordinates.
        Only rows are mirrored (downward vs. upward movement)
        """
        start_sq = action_idx // 36
        end_sq   = action_idx % 36
        s_r, s_c = start_sq // 6, start_sq % 6
        e_r, e_c = end_sq   // 6, end_sq   % 6
        flipped_start = (5 - s_r) * 6 + s_c
        flipped_end   = (5 - e_r) * 6 + e_c
        return flipped_start * 36 + flipped_end

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return
            
        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0

        if agent == "player_1":
            action = self._flip_action_index(action)

        # Decode action to real-board coordinates
        start_sq = action // 36
        end_sq = action % 36
        s_r, s_c = start_sq // 6, start_sq % 6
        e_r, e_c = end_sq // 6, end_sq % 6
        
        # Make move
        piece = self.board[s_r, s_c]
        self.board[s_r, s_c] = 0
        self.board[e_r, e_c] = piece
        
        is_jump = abs(s_r - e_r) == 2

        # handle capture
        if is_jump:
            mid_r, mid_c = (s_r + e_r) // 2, (s_c + e_c) // 2
            self.board[mid_r, mid_c] = 0
            
        # handle king promotion
        promoted = False
        if agent == "player_0" and e_r == 0 and piece == 1:
            self.board[e_r, e_c] = 2
            promoted = True
        elif agent == "player_1" and e_r == 5 and piece == -1:
            self.board[e_r, e_c] = -2
            promoted = True

        # handle logging 
        jump_label = " (JUMP)" if is_jump else ""
        with open("board_log.txt", "a") as f:
            f.write(f"\n{agent} moved ({s_r},{s_c}) to ({e_r},{e_c}){jump_label}\n")
            if promoted:
                f.write(f"  >>> man promoted to king at ({e_r},{e_c})\n")
            f.write(np.array2string(self.board) + "\n")
            
        self.num_moves += 1

        # check for multi-jump possibility: if the move was a jump and the piece wasn't just promoted, check if it can jump again
        if is_jump and not promoted:
            is_p0 = (agent == "player_0")
            player_sign = 1 if is_p0 else -1
            further_jumps = self._get_jumps_from_piece(e_r, e_c, player_sign, is_p0)
            if further_jumps:
                # Keep the same player active 
                self.multi_jump_piece = (e_r, e_c)
                self.rewards = {a: 0 for a in self.agents}
                self._accumulate_rewards()
                return
        # clean multi-jump state and switch player
        self.multi_jump_piece = None
        next_agent = self._agent_selector.next()
        
        # Game control
        next_legal_moves = self._get_legal_moves(next_agent)
        next_has_pieces = np.any(np.sign(self.board) == (1 if next_agent == "player_0" else -1))
        
        if not next_has_pieces or len(next_legal_moves) == 0:
            self.terminations = {a: True for a in self.agents}
            self.rewards[agent] = 1
            self.rewards[next_agent] = -1
        elif self.num_moves >= self.max_moves:
            self.truncations = {a: True for a in self.agents}
            self.rewards = {a: 0 for a in self.agents}
        else:
            self.rewards = {a: 0 for a in self.agents}
            
        self.agent_selection = next_agent
        self._accumulate_rewards()