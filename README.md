# CS-272 PA2: Checkers 6x6

| | |
|---|---|
| **Import** | from mycheckersenv import CheckersEnv |
| **Actions** | Discrete |
| **Parallel API** | No |
| **Manual Control** | No |
| **Agents** | agents= ['player_0', 'player_1'] |
| **Agents** | 2 |
| **Action Shape** | (1,) |
| **Action Values** | Discrete(1296) |
| **Observation Shape** | (6, 6) |
| **Observation Values** | [-2, -1, 0, 1, 2] |

Checkers with a 6x6 board. Players take turns moving their pieces but player_0 always starts first. Capturing an opponent's piece is mandatory when available, and multi-jump chains must be completed before the turn passes. A piece that reaches the opponent's back row is promoted to a king, which may move in all four diagonal directions. The game ends when a player captures all opponent pieces, blocks all opponent moves, or the move limit is reached.

## Observation Space

The observation is a dictionary containing an observation element, which is a 6x6 array representing the game board, and an action_mask which holds the legal moves, described below. The board is always shown such that the current players's pieces are positive and the opponent's pieces are negative. For the visualization, we see observe player_0's perspective. 

| Value | Meaning |
|---|---|
| 0 | Empty square |
| +1 | current agent's man |
| +2 | current agent's king |
| -1 | other agent's man |
| -2 | other agent's king |

### Legal Actions Mask

The legal moves available to the current agent are found in the action_mask of the observation dictionary. The action_mask is a binary vector of length 1296 where each index represents whether the action is legal (1) or not (0). If any jump is available, only jump actions will be unmasked (mandatory capture rule).


## Action Space

The action space is Discrete(1296), encoding all possible transitions on the 6x6 board, even blatantly illegal ones. An action is decoded as:

start_square = action // 36        # encodes (row * 6 + col) of the source
end_square   = action % 36         # encodes (row * 6 + col) of the destination


## Rewards

If an agent captures all opponent pieces or leaves the opponent with no legal moves, they are awarded +1 and the opponent is awarded -1. If the move limit (100 moves) is reached, both agents receive 0.

| Condition | Winner | Loser |
|---|---|---|
| Opponent has no pieces | +1 | -1`|
| Opponent has no legal moves | +1 | -1 |
| Move limit reached (draw) | 0 | 0 |


