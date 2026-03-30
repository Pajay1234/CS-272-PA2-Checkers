import mycheckersenv
import myagent
import argparse
import os
import numpy as np

NUM_EPISODES = 1_000
WEIGHTS_FILE = "agent_weights.pth"
LOG_INTERVAL = 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()

    load_weights=args.load

    env = mycheckersenv.CheckersEnv()
    
    # We use one agent for both players (self-play)
    agent = myagent.ACAgent(gamma=0.99, lr=0.001)
    
    # Load weights if the flag is set
    if load_weights and os.path.exists(WEIGHTS_FILE):
        agent.load(WEIGHTS_FILE)
        print(f"Loaded existing weights from '{WEIGHTS_FILE}'.")
    else:
        print("Starting training from scratch")

    # Running stats 
    stats = {"p0_wins": 0, "p1_wins": 0, "draws": 0}

    # Cumulative reward across all episodes
    cumulative_rewards = {"player_0": 0.0, "player_1": 0.0}

    for episode in range(NUM_EPISODES):
        env.reset()
        
        episode_loss = 0.0
        update_steps = 0
        
        transitions = {"player_0": None, "player_1": None}
        I_factor = {"player_0": 1.0, "player_1": 1.0}

        # track final rewards inside the loop — env.rewards is cleared after all agents are terminated, so we can't read it after the current agent_iter ends.
        final_rewards = {"player_0": 0.0, "player_1": 0.0}

        for agent_name in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            done = termination or truncation

            if done or reward != 0:
                final_rewards[agent_name] = reward

            # update
            if transitions[agent_name] is not None:
                prev_obs, prev_action = transitions[agent_name]
                step_loss = agent.update(
                    prev_obs, prev_action, reward, obs, done,
                    I_factor=I_factor[agent_name]
                )
                episode_loss += step_loss
                update_steps += 1
                I_factor[agent_name] *= agent.gamma

            # action selection
            if done:
                action = None
            else:
                action, _, _ = agent.get_action(obs)
                transitions[agent_name] = (obs, action)
                
            env.step(action)


        # collect metrics per epsiode
        avg_loss = episode_loss / update_steps if update_steps > 0 else 0.0

        r0 = final_rewards["player_0"]
        r1 = final_rewards["player_1"]
        if r0 > r1:
            winner = "player_0"
            stats["p0_wins"] += 1
        elif r1 > r0:
            winner = "player_1"
            stats["p1_wins"] += 1
        else:
            winner = "draw"
            stats["draws"] += 1

        p0_pieces = int(np.sum(np.sign(env.board) == 1))
        p1_pieces = int(np.sum(np.sign(env.board) == -1))
        piece_diff = p0_pieces - p1_pieces   

        print(
            f"Ep {episode + 1}/{NUM_EPISODES} | "
            f"Length: {env.num_moves} moves | "
            f"Winner: {winner} | "
            f"Pieces P0={p0_pieces} P1={p1_pieces} (diff {piece_diff:+d}) | "
            f"Avg Loss: {avg_loss:.4f}"
        )

        # Accumulate rewards
        cumulative_rewards["player_0"] += r0
        cumulative_rewards["player_1"] += r1

        if (episode + 1) % LOG_INTERVAL == 0:
            total = stats["p0_wins"] + stats["p1_wins"] + stats["draws"]
            print(
                f"  ── Last {LOG_INTERVAL} episodes: "
                f"P0 wins={stats['p0_wins']} ({100*stats['p0_wins']//total}%) | "
                f"P1 wins={stats['p1_wins']} ({100*stats['p1_wins']//total}%) | "
                f"Draws={stats['draws']} ({100*stats['draws']//total}%)"
            )
            stats = {"p0_wins": 0, "p1_wins": 0, "draws": 0}
        

    agent.save(WEIGHTS_FILE)
    print(f"\nModel weights saved to '{WEIGHTS_FILE}'.")

    print(f"Cumulative Rewards:")
    print(f"player_0 : {cumulative_rewards['player_0']:+.1f}")
    print(f"player_1 : {cumulative_rewards['player_1']:+.1f}")

    print("finished")