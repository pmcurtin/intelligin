# Import the W&B Python Library and log into W&B
import wandb
from src import DQNAgent, DoubleDQNAgent, SeqDoubleDQNAgent, ModifiedGinRummyEnv
from rlcard.agents import RandomAgent
import rlcard
import argparse
from rlcard.utils import set_seed
from pathlib import Path

wandb.login()

def run(env, is_training=False):
    trajectories = [[] for _ in range(env.num_players)]
    state, player_id = env.reset()

    # reset agent.h for seq agent
    for agent in env.agents:
        if isinstance(agent, SeqDoubleDQNAgent):
          agent.reset()

    # Loop to play the game
    trajectories[player_id].append(state)
    while not env.is_over():
        # Agent plays
        if not is_training:
            action, _ = env.agents[player_id].eval_step(state)
        else:
            action = env.agents[player_id].step(state)

        # Environment steps
        next_state, next_player_id = env.step(action, env.agents[player_id].use_raw)
        # Save action
        trajectories[player_id].append(action)

        # Set the state and player
        state = next_state
        player_id = next_player_id

        # Save state.
        if not env.game.is_over():
            trajectories[player_id].append(state)

    # Add a final state to all the players
    for player_id in range(env.num_players):
        state = env.get_state(player_id)
        trajectories[player_id].append(state)

    # Payoffs
    payoffs = env.get_payoffs()

    return trajectories, payoffs

def tournament(env, num):
    ''' Evaluate he performance of the agents in the environment

    Args:
        env (Env class): The environment to be evaluated.
        num (int): The number of games to play.

    Returns:
        A list of avrage payoffs for each player
    '''
    payoffs = [0 for _ in range(env.num_players)]
    counter = 0
    while counter < num:
        _, _payoffs = run(env, is_training=False)
        if isinstance(_payoffs, list):
            for _p in _payoffs:
                for i, _ in enumerate(payoffs):
                    payoffs[i] += _p[i]
                counter += 1
        else:
            for i, _ in enumerate(payoffs):
                payoffs[i] += _payoffs[i]
            counter += 1
    for i, _ in enumerate(payoffs):
        payoffs[i] /= counter
    return payoffs

def train(config, wandb_run):
    #set_seed(42)
    #env_train = rlcard.make("gin-rummy", config={
    #        #'seed': 42,
    #        'going_out_deadwood_count': 100, # always can knock
    #    })
    #env_eval = rlcard.make("gin-rummy", config={
    #        #'seed': 42,
    #    })
    env_train = ModifiedGinRummyEnv({
  	    'allow_step_back': False,
            'seed': None,
            'going_out_deadwood_count': 100, # always can knock
    })
    env_eval = ModifiedGinRummyEnv({'allow_step_back': False, 'seed': None})
    # DoubleDQNAgent(5*52, 110, device='cuda', batch_size=32, replay_memory_size=config.replay_size, learning_rate=config.learning_rate, num_hidden=config.hidden, hidden_size=config.hidden_size, epsilon_stop=config.epsilon_stop, epsilon=config.epsilon)
    # define sequence agent
    agents = [SeqDoubleDQNAgent(3*52, 110, replay_memory_size=config.replay_size, epsilon=config.epsilon, epsilon_stop=config.epsilon_stop, learning_rate=config.learning_rate, device='cuda', hidden_size=config.hidden_size), RandomAgent(num_actions=110)]
    env_train.set_agents(agents)
    env_eval.set_agents(agents)
    total = 0

    Path(f"./checkpoints/homegrown/{wandb_run.sweep_id}/{wandb_run.name}").mkdir(parents=True, exist_ok=True)

    l = 0
    f = True

    while total < config.steps:
        # generate trajectory
        trajectories, payoffs = run(env_train, is_training=True)
        trajectories = rlcard.utils.reorganize(trajectories, payoffs)
        # we changed the feed function for our SeqDoubleDQNAgent to injest an entire trajectory
        agents[0].feed(trajectories[0])

        total += len(trajectories[0])

        # some hacks to detect when we execute 50k steps
        if total // 50000 != l or f:
            # evaluate the model. Use our tournament function so it properly resets the hidden state
            payoffs = tournament(env_train, 50)
            payoffs_ = tournament(env_eval, 50)
            # log
            wandb.log({"tournament-train": payoffs[0], "tournament-eval": payoffs_[0]})
            # save checkpoint, if good
            if payoffs[0] >= 0.1:
                agents[0].save_checkpoint(f"./checkpoints/homegrown/{run.sweep_id}/{run.name}", f"dqn_{l+1}.pt")
            # hacks
            if not f: l += 1
            f = False

def main(args):
    run = wandb.init(project="gin_rummy", config=args)
    train(wandb.config, run)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-r", "--replay_size", type=int, default=20000)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5)
    parser.add_argument("-H", "--hidden", type=int, default=2)
    parser.add_argument("-hs", "--hidden_size", type=int, default=64)
    parser.add_argument("-es", "--epsilon_stop", type=int, default=100000)
    parser.add_argument("-e", "--epsilon", type=float, default=0.1)
    parser.add_argument("-s", "--steps", type=int, default=1000000)

    args = parser.parse_args()
    main(args)
