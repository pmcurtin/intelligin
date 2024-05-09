import wandb
from src import DQNAgent
from rlcard.agents import RandomAgent
import rlcard
import argparse
from rlcard.utils import set_seed
from pathlib import Path

wandb.login()

def train(config, run):
  #set_seed(42)
  env_train = rlcard.make("gin-rummy", config={
            #'seed': 42,
            'going_out_deadwood_count': 100, # always can knock
        })
  env_eval = rlcard.make("gin-rummy", config={
            #'seed': 42,
        })
  agents = [DQNAgent(5*52, 110, device='cuda', batch_size=32, replay_memory_size=config.replay_size, learning_rate=config.learning_rate, num_hidden=config.hidden, hidden_size=config.hidden_size, epsilon_stop=config.epsilon_stop, epsilon=config.epsilon), RandomAgent(num_actions=110)]
  env_train.set_agents(agents)
  env_eval.set_agents(agents)
  total = 0
  
  Path(f"./checkpoints/{run.sweep_id}/{run.name}").mkdir(parents=True, exist_ok=True)
  
  while total < config.steps:
    # generate trajectory
    trajectories, payoffs = env_train.run(is_training=True)
    trajectories = rlcard.utils.reorganize(trajectories, payoffs)
    for ts in trajectories[0]:
      # feed agent
      agents[0].feed(ts)  
      if total % 50000 == 0:
        # evaluate on envs
        payoffs = rlcard.utils.tournament(env_train, 50)
        payoffs_ = rlcard.utils.tournament(env_eval, 50)
        #print(f"Algorithm's average payoff: {payoffs[0]}, adversary's: {payoffs[1]}")
        # log
        wandb.log({"tournament-train": payoffs[0], "tournament-eval": payoffs_[0]})
        # checkpoint if good enough
        if payoffs[0] >= 0.1:
          agents[0].save_checkpoint(f"./checkpoints/{run.sweep_id}/{run.name}", f"dqn_{total}.pt")
      total += 1

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
