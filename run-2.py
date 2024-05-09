# Import the W&B Python Library and log into W&B
import wandb
from src import DoubleDQNAgent, ModifiedGinRummyEnv
from rlcard.agents import RandomAgent
import rlcard
import argparse
from rlcard.utils import set_seed
from pathlib import Path
from gin_rummy_rule_models import GinRummyNoviceRuleAgent

wandb.login()

def train(config, run):
  #set_seed(42)
#   env_train = rlcard.make("gin-rummy", config={
#             'seed': 42,
#            'going_out_deadwood_count': 100, # always can knock
#        })
#   env_eval = rlcard.make("gin-rummy", config={
#            #'seed': 42,
#        })
  # create some simple observation space environments...
  env_train = ModifiedGinRummyEnv({
  	    'allow_step_back': False,
            'seed': None,
            'going_out_deadwood_count': 100, # always can knock
       })
  env_eval = ModifiedGinRummyEnv({'allow_step_back': False, 'seed': None})
  # declare agents
  agents = [DoubleDQNAgent(3*52, 110, replay_memory_size=config.replay_size, epsilon=config.epsilon, epsilon_stop=config.epsilon_stop, learning_rate=config.learning_rate, device='cuda', hidden_size=config.hidden_size), GinRummyNoviceRuleAgent()]
  env_train.set_agents(agents)
  env_eval.set_agents(agents)
  total = 0
  
  Path(f"./checkpoints/homegrown/{run.sweep_id}/{run.name}").mkdir(parents=True, exist_ok=True)
  
  while total < config.steps:
    # generate trajectory with environment
    trajectories, payoffs = env_train.run(is_training=True)
    # call this rlcard function that transforms the trajectories into a resonable format
    trajectories = rlcard.utils.reorganize(trajectories, payoffs)
    for ts in trajectories[0]:
      # 'feed' the model a single transition
      agents[0].feed(ts)  
      # evaluate performance ever 50k steps, against random agents, in this case
      if total % 50000 == 0:
        # this function can be used to run a 'tournament' between agents
        payoffs = rlcard.utils.tournament(env_train, 50)
        payoffs_ = rlcard.utils.tournament(env_eval, 50)
        #print(f"Algorithm's average payoff: {payoffs[0]}, adversary's: {payoffs[1]}")
        # log the data
        wandb.log({"tournament-train": payoffs[0], "tournament-eval": payoffs_[0]})
        # save a checkpoint, if necessary. Specify cutoff so we don't run out of disk space.
        if payoffs_[0] >= -0.3:
          agents[0].save_checkpoint(f"./checkpoints/homegrown/{run.sweep_id}/{run.name}", f"dqn_{total}.pt")
      total += 1
  # save one final checkpoint for good measure.
  agents[0].save_checkpoint(f"./checkpoints/homegrown/{run.sweep_id}/{run.name}", f"dqn_{total}.pt")

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
