# Import the W&B Python Library and log into W&B
import wandb, rlcard, argparse
from src import DoubleDQNAgent, ModifiedGinRummyEnv
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed
from pathlib import Path
#from gin_rummy_rule_models import GinRummyNoviceRuleAgent
from copy import deepcopy

wandb.login()

# 1: Define objective/training function
def train(config, run):
  #set_seed(42)
  #env_train = rlcard.make("gin-rummy", config={
            #'seed': 42,
  #          'going_out_deadwood_count': 100, # always can knock
  #      })
  #env_eval = rlcard.make("gin-rummy", config={
  #          #'seed': 42,
  #      })
  env_train = ModifiedGinRummyEnv({
  	    'allow_step_back': False,
            'seed': None,
            'going_out_deadwood_count': 100, # always can knock
       })
  env_eval = ModifiedGinRummyEnv({'allow_step_back': False, 'seed': None})
  learned = DoubleDQNAgent(3*52, 110, replay_memory_size=config.replay_size, epsilon=config.epsilon, epsilon_stop=config.epsilon_stop, learning_rate=config.learning_rate, device='cuda', hidden_size=config.hidden_size)
  # here, adversary is just a deep copy of the learned agent
  adversary = deepcopy(learned)
  # a detail we had to add so that when adversary.step is called, it doesn't act epsilon-greedily.
  adversary.can_train = False
  agents = [learned, adversary]
  env_train.set_agents(agents)
  env_eval.set_agents(agents)
  # also make an agent pair with random agent for performance logging
  agents_rand = [learned, RandomAgent(num_actions=110)]
  #env_train.set_agents(agents)
  #env_eval.set_agents(agents)
  total = 0
  
  Path(f"./checkpoints/homegrown/{run.sweep_id}/{run.name}").mkdir(parents=True, exist_ok=True)
  
  while total < config.steps:
    # generate trajectories
    trajectories, payoffs = env_train.run(is_training=True)
    trajectories = rlcard.utils.reorganize(trajectories, payoffs)
    
    for t in trajectories[0]:
      # feed to agent, as usual.
      agents[0].feed(t)
      total += 1
      if total % 50000 == 0:
        # evaluate agent against the self-play adversary
        payoffs_train = rlcard.utils.tournament(env_train, 50)
        # also do it in the eval environment
        env_eval.set_agents(agents)
        payoffs_eval = rlcard.utils.tournament(env_eval, 50)
        # and finally, evaluate it with respect to the random agent
        env_eval.set_agents(agents_rand)
        payoffs_eval_rand = rlcard.utils.tournament(env_eval, 50)
        # log all these metrics
        wandb.log({"tournament-train": payoffs_train[0], "tournament-eval": payoffs_eval[0], "eval-rand": payoffs_eval_rand[0]})
        # checkpointing
        if payoffs_eval_rand[0] >= 0.1:
          learned.save_checkpoint(f"./checkpoints/homegrown/{run.sweep_id}/{run.name}", f"dqn_{total}.pt")
        # copy over the model to the adversary every 100k
        if total % 100000 == 0 and total != 0:
          adversary = deepcopy(learned)
          adversary.can_train = False
          agents = [learned, adversary]
          env_train.set_agents(agents)
          env_eval.set_agents(agents)
		          


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
