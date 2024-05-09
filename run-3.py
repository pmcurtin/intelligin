# Import the W&B Python Library and log into W&B
import wandb, random, rlcard, argparse
from src import DoubleDQNAgent, ModifiedGinRummyEnv
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed
from pathlib import Path
from gin_rummy_rule_models import GinRummyNoviceRuleAgent

wandb.login()

def train(config, run):
  #set_seed(42)
  # create some rich observation space environments
  env_train = rlcard.make("gin-rummy", config={
           'going_out_deadwood_count': 100, # always can knock
       })
  env_eval = rlcard.make("gin-rummy", config={
           #'seed': 42,
       })
  # env_train = ModifiedGinRummyEnv({
  # 	    'allow_step_back': False,
  #           'seed': None,
  #           'going_out_deadwood_count': 100, # always can knock
  #      })
  # env_eval = ModifiedGinRummyEnv({'allow_step_back': False, 'seed': None})
  # here, we have two sets of agents to train with. 
  learned = DoubleDQNAgent(5*52, 110, replay_memory_size=config.replay_size, epsilon=config.epsilon, epsilon_stop=config.epsilon_stop, learning_rate=config.learning_rate, device='cuda', hidden_size=config.hidden_size)
  agents_1 = [learned, RandomAgent(num_actions=110)]
  agents_2 = [learned, GinRummyNoviceRuleAgent()]
  total = 0
  
  Path(f"./checkpoints/homegrown/{run.sweep_id}/{run.name}").mkdir(parents=True, exist_ok=True)
  
  while total < config.steps:
    # randomly choose an agent pair, and randomly permute it's order
    agents = random.sample(random.sample([agents_1, agents_2], 1)[0], 2)
    env_train.set_agents(agents) # choose random opponent and position
    # generate trajectories
    trajectories, payoffs = env_train.run(is_training=True)
    trajectories = rlcard.utils.reorganize(trajectories, payoffs)
    for i, agent in enumerate(agents):
      if not isinstance(agent, DoubleDQNAgent): continue
      for ts in trajectories[i]:
        # feed transition
        agents[i].feed(ts)  
        if total % 50000 == 0:
          env_train.set_agents(agents_1)
          payoffs_train_rand = rlcard.utils.tournament(env_train, 50)
          env_eval.set_agents(agents_1)
          payoffs_eval_rand = rlcard.utils.tournament(env_eval, 50)
          env_eval.set_agents(agents_2)
          payoffs_eval_rule = rlcard.utils.tournament(env_eval, 50)
          wandb.log({"train-rand": payoffs_train_rand[0], "eval-rand": payoffs_eval_rand[0], "eval-rule": payoffs_eval_rule[0]})
          # checkpoint if sufficient average reward met
          if payoffs_eval_rand[0] >= 0.1:
            learned.save_checkpoint(f"./checkpoints/homegrown/{run.sweep_id}/{run.name}", f"dqn_{total}.pt")
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
