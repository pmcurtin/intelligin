import rlcard
from rlcard.agents import RandomAgent
import numpy as np
from functools import reduce
import torch
import torch.nn as nn
import random
from copy import deepcopy
import os
from collections import deque

# imports for the ModifiedGinRummyGame and ModifiedGinRummyEnv
from rlcard.games.gin_rummy.player import GinRummyPlayer
from rlcard.games.gin_rummy.round import GinRummyRound
from rlcard.games.gin_rummy.judge import GinRummyJudge
from rlcard.games.gin_rummy.utils.settings import Settings, DealerForRound

from rlcard.games.gin_rummy.utils.action_event import *

# utility functions and constants, mainly for the HumanAgent
o = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]
cards = reduce(lambda x,y: x+y, [[chr(127137+i+j*16) for i in o] for j in range(4)], [])
def render_gin_state(state: np.ndarray):
  # get unicode codepoints for each of the cards
  hand = ""
  # for each card in hand
  for i in range(52):
    if i % 13 == 0 and i != 0:
      hand += "\n"
    hand += cards[i] if state[0,i] else "＿"
  return f"\U0001F0A0 {cards[np.argmax(state[1])] if not np.all(state[1] == 0) else '＿'}\n\n" + hand

def select_card() -> str:
  card = input('Which card?\n -> ')
  i = ['a', '2', '3', '4', '5', '6', '7', '8', '9', 't', 'j', 'q', 'k']
  j = ['s', 'h', 'd', 'c']
  return 6 + i.index(card[0]) + j.index(card[1]) * 13

# a class for playing the environment as a human, in colab
class HumanAgent:
  def __init__(self, num_actions, name="Human"):
    self.use_raw = True
    self.num_actions = num_actions
    self.name = name
    print("""Action types:
0: Draw new card
1: Draw from discard pile
2: Declare dead hand
3: Gin
4: Discard a card
5: Knock
(-1: Quit)

Card selection format:
- Suits: s|h|d|c
- Cards: a|2|3|4|5|6|7|8|9|t|j|q|k
{Suit}{Card}, e.g. King of hearts = kh""")
    pass

  #@staticmethod
  def step(self, state):
    #print(state)
    print(f"{self.name}'s turn:")
    print(render_gin_state(state['obs']))
    legals = [a for a, _ in state['legal_actions'].items()]
    if 2 in legals:
      print("Must draw a card.")
    elif 5 in legals:
      print("Must Gin.")
    elif 4 in legals:
      print("Dead hand.")
      return rlcard.games.gin_rummy.utils.action_event.ActionEvent.decode_action(4)
    else:
      print("Must Discard/knock.")
    if legals == [0] or legals == [1]:
      return rlcard.games.gin_rummy.utils.action_event.ActionEvent.decode_action(legals[0])
    #print(legals)
    #print(state['legal_actions'])
    legal = False
    while not legal:
      try:
        a_type = int(input("Choose action type:\n -> "))
      except:
        print("Illegal selection. Please try again.")
        continue
      if a_type == -1:
        assert False
      match a_type:
        case 0:
          if 2 in legals:
            return rlcard.games.gin_rummy.utils.action_event.ActionEvent.decode_action(2)
        case 1:
          if 3 in legals:
            return rlcard.games.gin_rummy.utils.action_event.ActionEvent.decode_action(3)
        case 2:
          if 4 in legals:
            return rlcard.games.gin_rummy.utils.action_event.ActionEvent.decode_action(4)
        case 3:
          if 5 in legals:
            return rlcard.games.gin_rummy.utils.action_event.ActionEvent.decode_action(5)
        case 4:
          try:
            action = select_card()
            #print(action)
            if action in legals:
              return rlcard.games.gin_rummy.utils.action_event.ActionEvent.decode_action(action)
          except:
            pass
        case 5:
          try:
            action = select_card() + 52
            #print(action)
            if action in legals:
              return rlcard.games.gin_rummy.utils.action_event.ActionEvent.decode_action(action)
          except:
            pass

      print("Illegal selection. Please try again.")

  def eval_step(self, state):
    #print(self.last_other)
    #print(state['obs'][3])
    # this will only work in colab
    output.clear()
    #if self.last_other is not None:
    if np.any(state['obs'][2] == 1):
      # opponent picked up card, should be at max one other location
      i = np.where(state['obs'][2] == 1)[0][0]
      #print(i)
      print(f"Opponent drew {cards[i]} from discard pile.")
    #self.last_other = state['obs'][3]
    return self.step(state), {}


# based on the rlcard.games.gin_rummy.Game
class ModifiedGinRummyGame(rlcard.games.gin_rummy.Game):
  def __init__(self, allow_step_back=False):
    '''Initialize the class GinRummyGame
    '''
    super().__init__(allow_step_back)

  def init_game(self):
    ''' Initialize all characters in the game and start round 1
    '''
    # same as in the standard gin rummy game
    dealer_id = self.np_random.choice([0, 1])
    if self.settings.dealer_for_round == DealerForRound.North:
        dealer_id = 0
    elif self.settings.dealer_for_round == DealerForRound.South:
        dealer_id = 1
    self.actions = []
    self.round = GinRummyRound(dealer_id=dealer_id, np_random=self.np_random)
    for i in range(2):
        num = 11 if i == 0 else 10
        player = self.round.players[(dealer_id + 1 + i) % 2]
        self.round.dealer.deal_cards(player=player, num=num)
    current_player_id = self.round.current_player_id
    state = self.get_state(player_id=current_player_id)
    # these instance variables are added:
    self.picked_up_discard = False
    self.picked_up_discard_player = None
    return state, current_player_id

  def step(self, action: ActionEvent):
    ''' Perform game action and return next player number, and the state for next player
    '''
    if self.round.current_player_id != self.picked_up_discard_player:
      self.picked_up_discard = False
      #print("New player, resetting.")
    if isinstance(action, ScoreNorthPlayerAction):
        self.round.score_player_0(action)
    elif isinstance(action, ScoreSouthPlayerAction):
        self.round.score_player_1(action)
    elif isinstance(action, DrawCardAction):
        self.round.draw_card(action)
    elif isinstance(action, PickUpDiscardAction):
        self.round.pick_up_discard(action)
        self.picked_up_discard = True # keep track of whether opponent just picked up card from discard
        self.picked_up_discard_player = self.round.current_player_id
        #print(f"{self.picked_up_discard_player} picked up discard.")
    elif isinstance(action, DeclareDeadHandAction):
        self.round.declare_dead_hand(action)
    elif isinstance(action, GinAction):
        self.round.gin(action, going_out_deadwood_count=self.settings.going_out_deadwood_count)
    elif isinstance(action, DiscardAction):
        self.round.discard(action)
    elif isinstance(action, KnockAction):
        self.round.knock(action)
    else:
        raise Exception('Unknown step action={}'.format(action))
    self.actions.append(action)
    next_player_id = self.round.current_player_id
    next_state = self.get_state(player_id=next_player_id)
    return next_state, next_player_id

# heavily based on rlcard.envs.gin_rummy.GinRummyEnv
class ModifiedGinRummyEnv(rlcard.envs.gin_rummy.GinRummyEnv):
  def __init__(self, config):
    super().__init__(config)
    self.game = ModifiedGinRummyGame()
    # different state_shape... [5, 52] to [3, 52]
    self.state_shape = [[3, 52] for _ in range(self.num_players)]

  def _extract_state(self, state): 
    if self.game.is_over():
      obs = np.array([self._utils.encode_cards([]) for _ in range(3)])
      extracted_state = {'obs': obs, 'legal_actions': self._get_legal_actions()}
      extracted_state['raw_legal_actions'] = list(self._get_legal_actions().keys())
      extracted_state['raw_obs'] = obs
    else:
      # a lot of information is removed from the observation, here
      discard_pile = self.game.round.dealer.discard_pile
      #stock_pile = self.game.round.dealer.stock_pile
      top_discard = [] if not discard_pile else [discard_pile[-1]]
      #dead_cards = discard_pile[:-1]
      current_player = self.game.get_current_player()
      opponent = self.game.round.players[(current_player.player_id + 1) % 2]
      #print(opponent.known_cards)
      # importantly, we allow each player to know if their adversary picked up a card from the discard pile last turn, and what it was.
      known_cards = [opponent.known_cards[-1]] if self.game.picked_up_discard and self.game.round.current_player_id != self.game.picked_up_discard_player and len(opponent.known_cards) != 0 else []
      #unknown_cards = stock_pile + [card for card in opponent.hand if card not in known_cards]
      hand_rep = self._utils.encode_cards(current_player.hand)
      top_discard_rep = self._utils.encode_cards(top_discard)
      #dead_cards_rep = self._utils.encode_cards(dead_cards)
      known_cards_rep = self._utils.encode_cards(known_cards) # just the card that opponent just took from discard pile, if applicable
      #unknown_cards_rep = self._utils.encode_cards(unknown_cards)
      rep = [hand_rep, top_discard_rep, known_cards_rep]
      obs = np.array(rep)
      extracted_state = {'obs': obs, 'legal_actions': self._get_legal_actions(), 'raw_legal_actions': list(self._get_legal_actions().keys())}
      extracted_state['raw_obs'] = obs
    return extracted_state

# inspired by RLCard's DQN memory object.
class Memory():
  def __init__(self, replay_memory_size, batch_size):
    self.mem = [] #deque(maxlen=replay_memory_size)
    self.max_size = replay_memory_size
    self.batch_size = batch_size

  def save_transition(self, state, action, reward, next_state, done, legal_actions):
    if len(self.mem) > self.max_size:
      self.mem.pop(0)
    self.mem.append((state, action, reward, next_state, done, legal_actions))

  def sample_batch(self):
    if len(self.mem) >= self.batch_size:
        t = tuple(zip(*random.sample(self.mem, self.batch_size)))
        return tuple(map(np.array, t[:-1])) + tuple(t[-1:])
    raise NotImplementedError

  def checkpoint_attributes(self):
    return {
      'max_size': self.max_size,
      'batch_size': self.batch_size,
      'mem': self.mem
    }

  @classmethod
  def from_checkpoint(cls, checkpoint):
    mem = cls(checkpoint['max_size'], checkpoint['batch_size'])
    mem.mem = checkpoint['mem']
    return mem

# underlying q-network 
class QNet(nn.Module):

  def __init__(self, state_size, action_size, device, hidden_size=128, num_hidden=2):
    super().__init__()

    sizes = [state_size] + [hidden_size] * num_hidden
    # initialize MLP
    modules = [nn.BatchNorm1d(sizes[0])]
    for i in range(len(sizes)-1):
        modules.append(nn.Linear(sizes[i], sizes[i+1]))
        modules.append(nn.Tanh())
    modules.append(nn.Linear(sizes[-1], action_size))
    self.sequential = nn.Sequential(*modules)

    self.sequential.to(device)
    self.device = device
    self.state_size = state_size
    self.action_size = action_size
    self.hidden_size = hidden_size
    self.num_hidden = num_hidden

  def no_grad(self, state, train=False):
    # technically this isn't a great pytorch model because it accepts np arrays not tensors
    if train:
      self.sequential.train()
    else:
       self.sequential.eval()
    with torch.no_grad():
      return self.sequential(torch.from_numpy(state).float().view(-1, self.state_size).to(self.device))

  def forward(self, state):
    self.sequential.train()
    return self.sequential(torch.from_numpy(state).float().view(-1, self.state_size).to(self.device))

  def checkpoint_attributes(self):
    return {
      'seq': self.sequential.state_dict(),
      'state_size': self.state_size,
      'action_size': self.action_size,
      'device': self.device,
      'hidden_size': self.hidden_size,
      'num_hidden': self.num_hidden
    }

  @classmethod
  def from_checkpoint(cls, checkpoint):
    qnet = cls(
        checkpoint['state_size'],
        checkpoint['action_size'],
        checkpoint['device'],
        checkpoint['hidden_size'],
        checkpoint['num_hidden']
    )
    qnet.sequential.load_state_dict(checkpoint['seq'])
    return qnet
    
class DQNAgent(object):
    def __init__(self,
                 state_size,
                 num_actions,
                 replay_memory_size=20000,
                 replay_memory_init_size=100,
                 epsilon_stop=100000,
                 gamma=0.99,
                 epsilon=0.1,
                 batch_size=32,
                 learning_rate=0.00005,
                 save_path=None,
                 device='cpu',
                 num_hidden=2,
                 hidden_size=128):
        self.use_raw = False
        self.replay_memory_init_size = max(replay_memory_init_size, batch_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.device = torch.device(device)
        self.state_size = state_size

        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0

        # Create estimators
        self.q_net = QNet(state_size, num_actions, self.device, num_hidden=num_hidden, hidden_size=hidden_size)
	
        # Create replay memory
        self.memory = Memory(replay_memory_size, batch_size)

        # epsilon decay
        self.epsilon_stop = epsilon_stop
        self.epsilons = np.linspace(1, epsilon, epsilon_stop)

        self.optimizer = torch.optim.RMSprop(self.q_net.parameters(), lr=learning_rate)

        # Checkpoint saving parameters
        #self.save_path = save_path
        #self.save_every = save_every

    def feed(self, trajectory):
        state, action, reward, next_state, done = tuple(trajectory)
        if len(next_state['legal_actions'].keys()) != 0: # weird edge case we encountered
          self.memory.save_transition(state['obs'], action, reward, next_state['obs'], done, list(next_state['legal_actions'].keys()))
          self.total_t += 1
          if self.total_t - self.replay_memory_init_size >= 0:
              self.train()

    def step(self, state):
        # get q-values
        q_values = self.q_net.no_grad(np.expand_dims(state['obs'], 0)) # maybe some indexing to do here blah blah depending on q_net structure
        # identify legal actions, make epsilon
        legal_actions = list(state['legal_actions'].keys())
        epsilon = self.epsilons[min(self.total_t, self.epsilon_stop-1)]
        probs = (torch.ones(len(legal_actions), dtype=torch.float) * epsilon / len(legal_actions)).to(self.device)
        #print(legal_actions)
        #print(q_values)
        # make mask where non-legal actions have -inf value
        legal_inf_mask = torch.from_numpy(np.where(np.isin(np.arange(self.num_actions), legal_actions), 0, -np.inf)).float().to(self.device)
        # add mask and q-values, non-(-inf) values are q-values for legal actions
        masked_q_values = q_values + legal_inf_mask
        best_action_idx = legal_actions.index(torch.argmax(masked_q_values))
        probs[best_action_idx] += (1.0 - epsilon)
        action_idx = torch.multinomial(probs, 1) 

        return legal_actions[action_idx]

    def eval_step(self, state):
        # get q-values
        q_values = self.q_net.no_grad(np.expand_dims(state['obs'], 0)) #.detach().cpu().numpy()
        # do the same as self.step, but without epsilon-greedy selection
        legal_actions = list(state['legal_actions'].keys())
        legal_inf_mask = torch.from_numpy(np.where(np.isin(np.arange(self.num_actions), legal_actions), 0, -np.inf)).float().to(self.device)
        masked_q_values = q_values + legal_inf_mask
        best_action = torch.argmax(masked_q_values)
        info = {}
        return best_action, info

    def train(self):
        
        # get batch of transitions
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, legal_actions_batch   = self.memory.sample_batch()

        # get q-values at next_state_batch
        next_qs = self.q_net.no_grad(next_state_batch)
        # compute best (legal) actions in next state and their q-values for each example
        legal_inf_mask = torch.from_numpy(np.array([np.where(np.isin(np.arange(self.num_actions), legal_actions), 0, -np.inf) for legal_actions in legal_actions_batch])).float().to(self.device)
        masked_q_values = next_qs + legal_inf_mask
        best_action_values = torch.max(masked_q_values, dim=1).values
        # compute target = r + (inverse of done_batch)*gamma*best-next-q-values
        targets = torch.from_numpy(reward_batch).float().to(self.device) + (~torch.from_numpy(done_batch).to(self.device)).float()*self.gamma*best_action_values
        # compute mse loss over network given input state_batch, and on action_batch actions, with target as above
        loss = torch.nn.functional.mse_loss(targets, self.q_net(state_batch)[torch.arange(self.batch_size), action_batch])

        # take gradient step on this loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.train_t % 100 == 0:
          print(f'\rstep {self.total_t}, loss: {loss}', end='')

        self.train_t += 1
        
class DoubleDQNAgent(object):
    def __init__(self,
                 state_size,
                 num_actions,
                 replay_memory_size=20000,
                 replay_memory_init_size=100,
                 epsilon_stop=100000,
                 gamma=0.99,
                 epsilon=0.1,
                 batch_size=32,
                 learning_rate=0.00005,
                 save_path=None,
                 device='cpu',
                 num_hidden=2,
                 hidden_size=128):
        self.use_raw = False
        self.replay_memory_init_size = max(replay_memory_init_size, batch_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.device = torch.device(device)
        self.state_size = state_size
        
        self.can_train = True

        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0

        # Create estimators
        self.qnet = QNet(state_size, num_actions, self.device, num_hidden=num_hidden, hidden_size=hidden_size)
        self.qnet_prime = QNet(state_size, num_actions, self.device, num_hidden=num_hidden, hidden_size=hidden_size)

        self.tau = 0.01

        # Create replay memory
        self.memory = Memory(replay_memory_size, batch_size)

        # epsilon decay
        self.epsilon_stop = epsilon_stop
        self.epsilons = np.linspace(1, epsilon, epsilon_stop)

        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=learning_rate)

        # Checkpoint saving parameters
        self.save_path = save_path

    def feed(self, trajectory):
        # same as DQNAgent.step, defined earlier
        state, action, reward, next_state, done = tuple(trajectory)
        #if len(next_state['legal_actions'].keys()) != 0: # this somehow was no longer necessary
        self.memory.save_transition(state['obs'], action, reward, next_state['obs'], done, list(next_state['legal_actions'].keys()))
        self.total_t += 1
        if self.total_t - self.replay_memory_init_size >= 0 and self.total_t % 1 == 0:
            self.train()

    def step(self, state):
        # detail for self-play, disables e-greedy actions if !self.can_train
        if not self.can_train:
          return self.eval_step(state)[0]
        # using numpy arrays instead of tensors as in the earlier implementation benefited speed.
        # probably due to unnecessary tensor.to calls.
        # get q-values
        q_values = self.qnet.no_grad(np.expand_dims(state['obs'], 0)).cpu().detach().numpy()
        legal_actions = list(state['legal_actions'].keys())
        epsilon = self.epsilons[min(self.total_t, self.epsilon_stop-1)]
        probs = np.ones(len(legal_actions)) * epsilon / len(legal_actions)
        # mask
        legal_inf_mask = np.where(np.isin(np.arange(self.num_actions), legal_actions), 0, -np.inf)
        masked_q_values = q_values + legal_inf_mask
        best_action_idx = legal_actions.index(np.argmax(masked_q_values))
        probs[best_action_idx] += (1.0 - epsilon)
        # select action
        action_idx = np.random.choice(np.arange(len(probs)), p=probs)

        return legal_actions[action_idx]

    def eval_step(self, state):
        # same as self.step, but without e-greedy actions
        q_values = self.qnet.no_grad(np.expand_dims(state['obs'], 0)).cpu().detach().numpy()
        legal_actions = list(state['legal_actions'].keys())
        legal_inf_mask = np.where(np.isin(np.arange(self.num_actions), legal_actions), 0, -np.inf)
        masked_q_values = q_values + legal_inf_mask
        best_action = np.argmax(masked_q_values)
        info = {}
        return best_action, info

    def train(self):

        # sample a batch of transitions from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, legal_actions_batch = self.memory.sample_batch()

        # get q-values at next_state_batch
        next_qs = self.qnet.no_grad(next_state_batch).cpu().detach().numpy() # something like this
        # compute best (legal) actions in next state and their q-values for each example
        legal_inf_mask = np.array([np.where(np.isin(np.arange(self.num_actions), legal_actions), 0, -np.inf) for legal_actions in legal_actions_batch])
        masked_q_values = next_qs + legal_inf_mask
        best_actions = np.argmax(masked_q_values, axis=-1)
        # now get qvalue from qnet using best action according to qnet_prime
        best_action_values = self.qnet_prime.no_grad(next_state_batch)[torch.arange(self.batch_size), best_actions].cpu().detach().numpy()
        # compute target = r + (inverse of done_batch)*gamma*best-next-q-values
        targets = reward_batch + np.invert(done_batch)*self.gamma*best_action_values
        # compute mse loss over network given input state_batch, and on action_batch actions, with target as above
        loss = torch.nn.functional.mse_loss(torch.from_numpy(targets).float().to(self.device), self.qnet(state_batch)[torch.arange(self.batch_size), action_batch])

        # take gradient step on this loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Polyak averaging to update qnet_prime
        #params_q, params_qp = self.qnet.named_parameters(), self.qnet_prime.named_parameters()

        #new_params_qp = dict(params_qp)

        #for q_name, q_param in params_q:
        #  if q_name in new_params_qp:
        #      new_params_qp[q_name].data.copy_(self.tau*q_param.data + (1-self.tau)*new_params_qp[q_name].data)

        #self.qnet_prime.load_state_dict(new_params_qp)

        if self.train_t % 1000 == 0:
            self.qnet_prime = deepcopy(self.qnet)

        if self.train_t % 100 == 0:
          print(f'\rstep {self.total_t}, loss: {loss}', end='')

        self.train_t += 1

    def checkpoint_attributes(self):
      return {
          'qnet': self.qnet.checkpoint_attributes(),
          'memory': self.memory.checkpoint_attributes(),
          'tau': self.tau, # was for polyak averaging, which didn't really work. no longer used
          'total_t': self.total_t,
          'train_t': self.train_t,
          'replay_memory_init_size': self.replay_memory_init_size,
          'gamma': self.gamma,
          'epsilon': self.epsilon,
          'epsilon_stop': self.epsilon_stop,
          'batch_size': self.batch_size,
          'num_actions': self.num_actions,
          'learning_rate': self.learning_rate,
          'device': self.device,
          'save_path': self.save_path,
          'state_size': self.state_size,
          'optimizer': self.optimizer.state_dict(),
      }

    @classmethod
    def from_checkpoint(cls, checkpoint):
      agent = cls(
        state_size=checkpoint['state_size'],
        num_actions=checkpoint['num_actions'],
        replay_memory_init_size=checkpoint['replay_memory_init_size'],
        epsilon_stop=checkpoint['epsilon_stop'],
        gamma=checkpoint['gamma'],
        epsilon=checkpoint['epsilon'],
        batch_size=checkpoint['batch_size'],
        learning_rate=checkpoint['learning_rate'],
        save_path=checkpoint['save_path'],
        device=checkpoint['device'],
      )
      agent.total_t = checkpoint['total_t']
      agent.train_t = checkpoint['train_t']
      agent.qnet = QNet.from_checkpoint(checkpoint['qnet'])
      agent.qnet_prime = deepcopy(agent.qnet)
      agent.memory = Memory.from_checkpoint(checkpoint['memory'])
      agent.optimizer.load_state_dict(checkpoint['optimizer'])
      return agent

    def save_checkpoint(self, path, filename='checkpoint_dqn.pt'):
      torch.save(self.checkpoint_attributes(), os.path.join(path, filename))
      
class SeqMemory():
  def __init__(self, replay_memory_size, batch_size):
    self.mem = []
    self.max_size = replay_memory_size
    self.batch_size = batch_size
    # need an additional size var to keep track of total number of transitions, which is 
    # no longer just len(mem)
    self.size = 0

  def save_ep(self, trajectory):
    # trajectory: list[tuple[state, action, reward, new_state, done, legal_actions]] (or something)
    # 
    while self.size >= self.max_size:
      self.size -= len(self.mem.pop(0)[0])
    self.mem.append(trajectory) # tuple[list[state]...]?
    self.size += len(trajectory[0])

  def sample_batch(self): # fix for new mem type...
    if len(self.mem) >= self.batch_size:
        t = tuple(zip(*random.sample(self.mem, self.batch_size)))
        # it works, trust
        return tuple(map(lambda x: nn.utils.rnn.pad_sequence(map(lambda y: torch.tensor(y, dtype=torch.float32), x), batch_first=True).numpy(), t[:-1])) + tuple(t[-1:]), list(map(len, t[0]))
    raise NotImplementedError

  def checkpoint_attributes(self):
    return {
      'max_size': self.max_size,
      'batch_size': self.batch_size,
      'mem': self.mem,
      'size': self.size
    }

  @classmethod
  def from_checkpoint(cls, checkpoint):
    mem = cls(checkpoint['max_size'], checkpoint['batch_size'])
    mem.mem = checkpoint['mem']
    mem.size = checkpoint['size']
    return mem

class SeqQNet(nn.Module):

  def __init__(self, state_size, action_size, device, hidden_size=128, num_hidden=2):
    super().__init__()

    sizes = [state_size] + [hidden_size] * num_hidden
    # define simple GRU
    self.bnorm = nn.BatchNorm1d(state_size)
    # here, num_hidden becomes number of GRU units stacked.
    self.gru = nn.GRU(state_size, hidden_size, num_hidden, batch_first=True)
    self.linear = nn.Linear(sizes[-1], action_size)

    for m in [self.bnorm, self.gru, self.linear]:
      for p in m.parameters():
        if len(p.data.shape) > 1:
            nn.init.xavier_uniform_(p.data)

    self.to(device)
    self.device = device
    self.state_size = state_size
    self.action_size = action_size
    self.hidden_size = hidden_size
    self.num_hidden = num_hidden

  def no_grad(self, state, h, train=False):
    # again, not a good module as we accept np arrays
    if train:
      self.train()
    else:
      self.eval()

    with torch.no_grad():
      x = self.bnorm(torch.from_numpy(state).float().view(*state.shape[:2], -1).to(self.device).permute(0, 2, 1))
      x, h = self.gru(x.permute(0, 2, 1), h)
      return self.linear(x), h

  def forward(self, state, h):
    self.train()
    x = self.bnorm(torch.from_numpy(state).float().view(*state.shape[:2], -1).to(self.device).permute(0, 2, 1))
    x, h = self.gru(x.permute(0, 2, 1), h)
    return self.linear(x), h

  def checkpoint_attributes(self):
    return {
      'state_size': self.state_size,
      'action_size': self.action_size,
      'device': self.device,
      'hidden_size': self.hidden_size,
      'num_hidden': self.num_hidden,
      'bnorm': self.bnorm.state_dict(),
      'gru': self.gru.state_dict(),
      'linear': self.linear.state_dict()
    }

  @classmethod
  def from_checkpoint(cls, checkpoint):
    qnet = cls(
        checkpoint['state_size'],
        checkpoint['action_size'],
        checkpoint['device'],
        checkpoint['hidden_size'],
        checkpoint['num_hidden']
    )
    qnet.gru.load_state_dict(checkpoint['gru'])
    qnet.bnorm.load_state_dict(checkpoint['bnorm'])
    qnet.linear.load_state_dict(checkpoint['linear'])
    return qnet
    
class SeqDoubleDQNAgent(object):
    '''
    Approximate clone of rlcard.agents.dqn_agent.DQNAgent
    that depends on PyTorch instead of Tensorflow
    '''
    def __init__(self,
                 state_size,
                 num_actions,
                 replay_memory_size=20000,
                 replay_memory_init_size=100,
                 epsilon_stop=100000,
                 gamma=0.99,
                 epsilon=0.1,
                 batch_size=32,
                 learning_rate=0.00005,
                 save_path=None,
                 device='cpu',
                 num_hidden=2,
                 hidden_size=128):
        self.use_raw = False
        self.replay_memory_init_size = max(replay_memory_init_size, batch_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.device = torch.device(device)
        self.state_size = state_size

        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0

        # Create estimators
        self.qnet = SeqQNet(state_size, num_actions, self.device, num_hidden=num_hidden, hidden_size=hidden_size)
        self.qnet_prime = SeqQNet(state_size, num_actions, self.device, num_hidden=num_hidden, hidden_size=hidden_size)

        #self.tau = 0.01
        # Initial hidden state
        self.h = torch.zeros((num_hidden, 1, hidden_size)).to(self.device)
        self.hidden_size = hidden_size
        self.num_hidden = num_hidden

        # Create replay memory
        self.memory = SeqMemory(replay_memory_size, batch_size)

        # epsilon decay
        self.epsilon_stop = epsilon_stop
        self.epsilons = np.linspace(1, epsilon, epsilon_stop)

        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=learning_rate)

        # Checkpoint saving parameters
        self.save_path = save_path

    def reset(self):
        # resets hidden state to zeros
        self.h = torch.zeros_like(self.h).to(self.device)

    def feed(self, trajectory):
        t = []
        # trajectory is entire episode, now
        for t_ in trajectory:
          state, action, reward, next_state, done = tuple(t_)
          t.append((state['obs'], action, reward, next_state['obs'], done, list(next_state['legal_actions'].keys())))
        t = tuple(zip(*t))
        self.memory.save_ep(tuple(map(np.array, t[:-1])) + tuple(t[-1:]))
        self.total_t += len(trajectory)
        if len(self.memory.mem) - self.replay_memory_init_size >= 0 and self.total_t % 1 == 0:
            self.train()

    def step(self, state):
        # assume that self.h is initiliazed properly. requires agent.reset() after an episode to reset self.h to zeros.

        # fetch q-values
        q_values, self.h = self.qnet.no_grad(state['obs'][np.newaxis, np.newaxis, :], self.h)
        q_values = q_values.cpu().detach().numpy()
        legal_actions = list(state['legal_actions'].keys())
        epsilon = self.epsilons[min(self.total_t, self.epsilon_stop-1)]
        probs = np.ones(len(legal_actions)) * epsilon / len(legal_actions)
        #print(legal_actions)
        #print(q_values)
        # mask as usual
        legal_inf_mask = np.where(np.isin(np.arange(self.num_actions), legal_actions), 0, -np.inf)
        masked_q_values = q_values + legal_inf_mask
        best_action_idx = legal_actions.index(np.argmax(masked_q_values))
        probs[best_action_idx] += (1.0 - epsilon)
        action_idx = np.random.choice(np.arange(len(probs)), p=probs)

        return legal_actions[action_idx]

    def eval_step(self, state):
        # same as above, no e-greedy
        q_values, self.h = self.qnet.no_grad(state['obs'][np.newaxis, np.newaxis, :], self.h)
        q_values = q_values.cpu().detach().numpy()
        legal_actions = list(state['legal_actions'].keys())
        legal_inf_mask = np.where(np.isin(np.arange(self.num_actions), legal_actions), 0, -np.inf)
        masked_q_values = q_values + legal_inf_mask
        best_action = np.argmax(masked_q_values)
        info = {}
        return best_action, info

    def train(self):
        # get episode lengths, which are not all equal
        dat, lens = self.memory.sample_batch()
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, legal_actions_batch = dat
        done_batch = done_batch.astype(int)
        action_batch = action_batch.astype(int)
        # properly init. hidden values for all upcoming calls on next_state_batch
        h = torch.zeros((self.num_hidden, self.batch_size, self.hidden_size)).to(self.device)
        h_ = self.qnet.no_grad(state_batch[:, :1], h)[1] # get hidden from first state (for init hidden of next_state_batch
        h_p = self.qnet_prime.no_grad(state_batch[:, :1], h)[1] # same as above but with other network
        # get q-values at next_state_batch
        next_qs, _ = self.qnet.no_grad(next_state_batch, h_) # use h_ here not h, for proper init
        next_qs = next_qs.cpu().detach().numpy()
        # compute best (legal) actions in next state and their q-values for each example
        # yes this is terrible, but just a version of the DoubleDQNAgent one with an extra dimension for timesteps, basically
        legal_inf_mask = nn.utils.rnn.pad_sequence([torch.from_numpy(np.array([np.where(np.isin(np.arange(self.num_actions), legal_actions), 0, -np.inf)
                                                                                for legal_actions in episode_legal_actions])) for episode_legal_actions in legal_actions_batch], batch_first=True).numpy()
        masked_q_values = next_qs + legal_inf_mask
        best_actions = np.argmax(masked_q_values, axis=-1)
        # now get qvalue from qnet using best action according to qnet_prime
        best_action_values, _ = self.qnet_prime.no_grad(next_state_batch, h_p) # use h_p here
        # gather the values by the best actions
        best_action_values = torch.gather(best_action_values.cpu(), 2, torch.from_numpy(best_actions[:, :, np.newaxis])).squeeze(2).detach().numpy() # probably wrong index shapes?
        # compute target = r + (inverse of done_batch)*gamma*best-next-q-values
        targets = torch.from_numpy(reward_batch + np.invert(done_batch)*self.gamma*best_action_values).float().to(self.device) # shapes?
        # more lovely terrible pytorch
        qs = torch.gather(self.qnet(state_batch, h)[0], 2, torch.from_numpy(action_batch[:, :, np.newaxis]).to(self.device)).squeeze(2) # use h not h_ or h_p

        # compute mse loss over network given input state_batch, and on action_batch actions, with target as above
        losses = torch.nn.functional.mse_loss(targets, qs, reduction='none') # probably wrong shape again
        # mask out loss terms on the padded out terms, based on lens
        losses[torch.arange(0, losses.size()[-1], dtype=torch.float32).repeat(self.batch_size,1) >= torch.tensor(lens, dtype=torch.int).unsqueeze(1)] = 0
        #print(losses)
        #samples = [random.randint(0, l-1) for l in lens]
        # simplify this line, probably:
        #losses[torch.ne(torch.arange(0, losses.size()[-1], dtype=torch.int).repeat(self.batch_size,1), torch.tensor(samples, dtype=torch.int).unsqueeze(1).repeat(1, losses.size()[-1]))] = 0
        loss = torch.mean(losses)
        # take gradient step on this loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Polyak averaging to update qnet_prime
        #params_q, params_qp = self.qnet.named_parameters(), self.qnet_prime.named_parameters()

        #new_params_qp = dict(params_qp)

        #for q_name, q_param in params_q:
        #  if q_name in new_params_qp:
        #      new_params_qp[q_name].data.copy_(self.tau*q_param.data + (1-self.tau)*new_params_qp[q_name].data)

        #self.qnet_prime.load_state_dict(new_params_qp)

        if self.train_t % 1000 == 0 and self.train_t != 0:
            self.qnet_prime = deepcopy(self.qnet)
        #if self.train_t % 100 == 0:
        print(f'\rstep {self.total_t}, loss: {loss}', end='')
        self.train_t += 1 # not really accurate, since each minibatch trains on <batch_size> episodes

    def checkpoint_attributes(self):
      return {
          'qnet': self.qnet.checkpoint_attributes(),
          'memory': self.memory.checkpoint_attributes(),
          'tau': self.tau,
          'total_t': self.total_t,
          'train_t': self.train_t,
          'replay_memory_init_size': self.replay_memory_init_size,
          'gamma': self.gamma,
          'epsilon': self.epsilon,
          'epsilon_stop': self.epsilon_stop,
          'batch_size': self.batch_size,
          'num_actions': self.num_actions,
          'learning_rate': self.learning_rate,
          'device': self.device,
          'save_path': self.save_path,
          'state_size': self.state_size,
          'optimizer': self.optimizer.state_dict(),
          'num_hidden': self.num_hidden,
          'hidden_size': self.hidden_size
      }

    @classmethod
    def from_checkpoint(cls, checkpoint):
      agent = cls(
        state_size=checkpoint['state_size'],
        num_actions=checkpoint['num_actions'],
        replay_memory_init_size=checkpoint['replay_memory_init_size'],
        epsilon_stop=checkpoint['epsilon_stop'],
        gamma=checkpoint['gamma'],
        epsilon=checkpoint['epsilon'],
        batch_size=checkpoint['batch_size'],
        learning_rate=checkpoint['learning_rate'],
        save_path=checkpoint['save_path'],
        device=checkpoint['device'],
        hidden_size=checkpoint['hidden_size'],
        num_hidden=checkpoint['num_hidden']
      )
      agent.total_t = checkpoint['total_t']
      agent.train_t = checkpoint['train_t']
      agent.qnet = QNet.from_checkpoint(checkpoint['qnet'])
      agent.qnet_prime = deepcopy(agent.qnet)
      agent.memory = Memory.from_checkpoint(checkpoint['memory'])
      agent.optimizer.load_state_dict(checkpoint['optimizer'])
      return agent

    def save_checkpoint(self, path, filename='checkpoint_dqn.pt'):
      torch.save(self.checkpoint_attributes(), os.path.join(path, filename))


def run(env, is_training=False):
    trajectories = [[] for _ in range(env.num_players)]
    state, player_id = env.reset()

    # reset agent.h for seq agent - only change from the standard library function
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
        _, _payoffs = run(env, is_training=False) # here use run instead of env.run, only change
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

env = rlcard.make("gin-rummy")
#print(env.state_shape, env.num_actions)
# training
agents = [DoubleDQNAgent(5*52, 110, device='cuda', batch_size=16, replay_memory_size=50000, learning_rate=1e-5), RandomAgent(num_actions=110)]
#agents = [rlcard.agents.DQNAgent(num_actions=110, state_shape=[5, 52], mlp_layers=[512, 512, 512, 512], epsilon_decay_steps=200000, replay_memory_size=50000), GinRummyNoviceRuleAgent()]
# env = ModifiedGinRummyEnv({
#         'allow_step_back': False,
#         'seed': None,
#         'is_allowed_to_discard_picked_up_card': False,
#         })
env.set_agents(agents)

for episode in range(500):

  trajectories, payoffs = env.run(is_training=True)
  trajectories = rlcard.utils.reorganize(trajectories, payoffs)
  for t in trajectories[0]:
    agents[0].feed(t)
  # if episode % 100 == 0 and episode != 0:
  #   print("\nRunning evaluation tournament:")
  #   payoffs = tournament(env, 1000)
  #   print(f"Algorithm's average payoff: {payoffs[0]}")
