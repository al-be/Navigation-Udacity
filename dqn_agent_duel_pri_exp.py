import numpy as np
import random
from collections import namedtuple, deque
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,MultiStepLR
from model_dueiling_DQN import QNetwork
from starr import SumTreeArray
import torch
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, seed,buffer_size, epsiolon, alpha, beta,\
                  beta_increa,batch_size,tau,gamma,lr,update_every,alpha_decay_rate,\
                 max_beta_val,min_alpha_val):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.tau=tau
        self.batch_size=batch_size
        self.gamma=gamma
        self.lr=lr
        self.update_every=update_every

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), self.lr)

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++Replay memory
        self.memory = Memory(seed, buffer_size, epsiolon, alpha, beta, beta_increa, batch_size,alpha_decay_rate)
        
        self.t_step = 0
        
    def calculate_error(self,states, actions, rewards, next_states, dones):

        _,Q_locals_next_index = self.qnetwork_local(next_states).detach().max(1)#[0].unsqueeze(1)

        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1,Q_locals_next_index.unsqueeze(1))#.squeeze(1)

        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).detach().gather(1, actions.unsqueeze(1))#.squeeze(1)

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++ Compute loss
        loss = (Q_expected- Q_targets).cpu().numpy()
        return loss

    def step(self,state, action, reward, next_state, done):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++Save experience in replay memory
        #states, actions, rewards, next_states, dones = experiences
        
        next_states = torch.from_numpy(np.array([next_state])).float().to(device)
        actions = torch.from_numpy(np.array([action])).long().to(device)
        states = torch.from_numpy(np.array([state])).float().to(device)
        rewards = torch.from_numpy(np.array([reward])).float().to(device)
        dones = torch.from_numpy(np.array([done])).float().to(device)

        loss=self.calculate_error(states, actions, rewards, next_states, dones)
        self.memory.add(loss,state, action, reward, next_state, done)
          
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++ Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
          # If enough samples are available in memory, get random subset and learn
          if len(self.memory) > self.batch_size:
            experiences_weight_loss = self.memory.sample()
            
            self.learn(experiences_weight_loss)
            
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        #+++++++++++++++++++++++++++++++++++++++++++++++++++ Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self,experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        loss_e,weights,indexes,states, actions, rewards, next_states, dones = experiences

        #-------------------------------- Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        #--------------------------------- Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # --------------------------------Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        if weights is None:
            weights=torch.ones_like(Q_expected.detach())

        #-------------------------------- Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)#*weights
        #-------------------------------- Minimize the loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.memory.update(indexes, loss_e)

        # --------------------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)                     

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

class Memory:
    
    def __init__(self, seed=0, buffer_size=10e5, epsiolon=0.001, alpha=0.6, beta=0.4, beta_increa=0.001, batch_size=64,\
               alpha_decay_rate= 0.99999,max_beta_val=0.9,min_alpha_val=0.2):
        self.buffer_size = int(buffer_size)
        self.epsiolon = epsiolon
        self.alpha = alpha
        self.beta = beta
        self.beta_increa = beta_increa
        self.counter=0
        self.alpha_decay = alpha_decay_rate
        self.max_beta_val=max_beta_val
        self.min_alpha_val=min_alpha_val

        self.sum_tree = SumTreeArray(self.buffer_size, dtype='float32')
        self.sum_tree.sumtree()
        self.memory = deque(maxlen=self.buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


        self.seed = random.seed(seed)

    def cal_priority(self, error):

        return (np.abs(error) + self.epsiolon) ** self.alpha

    def add(self, error, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)

        self.counter = (self.counter + 1) % self.buffer_size

        self.memory.append(e)


        self.sum_tree[self.counter]=self.cal_priority(error)
        self.counter=min(self.counter+1,self.buffer_size)


    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        probal=self.sum_tree[:self.counter]/self.sum_tree[:self.counter].sum()
        self.beta = np.min([self.max_beta_val, self.beta + self.beta_increa])

        indexes =np.random.choice(self.counter, self.batch_size, p=probal)

        self.alpha*=self.alpha_decay
        self.alpha = np.max([self.min_alpha_val, self.alpha])


        prob=self.sum_tree[indexes[:]] / self.sum_tree[:self.counter].sum()

        experiences=[]
        for index in indexes:
            try:
                experiences.append(self.memory[index])
            except:
                pass
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        weights=(self.counter*prob)**(-self.beta)
        weights=weights/weights.max()
        error=self.sum_tree[indexes[:]]

        return (error,weights,indexes, states, actions, rewards, next_states, dones)


    def update(self, indexes, errors):

        for index,error in zip(indexes, errors):
            self.sum_tree[index]= self.cal_priority(error)
            
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)