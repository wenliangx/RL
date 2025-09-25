from .state import State, addPlayerCard
from .actions import hit, stick, actions
from .policy import Policy, PolicyTorch
import numpy as np
import torch


class Env:
    def __init__(self, init_state: State):
        self.init_state = init_state

    def run(self, policy: Policy):
        self.states_rewards = []
        player_sum_list = []
        dealer_card_list = []
        usable_ace_list = []
        rewards_list = []
        state = self.init_state
        self.othoer_dealer_cards = [int(np.random.randint(1, 14, size=1)[0])]

        if state.player_sum == 21 and self.dealer_sum != 21:
            player_sum_list.append(state.player_sum)
            dealer_card_list.append(state.dealer_card)
            usable_ace_list.append(state.usable_ace)
            rewards_list.append(1.0)
            return player_sum_list, dealer_card_list, usable_ace_list, rewards_list
        elif state.player_sum == 21 and self.dealer_sum == 21:
            player_sum_list.append(state.player_sum)
            dealer_card_list.append(state.dealer_card)
            usable_ace_list.append(state.usable_ace)
            rewards_list.append(0.0)
            return player_sum_list, dealer_card_list, usable_ace_list, rewards_list
        player_sum_list.append(state.player_sum)
        dealer_card_list.append(state.dealer_card)
        usable_ace_list.append(state.usable_ace)
        rewards_list.append(0.0)
        
        while True:
            action = policy.action(state)
            if action == hit:
                card = int(np.random.randint(1, 14, size=1)[0])
                state = addPlayerCard(state, card)
                if self._check_if_stop(state):
                    reward = self._get_reward(state)
                    player_sum_list.append(state.player_sum)
                    dealer_card_list.append(state.dealer_card)
                    usable_ace_list.append(state.usable_ace)
                    rewards_list.append(reward)
                    break
                else:
                    reward = self._get_reward(state)
                    player_sum_list.append(state.player_sum)
                    dealer_card_list.append(state.dealer_card)
                    usable_ace_list.append(state.usable_ace)
                    rewards_list.append(0.0)
            elif action == stick:
                while True:
                    dealer_action = self._get_dealer_action()
                    if dealer_action == hit:
                        card = int(np.random.randint(1, 14, size=1)[0])
                        self.othoer_dealer_cards.append(card)
                        if self._check_if_stop(state):
                            reward = self._get_reward(state)
                            player_sum_list.append(state.player_sum)
                            dealer_card_list.append(state.dealer_card)
                            usable_ace_list.append(state.usable_ace)
                            rewards_list.append(reward)
                            break
                    else:
                        reward = self._get_reward(state)
                        player_sum_list.append(state.player_sum)
                        dealer_card_list.append(state.dealer_card)
                        usable_ace_list.append(state.usable_ace)
                        rewards_list.append(reward)
                        break
                break
        return player_sum_list, dealer_card_list, usable_ace_list, rewards_list


    def _get_dealer_action(self):
        if self.dealer_sum < 17:
            return hit
        else:
            return stick

    @property
    def dealer_sum(self):
        usable = True
        if self.init_state.dealer_card == 1:
            dealer_sum = 11
        elif self.init_state.dealer_card >= 10:
            dealer_sum = 10
        else:
            dealer_sum = self.init_state.dealer_card
        for card in self.othoer_dealer_cards:
            if card == 1:
                if dealer_sum + 11 <= 21:
                    dealer_sum += 11
                else:
                    dealer_sum += 1
            elif card >= 10:
                dealer_sum += 10
            else:
                dealer_sum += card
            if dealer_sum > 21 and usable:
                dealer_sum -= 10
                usable = False
        return dealer_sum
    
    def _check_if_stop(self, state: State):
        if state.player_sum > 21 or self.dealer_sum > 21:
            return True
        return False

    def _get_reward(self, state: State):
        dealer_sum = self.dealer_sum
        if state.player_sum > 21:
            return -1.0
        if self.dealer_sum > 21:
            return 1.0
        if state.player_sum == dealer_sum:
            return 0.0
        return 1.0 if state.player_sum > dealer_sum else -1.0
    

class EnvTorchImpl:
    def __init__(self, env_num, device):
        first_card = torch.randint(1, 14, size=(env_num,), device=device)
        first_card = torch.where(first_card > 10, torch.full_like(first_card, 10), first_card)
        m_eq_1 = torch.where(first_card == 1, torch.ones(size=first_card.size(), device=device, dtype=torch.int), torch.zeros(size=first_card.size(), device=device, dtype=torch.int))
        self.sum = torch.where(first_card == 1, torch.full_like(first_card, 11), first_card)
        self.has_ace = m_eq_1
        self.usable_ace = m_eq_1
    def add_card(self, mark_stop: torch.Tensor):
        card = torch.randint(1, 14, size=self.sum.size(), device=self.sum.device)
        card = card * (1 - mark_stop)
        card = torch.where(card > 10, torch.full_like(card, 10), card)
        m_eq_1 = torch.where(card == 1, torch.ones(size=card.size(), device=self.sum.device, dtype=torch.int), torch.zeros(size=card.size(), device=self.sum.device, dtype=torch.int))
        self.has_ace = torch.clamp(self.has_ace + m_eq_1, max=1)
        self.sum += card
        m_below_11 = torch.where(self.sum<=11, torch.ones(size=self.sum.size(), device=self.sum.device, dtype=torch.int), torch.zeros(size=self.sum.size(), device=self.sum.device, dtype=torch.int))
        m_add_10 = self.has_ace * (1-self.usable_ace) * m_below_11
        self.sum += m_add_10 * 10
        self.usable_ace += m_add_10

class DealerPolicy():
    def action(self, dealer_sum: torch.Tensor):
        mask = dealer_sum >= 17
        return torch.where(mask, torch.zeros(size=dealer_sum.size(), device=dealer_sum.device, dtype=torch.int), torch.ones(size=dealer_sum.size(), device=dealer_sum.device, dtype=torch.int))

class SimplePolicy(PolicyTorch):
    def action(self, player_sum: torch.Tensor, dealer_card: torch.Tensor, usable_ace: torch.Tensor):
        mask = player_sum >= 20
        return torch.where(mask, torch.zeros(size=player_sum.size(), device=player_sum.device, dtype=torch.int), torch.ones(size=player_sum.size(), device=player_sum.device, dtype=torch.int))

class EnvTorch:
    def __init__(self, env_num, player_policy: PolicyTorch, device, gamma=1.0):
        self.gamma = gamma
        self.env_num = env_num
        self.player = EnvTorchImpl(env_num, device)
        self.dealer = EnvTorchImpl(env_num, device)
        self.dealer_card = [self.dealer.sum.clone()]
        self.device = device
        self.player.add_card(torch.zeros(size=(self.env_num,), device=device, dtype=torch.int))
        self.player_sum = [self.player.sum.clone()]
        self.usable_ace = [self.player.usable_ace.clone()]
        self.device = device
        self.dealer.add_card(torch.zeros(size=(self.env_num,), device=device, dtype=torch.int))
        self.dealer_policy = DealerPolicy()
        self.player_policy = player_policy
    
    def _step_dealer(self):
        policy = self.dealer_policy.action(self.dealer.sum)
        if torch.sum(policy) == 0:
            return False
        self.dealer.add_card(1-policy)
        return True

    def _step_player(self, i):
        if_continue = torch.where(self.player.sum < 21, torch.ones(size=self.player_sum[0].size(), device=self.device, dtype=torch.int), torch.zeros(size=self.player_sum[0].size(), device=self.device, dtype=torch.int))
        policy = self.player_policy.action(self.player.sum, self.dealer_card[0], self.player.usable_ace)
        policy = policy * if_continue
        m_not_stop = torch.where(self.stop_time==0, torch.ones(size=self.stop_time.size(), device=self.device, dtype=torch.int), torch.zeros(size=self.stop_time.size(), device=self.device, dtype=torch.int))
        self.stop_time += (1-policy)*m_not_stop*i
        if torch.sum(policy) == 0:
            return False
        self.player.add_card(1-policy)
        self.player_sum.append(self.player.sum.clone())
        self.usable_ace.append(self.player.usable_ace.clone())
        return True
    
    def _get_reward(self):
        reward = torch.zeros(size=(self.env_num,), device=self.device, dtype=torch.float)
        m1 = torch.where(self.dealer.sum > 21, torch.ones(size=(self.env_num,), device=self.device, dtype=torch.int), torch.zeros(size=(self.env_num,), device=self.device, dtype=torch.int))
        m2 = torch.where(self.player.sum > 21, torch.ones(size=(self.env_num,), device=self.device, dtype=torch.int), torch.zeros(size=(self.env_num,), device=self.device, dtype=torch.int))
        m = torch.clamp(m1 + m2, max=1)
        reward = torch.where(self.dealer.sum > 21, torch.full_like(reward, 1.0), reward)
        reward = torch.where(self.player.sum > 21, torch.full_like(reward, -1.0), reward)
        m_gt = (1-m)*torch.where(self.player.sum > self.dealer.sum, torch.ones(size=reward.size(), device=self.device, dtype=torch.int), torch.zeros(size=reward.size(), device=self.device, dtype=torch.int))
        reward = torch.where(m_gt==1, torch.full_like(reward, 1.0), reward)
        m_lt = (1-m)*torch.where(self.player.sum < self.dealer.sum, torch.ones(size=reward.size(), device=self.device, dtype=torch.int), torch.zeros(size=reward.size(), device=self.device, dtype=torch.int))
        reward = torch.where(m_lt==1, torch.full_like(reward, -1.0), reward)
        return reward
    
    def run(self):
        i = 1
        self.stop_time = torch.zeros(size=(self.env_num,), device=self.device, dtype=torch.int)
        while self._step_player(i):
            i += 1
        while self._step_dealer():
            pass
        reward = self._get_reward()
        returns = [reward.clone() for _ in range(i)]
        V = [[] for _ in range(200)]
        for t in range(i-2, -1, -1):
            m = torch.where(self.stop_time>t, self.gamma * torch.ones(size=self.stop_time.size(), device=self.device, dtype=torch.int), torch.ones(size=self.stop_time.size(), device=self.device, dtype=torch.int))
            returns[t] = returns[t+1] * m
        for t in range(i):
            player_sum = self.player_sum[t].to("cpu")
            dealer_card = torch.where(self.dealer_card[0] > 10, torch.full_like(self.dealer_card[0], 1), self.dealer_card[0]).to("cpu")
            usable_ace = self.usable_ace[t].to("cpu")
            for j in range(self.env_num):
                if self.stop_time[j] >= t + 1:
                    if(player_sum[j] >= 12 and player_sum[j] <= 21):
                        try:
                            V[int(player_sum[j]-12)+10*int(dealer_card[j]-1)+100*int(usable_ace[j])].append(returns[t][j].item())
                        except:
                            print(player_sum[j], dealer_card[j], usable_ace[j], returns[t][j])
        values = [sum(v)/len(v) if len(v)>0 else 0.0 for v in V]
        return values
    