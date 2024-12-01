# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim

# class PokerBotNN(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(PokerBotNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, output_size)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)

# class AdvancedPokerBot:
#     def __init__(self, name, bankroll=1000):
#         self.name = name
#         self.bankroll = bankroll
#         self.hand = []
#         self.model = PokerBotNN(input_size=10, output_size=3) 
#         self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
#         self.loss_fn = nn.MSELoss()
#     def collect_game_data(bot, num_games=1000):
#         """
#         Collect game data for training.
#         """
#         game_data = []
#         for _ in range(num_games):
#             bot.deal_cards(random.sample(["2H", "3D", "4S", "5C", "6H", "7D", "8S", "9C", "10H", "JH", "QH", "KH", "AH"], 2))
#             community_cards = random.sample(["2H", "3D", "4S", "5C", "6H", "7D", "8S", "9C", "10H", "JH", "QH", "KH", "AH"], 5)
#             hand_strength = bot.evaluate_hand_strength(community_cards)
#             input_features = bot.get_features(community_cards, pot_size=100, hand_strength=hand_strength)
            
#             action, _ = bot.make_decision(community_cards, pot_size=100)
#             action_index = {"fold": 0, "call": 1, "raise": 2}[action]

#             reward = 10 if hand_strength > 1 else -10

#             next_features = bot.get_features(community_cards, pot_size=200, hand_strength=hand_strength)
#             game_data.append((input_features, action_index, reward, next_features))
#         return game_data



#     def deal_cards(self, deck):
#         self.hand = random.sample(deck, 2)

#     def evaluate_hand_strength(self, community_cards):
#         """
#         Improved hand strength evaluation using rules.
#         Replace with a library like Deuces for better accuracy.
#         """
#         all_cards = self.hand + community_cards
#         values = [card[:-1] for card in all_cards]
#         suits = [card[-1] for card in all_cards]

#         pair = any(values.count(value) == 2 for value in values)
#         flush = suits.count(suits[0]) >= 4  # Check for potential flush
        
#         straight = False
#         value_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
#                     '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
#         value_counts = [value_map[value] for value in values]
#         value_counts = sorted(set(value_counts))
#         for i in range(len(value_counts) - 4):
#             if value_counts[i + 4] - value_counts[i] == 4:
#                 straight = True

#         if flush:
#             return 3  
#         elif straight:
#             return 2 
#         elif pair:
#             return 1 
#         else:
#             return 0


#     def make_decision(self, community_cards, pot_size):
#         """
#         Use the trained neural network to make a decision.
#         """
#         hand_strength = self.evaluate_hand_strength(community_cards)
#         input_features = self.get_features(community_cards, pot_size, hand_strength)
#         input_tensor = torch.tensor(input_features, dtype=torch.float32)

#         with torch.no_grad():
#             output = self.model(input_tensor)
#             action = torch.argmax(output).item()

#         if action == 0:
#             return "fold", 0
#         elif action == 1:
#             return "call", min(self.bankroll, pot_size)
#         elif action == 2:
#             return "raise", min(self.bankroll, pot_size * 2)
        
        

#     def get_features(self, community_cards, pot_size, hand_strength):
#         """
#         Generate input features for the neural network.
#         """
#         features = [
#             hand_strength,  # Calculated hand strength
#             len(community_cards),  # Number of community cards
#             pot_size,  # Current pot size
#             self.bankroll,  # Remaining bankroll
#             pot_size / self.bankroll if self.bankroll > 0 else 0,  # Pot odds
#             random.random(),  # Placeholder for opponent modeling
#         ]
#         return features + [0] * (10 - len(features))  # Pad to match input size

#     def train(self, game_data):
#         """
#         Train the model using past game data.
#         """
#         for state, action, reward, next_state in game_data:
#             state_tensor = torch.tensor(state, dtype=torch.float32)
#             next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
#             action_tensor = torch.tensor(action, dtype=torch.long)
#             reward_tensor = torch.tensor(reward, dtype=torch.float32)

#             q_values = self.model(state_tensor)
#             next_q_values = self.model(next_state_tensor)

#             target = q_values.clone()
#             target[action] = reward + 0.99 * torch.max(next_q_values).item()

#             loss = self.loss_fn(q_values, target.detach())

#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.buffer = []
#         self.capacity = capacity

#     def push(self, experience):
#         if len(self.buffer) >= self.capacity:
#             self.buffer.pop(0)
#         self.buffer.append(experience)

#     def sample(self, batch_size):
#         return random.sample(self.buffer, batch_size)

# def train_with_replay(bot, game_data, replay_buffer, batch_size=32):
#     replay_buffer.extend(game_data)

#     if len(replay_buffer) < batch_size:
#         return

#     batch = replay_buffer.sample(batch_size)
#     for state, action, reward, next_state in batch:
#         state_tensor = torch.tensor(state, dtype=torch.float32)
#         next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
#         action_tensor = torch.tensor(action, dtype=torch.long)
#         reward_tensor = torch.tensor(reward, dtype=torch.float32)

#         q_values = bot.model(state_tensor)
#         next_q_values = bot.model(next_state_tensor)

#         target = q_values.clone()
#         target[action] = reward + 0.99 * torch.max(next_q_values).item()

#         loss = bot.loss_fn(q_values, target.detach())
#         bot.optimizer.zero_grad()
#         loss.backward()
#         bot.optimizer.step()


# # # Example Simulation
# # def simulate_game():
# #     community_cards = ["10H", "8H", "4D", "2S", "9C"]
# #     bot = AdvancedPokerBot(name="RichelleBot")
# #     bot.deal_cards(["KH", "QH"])

# #     print(f"Bot's hand: {bot.hand}")
# #     print(f"Community cards: {community_cards}")

# #     action, amount = bot.make_decision(community_cards, pot_size=100)
# #     print(f"Bot decides to {action} with amount {amount}")

# # simulate_game()



# def simulate_game():
#     suits = ['H', 'D', 'S', 'C']
#     ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
#     deck = [rank + suit for rank in ranks for suit in suits]

#     bot = AdvancedPokerBot(name="RichelleBot")
#     bot.deal_cards(deck)
#     community_cards = random.sample([card for card in deck if card not in bot.hand], 5)

#     pot_size = 100
#     action, amount = bot.make_decision(community_cards, pot_size)

#     print(f"Bot's hand: {bot.hand}")
#     print(f"Community cards: {community_cards}")
#     print(f"Bot decides to {action} with amount {amount}")


# simulate_game()