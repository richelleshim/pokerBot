
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from deuces import Card, Evaluator, Deck

class PokerBotNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(PokerBotNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    # def forward(self, x):
    #     x = torch.relu(self.fc1(x))
    #     x = torch.relu(self.fc2(x))
    #     return self.fc3(x)
        


        # adding softmax layer instead since wtf 

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class AdvancedPokerBot:
    def __init__(self, name, bankroll=1000):
        self.name = name
        self.bankroll = bankroll
        self.hand = []
        self.model = PokerBotNN(input_size=10, output_size=3).to(self.get_device())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.evaluator = Evaluator()

    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def collect_game_data(self, num_games=1000):
        """
        Collect game data for training.
        """
        game_data = []
        for _ in range(num_games):
            deck = Deck()
            self.hand = [deck.draw(), deck.draw()]
            community_cards = [deck.draw() for _ in range(5)]
            hand_strength = self.evaluate_hand_strength(community_cards)
            input_features = self.get_features(community_cards, pot_size=100, hand_strength=hand_strength)
            action, _ = self.make_decision(community_cards, pot_size=100)
            action_index = {"fold": 0, "call": 1, "raise": 2}[action]

            reward = 10 if hand_strength > 1 else -10  # Simplistic reward system

            next_features = self.get_features(community_cards, pot_size=200, hand_strength=hand_strength)
            game_data.append((input_features, action_index, reward, next_features))
        return game_data

    def evaluate_hand_strength(self, community_cards):
        """
        Use Deuces to evaluate hand strength and normalize.
        """
        raw_score = self.evaluator.evaluate(self.hand, community_cards)
        max_score = 7462 
        normalized_score = (max_score - raw_score) / max_score  # Normalize to [0, 1]
        return normalized_score

    def make_decision(self, community_cards, pot_size):
        hand_strength = self.evaluate_hand_strength(community_cards)
        input_features = self.get_features(community_cards, pot_size, hand_strength)
        input_tensor = torch.tensor(input_features, dtype=torch.float32).to(self.get_device())

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = output.tolist()

        action = torch.argmax(output).item()

        # Avoid folding strong hands
        if action == 0 and hand_strength > 0.5:
            action = 1  # Call instead of folding strong hands

        print(f"Normalized hand strength: {hand_strength}")
        print(f"NN Input Features: {input_features}")
        print(f"NN Probabilities: {probabilities}")
        print(f"Chosen Action: {action}")

        if action == 0:
            return "fold", 0
        elif action == 1:
            return "call", min(self.bankroll, pot_size)
        elif action == 2:
            return "raise", min(self.bankroll, pot_size * 2)

    # def make_decision(self, community_cards, pot_size):
    #     """
    #     Use the trained neural network to make a decision.
    #     """
    #     hand_strength = self.evaluate_hand_strength(community_cards)
    #     input_features = self.get_features(community_cards, pot_size, hand_strength)
    #     input_tensor = torch.tensor(input_features, dtype=torch.float32).to(self.get_device())

    #     with torch.no_grad():
    #         output = self.model(input_tensor)
    #         action = torch.argmax(output).item()

    #     # Debugging 
    #     print(f"Hand strength: {hand_strength}")
    #     print(f"Input features: {input_features}")
    #     print(f"NN Output: {output.tolist()}")

    #     if action == 0:
    #         return "fold", 0
    #     elif action == 1:
    #         return "call", min(self.bankroll, pot_size)
    #     elif action == 2:
    #         return "raise", min(self.bankroll, pot_size * 2)


    def get_features(self, community_cards, pot_size, hand_strength):
        """
        Generate input features for the neural network.
        """
        features = [
            hand_strength,  # Calculated hand strength
            len(community_cards),  # Number of community cards
            pot_size,  # Current pot size
            self.bankroll,  # Remaining bankroll
            pot_size / self.bankroll if self.bankroll > 0 else 0,  # Pot odds
            random.random(),  # Placeholder for opponent modeling
        ]
        return features + [0] * (10 - len(features))  # Pad to match input size

    def train(self, game_data):
        """
        Train the model using past game data.
        """
        for state, action, reward, next_state in game_data:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.get_device())
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.get_device())
            action_tensor = torch.tensor(action, dtype=torch.long).to(self.get_device())
            reward_tensor = torch.tensor(reward, dtype=torch.float32).to(self.get_device())

            q_values = self.model(state_tensor)
            next_q_values = self.model(next_state_tensor)

            target = q_values.clone()
            target[action] = reward + 0.99 * torch.max(next_q_values).item()

            loss = self.loss_fn(q_values, target.detach())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Simulation function

def simulate_game():
    bot = AdvancedPokerBot(name="RichelleBot")
    deck = Deck()
    bot.hand = [deck.draw(), deck.draw()]
    community_cards = [deck.draw() for _ in range(5)]

    pot_size = 100

    action, amount = bot.make_decision(community_cards, pot_size)

    print(f"Bot's hand:")
    Card.print_pretty_cards(bot.hand)  
    print(f"\nCommunity cards:")
    Card.print_pretty_cards(community_cards)
    print(f"\nBot decides to {action} with amount {amount}")

simulate_game()
