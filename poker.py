#sympy 


import random
class PokerBot:
    def __init__(self, name):
        self.name = name
        self.hand = []
        self.bankroll = 1000

    def deal_cards(self, deck):
        self.hand = random.sample(deck, 2)

    def evaluate_hand_strength(self, community_cards):
        """
        Improved hand strength evaluation using rules.
        Replace with a library like Deuces for better accuracy.
        """
        all_cards = self.hand + community_cards
        values = [card[:-1] for card in all_cards]
        suits = [card[-1] for card in all_cards]

        pair = any(values.count(value) == 2 for value in values)
        flush = suits.count(suits[0]) >= 4 
        
        straight = False
        value_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                    '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        value_counts = [value_map[value] for value in values]
        value_counts = sorted(set(value_counts))
        for i in range(len(value_counts) - 4):
            if value_counts[i + 4] - value_counts[i] == 4:
                straight = True

        if flush:
            return 3  
        elif straight:
            return 2 
        elif pair:
            return 1 
        else:
            return 0  

    def make_decision(self, community_cards, pot_size):
        """
        Decide to fold, call, or raise based on a basic heuristic.
        """
        hand_strength = self.evaluate_hand_strength(community_cards)
        
        if hand_strength == 3:  # Flush
            return "raise", min(self.bankroll, pot_size * 2)
        elif hand_strength == 1:  # Pair
            return "call", min(self.bankroll, pot_size // 2)
        else:
            return "fold", 0

    def __str__(self):
        return f"PokerBot({self.name}, Bankroll: {self.bankroll})"
    

# def simulate_game():
#     community_cards = ["10H", "8H", "4D", "2S", "9C"]
#     bot = PokerBot(name="RichelleBot")
#     bot.deal_cards(["KH", "QH"])

#     print(f"Bot's hand: {bot.hand}")
#     print(f"Community cards: {community_cards}")

#     action, amount = bot.make_decision(community_cards, pot_size=100)
#     print(f"Bot decides to {action} with amount {amount}")

# simulate_game()


def simulate_game():
    suits = ['H', 'D', 'S', 'C']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    deck = [rank + suit for rank in ranks for suit in suits]

    bot = PokerBot(name="RichelleBot")
    bot.deal_cards(deck)
    community_cards = random.sample([card for card in deck if card not in bot.hand], 5)

    pot_size = 100
    action, amount = bot.make_decision(community_cards, pot_size)

    print(f"Bot's hand: {bot.hand}")
    print(f"Community cards: {community_cards}")
    print(f"Bot decides to {action} with amount {amount}")


simulate_game()


