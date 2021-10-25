import enum

int_to_jcards = {
    0: "ace:spades",      1: "ace:clubs",       2: "ace:diamonds",        3: "ace:hearts",
    4: "two:spades",      5: "two:clubs",       6: "two:diamonds",        7: "two:hearts",
    8: "three:spades",    9: "three:clubs",     10: "three:diamonds",     11: "three:hearts",
    12: "four:spades",    13: "four:clubs",     14: "four:diamonds",      15: "four:hearts",
    16: "five:spades",    17: "five:clubs",     18: "five:diamonds",      19: "five:hearts",
    20: "six:spades",     21: "six:clubs",      22: "six:diamonds",       23: "six:hearts",
    24: "seven:spades",   25: "seven:clubs",    26: "seven:diamonds",     27: "seven:hearts",
    28: "eight:spades",   29: "eight:clubs",    30: "eight:diamonds",     31: "eight:hearts",
    32: "nine:spades",    33: "nine:clubs",     34: "nine:diamonds",      35: "nine:hearts",
    36: "ten:spades",     37: "ten:clubs",      38: "ten:diamonds",       39: "ten:hearts",
    40: "jack:spades",    41: "jack:clubs",     42: "jack:diamonds",      43: "jack:hearts",
    44: "queen:spades",   45: "queen:clubs",    46: "queen:diamonds",     47: "queen:hearts",
    48: "king:spades",    49: "king:clubs",     50: "king:diamonds",      51: "king:hearts"
}

jcards_to_int = { int_to_jcards[x]:x for x in int_to_jcards }

ranks = {
    "two" : 2,
    "three" : 3,
    "four" : 4,
    "five" : 5,
    "six" : 6,
    "seven" : 7,
    "eight" : 8,
    "nine" : 9,
    "ten" : 10,
    "jack" : 10,
    "queen" : 10,
    "king" : 10,
    "ace" : (1, 11)
}

class Suit(enum.Enum):
    spades = "spades"
    clubs = "clubs"
    diamonds = "diamonds"
    hearts = "hearts"

class Card:
    def __init__(self, suit, rank, value):
        self.suit = suit
        self.rank = rank
        self.value = value

    def name(self):
        return self.rank + ":" + self.suit.value
        
    def __str__(self):
        return self.rank + " of " + self.suit.value
