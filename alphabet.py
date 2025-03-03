# Import libraries
from imports import *

# Number of propositional variables
t_nu = 5  # Variables: p, q, r, s, t defined as numbers

# Define numerical values for symbols
symb = {
    "DE": t_nu + 1,  # ⊢ DERIVES (not used in formula generation)
    "LB": t_nu + 2,  # (
    "RB": t_nu + 3,  # )
    "NO": t_nu + 4,  # ¬
    "TH": t_nu + 5,  # →
    "OR": t_nu + 6,  # ∨
    "AN": t_nu + 7,  # ∧
    "FA": t_nu + 8,  # ⊥
    "PAD": 0,        # Padding token
    "SOS": t_nu + 9, # Start of sequence
    "EOS": t_nu + 10 # End of sequence
}

# Reverse mapping from numerical values to symbols
symb_reverse = {0: ""}
for i in range(1, t_nu + 1):
    symb_reverse[i] = chr(ord('p') + i - 1)
symb_reverse.update({
    t_nu + 1: "⊢",
    t_nu + 2: "(",
    t_nu + 3: ")",
    t_nu + 4: "¬",
    t_nu + 5: "→",
    t_nu + 6: "∨",
    t_nu + 7: "∧",
    t_nu + 8: "⊥",
    t_nu + 9: "<SOS>",
    t_nu + 10: "<EOS>"
})

# Mapping from symbols to numerical values
symb_map = {v: k for k, v in symb_reverse.items()}