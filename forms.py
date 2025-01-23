# Import necessary libraries and modules
from imports import *
import alphabet

# Function to generate a random propositional variable
# t_nu: Upper limit for random integer generation
def rd_f(t_nu):
    return randint(1, t_nu)  # Randomly selects a number between 1 and t_nu as a propositional variable

# Function to generate a random well-formed formula (wff)
# form: Current formula being constructed
# depth: Current depth of nested sub-formulas
# max_depth: Maximum allowed depth of recursion to prevent overly complex structures
# t_nu: Upper limit for generating random propositional variables
def gen_wff(form=None, depth=0, max_depth=3, t_nu=5):
    # Base case: if maximum depth is reached or randomly chosen to stop
    if depth >= max_depth or random() < 0.6:
        # If no formula is provided, generate a new propositional variable
        if form is None:
            return rd_f(t_nu)
        else:
            return form  # Return the current formula if it exists
    else:
        # Generate a new sub-formula recursively if none exists
        if form is None:
            form = gen_wff(depth=depth + 1, max_depth=max_depth)
        # Randomly choose a rule to construct the well-formed formula
        rule = choice(wff_rules)
        subform = rule(form, depth)  # Apply the chosen rule
        return subform  # Return the constructed sub-formula

# Rules for generating well-formed formulas using logical operations

# Conjunction rule (A)
# Combines the current formula with another randomly generated sub-formula using 'AND' (AN)
def cona(form1, depth):
    return [form1, alphabet.symb["AN"], gen_wff(None, depth + 1)]

# Conjunction rule (B)
# Combines a new random sub-formula with the current formula using 'AND' (AN)
def conb(form1, depth):
    return [gen_wff(None, depth + 1), alphabet.symb["AN"], form1]

# Disjunction rule (A)
# Combines the current formula with another random sub-formula using 'OR'
def disa(form1, depth):
    return [form1, alphabet.symb["OR"], gen_wff(None, depth + 1)]

# Disjunction rule (B)
# Combines a new random sub-formula with the current formula using 'OR'
def disb(form1, depth):
    return [gen_wff(None, depth + 1), alphabet.symb["OR"], form1]

# Implication rule (A)
# Creates an implication from the current formula to a new random sub-formula
def th_a(form1, depth):
    return [form1, alphabet.symb["TH"], gen_wff(None, depth + 1)]

# Implication rule (B)
# Creates an implication from a new random sub-formula to the current formula
def th_b(form1, depth):
    return [gen_wff(None, depth + 1), alphabet.symb["TH"], form1]

# Negation rule
# Negates the current formula
def neg(form1, depth):
    return [alphabet.symb["NO"], form1]

# A list of rules to randomly choose from when generating well-formed formulas
wff_rules = [cona, conb, disa, disb, th_a, th_b, neg]