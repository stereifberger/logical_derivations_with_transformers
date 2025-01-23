# Import necessary libraries and modules
from imports import *
import forms
import alphabet

# Functions defining derivation rules for IPL (Intuitionistic Propositional Logic)
def fa_e(prem):  # Falsum Elimination: From ⊥, infer any formula
    return forms.gen_wff(form=None, depth=0)

def no_e(prem):  # Negation Elimination: From ¬A and A, infer ⊥
    return alphabet.symb["FA"]

def n_ia(premises):  # Negation Introduction
    # From A ⊢ ⊥, infer ¬A
    return [alphabet.symb["NO"], premises[0]]

def an_i(prem):  # Conjunction Introduction: From A and B, infer A ∧ B
    return [prem[0], alphabet.symb["AN"], prem[1]]

def a_ea(prem):  # Conjunction Elimination A: From A ∧ B, infer A
    return prem[0][0]

def a_eb(prem):  # Conjunction Elimination B: From A ∧ B, infer B
    return prem[0][2]

def t_ea(prem):  # Implication Elimination A (Modus Ponens): From A and A → B, infer B
    return prem[1][2]

def t_eb(prem):  # Implication Elimination B: From A → B and A, infer B
    return prem[0][2]

def th_i(prem):  # Implication Introduction: From assumption A to derive B, infer A → B
    return [prem[0], alphabet.symb["TH"], prem[1]]

def o_ia(prem):  # Disjunction Introduction A: From A, infer A ∨ B
    return [prem[0], alphabet.symb["OR"], forms.gen_wff(form=None, depth=0)]

def o_ib(prem):  # Disjunction Introduction B: From A, infer B ∨ A
    return [forms.gen_wff(form=None, depth=0), alphabet.symb["OR"], prem[0]]

# Additional rule for Classical Propositional Logic (CPL)
def d_ne(prem):  # Double Negation Elimination: From ¬¬A, infer A
    return prem[0][1]

# List of rules categorized for IPL and CPL
ipl_rules = [fa_e, no_e, n_ia, an_i, a_ea, a_eb, t_ea, t_eb, th_i, o_ia, o_ib]
cpl_rules = ipl_rules + [d_ne]

# Helper functions to identify type of formula
def is_negation(f):
    # Identifies if given formula is a negation
    return isinstance(f, list) and len(f) == 2 and f[0] == alphabet.symb["NO"]

def is_implication(f):
    # Identifies if given formula is an implication
    return isinstance(f, list) and len(f) == 3 and f[1] == alphabet.symb["TH"]

def is_conjunction(f):
    # Identifies if given formula is a conjunction
    return isinstance(f, list) and len(f) == 3 and f[1] == alphabet.symb["AN"]

def is_double_negation(f):
    # Identifies if given formula is a double negation
    return is_negation(f) and is_negation(f[1])

# Function to check applicability of a rule to given premises
def check(rule, prem):
    if len(prem) == 1:
        if rule == fa_e:  # Check for Falsum Elimination
            if prem[0] == alphabet.symb["FA"]:
                return True
        if rule in [a_ea, a_eb]:  # Check for Conjunction Elimination
            if is_conjunction(prem[0]):
                return True
        if rule in [o_ia, o_ib]:  # Check for Disjunction Introduction
            return True
        if rule == d_ne:  # Check for Double Negation Elimination
            if is_double_negation(prem[0]):
                return True
    elif len(prem) == 2:
        if rule == n_ia:  # Check for Negation Introduction
            if prem[1] == alphabet.symb["FA"]:
                return True
        elif rule == no_e:  # Check for Negation Elimination
            if (is_negation(prem[0]) and prem[0][1] == prem[1]) or (is_negation(prem[1]) and prem[1][1] == prem[0]):
                return True
        elif rule == an_i:  # Check for Conjunction Introduction
            return True
        elif rule == t_ea:  # Check for Implication Elimination A
            if is_implication(prem[1]) and prem[1][0] == prem[0]:
                return True
        elif rule == t_eb:  # Check for Implication Elimination B
            if is_implication(prem[0]) and prem[0][0] == prem[1]:
                return True
        elif rule == th_i:  # Check for Implication Introduction
            return True
    return False

# Function to get applicable rules for given premises
def get_applicable_rules(premises, rules):
    applicable_rules = []
    for rule in rules:
        if check(rule, premises):
            applicable_rules.append(rule)
    return applicable_rules

# Function to generate a derivation with the given set of rules
def generate_derivation(rules, max_steps=2):
    formulas = []
    derivation_steps = []
    # Start with initial formulas (premises)
    num_premises = randint(1, 2)
    for _ in range(num_premises):
        formula = forms.gen_wff(form=None, depth=0)
        formulas.append(formula)
        derivation_steps.append({'premises': [], 'conclusion': formula, 'rule': 'Premise'})

    num_steps = randint(1, max_steps)
    for _ in range(num_steps):
        # Choose 1 or 2 formulas from previous formulas
        num_premises = choice([1, 2])
        if len(formulas) < num_premises:
            num_premises = len(formulas)
        selected_premises = sample(formulas, num_premises)

        # Get applicable rules for the selected premises
        applicable_rules = get_applicable_rules(selected_premises, rules)
        if applicable_rules:
            rule = choice(applicable_rules)
            # Apply the selected rule
            new_formula = rule(selected_premises)
            formulas.append(new_formula)
            derivation_steps.append({'premises': selected_premises, 'conclusion': new_formula, 'rule': rule.__name__})
        else:
            # No applicable rules, stop derivation
            break
    # Identify premises and final conclusion of the derivation
    premises = [step['conclusion'] for step in derivation_steps if step['rule'] == 'Premise']
    conclusion = derivation_steps[-1]['conclusion']
    return premises, conclusion, derivation_steps