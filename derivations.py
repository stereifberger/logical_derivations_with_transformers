# Import libraries
from imports import *
import forms
import alphabet

# Derivation rules for IPL (Intuitionistic Propositional Logic)
def fa_e(prem):  # Falsum Elimination: From ⊥, derive any formula
    return forms.gen_wff(form=None, depth=0)
def no_e(prem):  # Negation Elimination: From ¬A and A, derive ⊥
    return alphabet.symb["FA"]
def n_ia(premises):  # Negation Introduction
    # From A ⊢ ⊥ infer ¬A
    return [alphabet.symb["NO"], premises[0]]
def an_i(prem):  # Conjunction Introduction: From A and B, derive A ∧ B
    return [prem[0], alphabet.symb["AN"], prem[1]]
def a_ea(prem):  # Conjunction Elimination A: From A ∧ B, derive A
    return prem[0][0]
def a_eb(prem):  # Conjunction Elimination B: From A ∧ B, derive B
    return prem[0][2]
def t_ea(prem):  # Implication Elimination A (Modus Ponens): From A and A → B, derive B
    return prem[1][2]
def t_eb(prem):  # Implication Elimination B: From A → B and A, derive B
    return prem[0][2]
def th_i(prem):  # Implication Introduction: From assumption A to derive B, infer A → B
    return [prem[0], alphabet.symb["TH"], prem[1]]
def o_ia(prem):  # Disjunction Introduction A: From A, derive A ∨ B
    return [prem[0], alphabet.symb["OR"], forms.gen_wff(form=None, depth=0)]
def o_ib(prem):  # Disjunction Introduction B: From A, derive B ∨ A
    return [forms.gen_wff(form=None, depth=0), alphabet.symb["OR"], prem[0]]
# Additional rule for Classical Propositional Logic (CPL)
def d_ne(prem):  # Double Negation Elimination: From ¬¬A, derive A
    return prem[0][1]

# List of rules for IPL and CPL
ipl_rules = [fa_e, no_e, n_ia, an_i, a_ea, a_eb, t_ea, t_eb, th_i, o_ia, o_ib]
cpl_rules = ipl_rules + [d_ne]

# Helper functions
def is_negation(f):
    return isinstance(f, list) and len(f) == 2 and f[0] == alphabet.symb["NO"]
def is_implication(f):
    return isinstance(f, list) and len(f) == 3 and f[1] == alphabet.symb["TH"]
def is_conjunction(f):
    return isinstance(f, list) and len(f) == 3 and f[1] == alphabet.symb["AN"]
def is_double_negation(f):
    return is_negation(f) and is_negation(f[1])

# Function to check applicability of a rule to premises
def check(rule, prem):
    if len(prem) == 1:
        if rule == fa_e:  # Falsum Elimination
            if prem[0] == alphabet.symb["FA"]:
                return True
        if rule in [a_ea, a_eb]:  # Conjunction Elimination
            if is_conjunction(prem[0]):
                return True
        if rule in [o_ia, o_ib]:  # Disjunction Introduction
            return True
        if rule == d_ne:  # Double Negation Elimination
            if is_double_negation(prem[0]):
                return True
    elif len(prem) == 2:
        if rule == n_ia:  # Negation Introduction
            # From A ⊢ ⊥ infer ¬A. Here we assume prem[1] is ⊥
            if prem[1] == alphabet.symb["FA"]:
                return True
        elif rule == no_e:  # Negation Elimination
            if (is_negation(prem[0]) and prem[0][1] == prem[1]) or (is_negation(prem[1]) and prem[1][1] == prem[0]):
                return True
        elif rule == an_i:  # Conjunction Introduction
            return True
        elif rule == t_ea:  # Implication Elimination A
            if is_implication(prem[1]) and prem[1][0] == prem[0]:
                return True
        elif rule == t_eb:  # Implication Elimination B
            if is_implication(prem[0]) and prem[0][0] == prem[1]:
                return True
        elif rule == th_i:  # Implication Introduction
            # From assumption prem[0] to derive prem[1], infer prem[0] → prem[1]
            return True
    return False

# Function to get applicable rules for given premises
def get_applicable_rules(premises, rules):
    applicable_rules = []
    for rule in rules:
        if check(rule, premises):
            applicable_rules.append(rule)
    return applicable_rules

# Function to generate a derivation
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

        # Get applicable rules
        applicable_rules = get_applicable_rules(selected_premises, rules)
        if applicable_rules:
            rule = choice(applicable_rules)
            # Apply the rule
            new_formula = rule(selected_premises)
            formulas.append(new_formula)
            derivation_steps.append({'premises': selected_premises, 'conclusion': new_formula, 'rule': rule.__name__})
        else:
            # No applicable rules, stop derivation
            break
    premises = [step['conclusion'] for step in derivation_steps if step['rule'] == 'Premise']
    conclusion = derivation_steps[-1]['conclusion']
    return premises, conclusion, derivation_steps