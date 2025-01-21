# Import libraries
from imports import *
import alphabet

# Function to generate a random propositional variable
def rd_f(t_nu):
    return randint(1, t_nu)

def gen_wff(form=None, depth=0, max_depth=3, t_nu=5):
    if depth >= max_depth or random() < 0.6:
        if form is None:
            return rd_f(t_nu)
        else:
            return form
    else:
        if form is None:
            form = gen_wff(depth=depth + 1, max_depth=max_depth)
        rule = choice(wff_rules)
        subform = rule(form, depth)
        return subform

# Rules for generating well-formed formulas
def cona(form1, depth):  # Conjunction A
    return [form1, alphabet.symb["AN"], gen_wff(None, depth + 1)]
def conb(form1, depth):  # Conjunction B
    return [gen_wff(None, depth + 1), alphabet.symb["AN"], form1]
def disa(form1, depth):  # Disjunction A
    return [form1, alphabet.symb["OR"], gen_wff(None, depth + 1)]
def disb(form1, depth):  # Disjunction B
    return [gen_wff(None, depth + 1), alphabet.symb["OR"], form1]
def th_a(form1, depth):  # Implication A
    return [form1, alphabet.symb["TH"], gen_wff(None, depth + 1)]
def th_b(form1, depth):  # Implication B
    return [gen_wff(None, depth + 1), alphabet.symb["TH"], form1]
def neg(form1, depth):    # Negation
    return [alphabet.symb["NO"], form1]

wff_rules = [cona, conb, disa, disb, th_a, th_b, neg]