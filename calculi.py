# Import libraries
from imports import *

# Generate random well-formed-formulas
def rd_f(t_nu):
    return randint(1,t_nu)

# Define numerical values for symbols:
symb = {"DE": t_nu + 1,                                     # DERIVES
        "LB": t_nu + 2,                                     # LEFT BRACKET
        "RB": t_nu + 3,                                     # RIGHT BRACKET
        "NO": t_nu + 4,                                     # NOT
        "TH": t_nu + 5,                                     # THEN
        "OR": t_nu + 6,                                     # OR
        "AN": t_nu + 7,                                     # AND
        "FA": t_nu + 8}                                     # FALSUM

symb_reverse = {0: "",
        1: "p",
        2: "q",
        3: "r",
        4: "s",
        5: "t",
        t_nu + 1: "⊢",                                     # DERIVES
        t_nu + 2: "(",                                     # LEFT BRACKET
        t_nu + 3: ")",                                     # RIGHT BRACKET
        t_nu + 4: "¬",                                     # NOT
        t_nu + 5: "→",                                     # THEN
        t_nu + 6: "∨",                                     # OR
        t_nu + 7: "∧",                                     # AND
        t_nu + 8: "⊥"}                                     # FALSUM

# Function for recursive generation of well-formed formulas
def gen_wff(form, depth, max=1):
    if depth > max or random() < 0.6:
        return form
    else:
        rule = choice(wff_rules)
        subform = rule(form, depth)
        return subform

# Rules for generating well formed formulas
def cona(form1, depth):                                     # CONJUNCTION A
    return [form1, symb["AN"], gen_wff(randint(1, t_nu), depth + 1)]

def conb(form1, depth):                                     # CONJUNCTION B
    return [gen_wff(randint(1, t_nu), depth + 1), symb["AN"], form1]

def disa(form1, depth):                                     # DISJUNCTION A
    return [form1, symb["OR"], gen_wff(randint(1, t_nu), depth + 1)]

def disb(form1, depth):                                     # DISJUNCTION B
    return [gen_wff(randint(1, t_nu), depth + 1), symb["OR"], form1]

def th_a(form1, depth):                                     # IMPLICATION A
    return [form1, symb["TH"], gen_wff(randint(1, t_nu), depth + 1)]

def th_b(form1, depth):                                     # IMPLICATION B
    return [gen_wff(randint(1, t_nu), depth + 1), symb["TH"], form1]

def neg(form1, depth):                                      # NEGATION
    return [symb["NO"], form1]

wff_rules = [cona, conb, disa, disb, th_a, th_b, neg]

# Definition of rules of intuitionistic propositional logic (IPL):
def fa_e(prem):                                             # FALSUM ELIMINATION
    return rd_f(t_nu)

def no_e(prem):                                             # NEGATION ELIMINATION
    return symb["FA"]

def n_ia(prem):                                             # NEGATION INTRODUCTION A
    return [symb["NO"]] + [prem[1]]

def n_ib(prem):                                             # NEGATION INTRODUCTION B
    return [symb["NO"]] + [prem[0]]

def an_i(prem):                                             # CONJUNCTION INTRODUCTION
    return [prem[0]] + [symb["AN"]] + [prem[1]]

def a_ea(prem):                                             # CONJUNCTION ELIMINATION A
    return prem[0][0]

def a_eb(prem):                                             # CONJUNCTION ELIMINATION B
    return prem[0][2]

def t_ea(prem):                                             # IMPLICATION ELIMINATION A
    return prem[0][2]

def t_eb(prem):                                             # IMPLICATION ELIMINATION B
    return prem[1][2]

def th_i(prem):                                             # IMPLICATION INTRODUCTION
    return [prem[0]] + [symb["TH"]] + [prem[1]]

def o_ia(prem):                                             # DISJUNCTION INTRODUCTION A
    return [prem[0]] + [symb["OR"]] + [rd_f(t_nu)]

def o_ib(prem):                                             # DISJUNCTION INTRODUCTION B
    return [rd_f(t_nu)] + [symb["OR"]] + [prem[0]]

# Classical propositional logic (CPL) contains in addition:
def d_ne(prem):                                             # DOUBLE NEGATION
    return prem[2]

# List the rules of IPL
ipl = [fa_e, no_e, n_ia, n_ib, an_i, a_ea,
       a_eb, t_ea, t_eb, th_i, o_ia, o_ib]

# List the rules of CPL
cl = [fa_e, no_e, n_ia, n_ib, an_i, a_ea,
       a_eb, t_ea, t_eb, th_i, o_ia, o_ib, d_ne]

# Define check for applicability of rule to premises via syntactic conditions:
def check(rule, prem = list):
    if len(prem) == 1:
        if rule == fa_e:                                    # FALSUM ELIMINATION
            if symb["FA"] == prem[0]:
                return True
        if rule == a_ea or rule == a_eb:                    # CONJUNCTION ELIMINATION A and B
            if not isinstance(prem[0], int):
                if len(prem[0]) == 3:
                    if prem[0][1] == symb["AN"]:
                        return True
        if rule == o_ia or rule == o_ib:                    # DISJUNCTION INTRODUCTION A and B
            return True
        if rule == d_ne:
            if not isinstance(prem[0], int):
                if len(prem[0]) == 3:                           # DOUBLE NEGATION
                    if prem[0][0] == symb["NO"]:
                        if prem[0][1] == symb["NO"]:
                            return True
    if len(prem) == 2:
        if symb["FA"] == prem[0] or symb["FA"] == prem[1]:
            if rule == n_ia:                                # NEGATION INTRODUCTION A
                if prem[0] == symb["FA"]:
                    if prem[1] != symb["FA"]:
                        return True
            if rule == n_ib:                                # NEGATION INTRODUCTION B
                if prem[1] == symb["FA"]:
                    if prem[0] != symb["FA"]:
                        return True
        else:
            if rule == no_e:                                # NEGATION ELIMINATION
                if not isinstance(prem[0], int):
                    if len(prem[0]) == 2:
                        if isinstance(prem[1],int):
                            if prem[0][0] == symb["NO"]:
                                if prem[0][1] == prem[1]:
                                    return True
                if not isinstance(prem[1], int):
                    if len(prem[1]) == 2:
                        if isinstance(prem[0], int):
                            if prem[1][0] == symb["NO"]:
                                if prem[1][1] == prem[0]:
                                    return True
            if rule == an_i:                                # CONJUNCTION INTRODUCTION
                return True
            if rule == t_ea:                                # IMPLICATION ELIMINATION A
                if not isinstance(prem[0], int):
                    if len(prem[0]) == 3:
                        if isinstance(prem[1], int):
                            if prem[0][1] == symb["TH"]:
                                if prem[0][0] == prem[1]:
                                    return True
            if rule == t_eb:                                # IMPLICATION ELIMINATION B
                if not isinstance(prem[1], int):
                    if len(prem[1]) == 3:
                        if prem[0] == int:
                            if prem[1][1] == symb["TH"]:
                                if prem[1][0] == prem[0]:
                                    return True
            if rule == th_i:                                # IMPLICATION INTRODUCTION
                return True