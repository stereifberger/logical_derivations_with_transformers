# Import libraries
from imports import *
import alphabet

# Function to convert a formula to a string
def formula_to_string(f):
    if isinstance(f, int):
        return alphabet.symb_reverse[f]
    elif isinstance(f, list):
        if len(f) == 2 and f[0] == alphabet.symb["NO"]:  # Negation
            return alphabet.symb_reverse[alphabet.symb["NO"]] + formula_to_string(f[1])
        elif len(f) == 3:
            left = formula_to_string(f[0])
            op = alphabet.symb_reverse[f[1]]
            right = formula_to_string(f[2])
            return f'({left} {op} {right})'
        else:
            return ''.join(formula_to_string(subf) for subf in f)
    else:
        return str(f)

# Function to flatten formulas into tokens
def flatten_formula(f):
    if isinstance(f, int):
        return [f]
    elif isinstance(f, list):
        tokens = []
        if len(f) == 2 and f[0] == alphabet.symb["NO"]:  # Negation
            tokens.append(alphabet.symb["NO"])
            tokens.extend(flatten_formula(f[1]))
        elif len(f) == 3:
            tokens.append(alphabet.symb["LB"])
            tokens.extend(flatten_formula(f[0]))
            tokens.append(f[1])  # Operator
            tokens.extend(flatten_formula(f[2]))
            tokens.append(alphabet.symb["RB"])
        else:
            for subf in f:
                tokens.extend(flatten_formula(subf))
        return tokens
    else:
        return []

# Prepare data for the transformer model
def prepare_data(num_samples=1000):
    data = []
    for _ in range(num_samples):
        premises, conclusion, derivation_steps = derivations.generate_derivation(derivations.ipl_rules)
        if derivation_steps:
            input_tokens = []
            for premise in premises:
                input_tokens.extend(flatten_formula(premise))
#                input_tokens.append(symb["DE"])  # Separator between premises
            input_tokens.append(alphabet.symb["DE"])  # Separator before conclusion
            input_tokens.extend(flatten_formula(conclusion))
            input_tokens.append(alphabet.symb["EOS"])  # End of sequence token

            target_tokens = []
            for step in derivation_steps:
                target_tokens.extend(flatten_formula(step['conclusion']))
                #target_tokens.append(symb["DE"])  # Separator between derivation steps
            target_tokens.append(alphabet.symb["EOS"])  # End of sequence token

            data.append((input_tokens, target_tokens))

    # Find maximum lengths
    max_input_len = max(len(pair[0]) for pair in data)
    max_target_len = max(len(pair[1]) for pair in data)

    # Pad sequences to the same length and create tensors
    padded_inputs = []
    padded_targets = []
    for input_seq, target_seq in data:
        input_seq = input_seq + [alphabet.symb["PAD"]] * (max_input_len - len(input_seq))
        target_seq = [alphabet.symb["SOS"]] + target_seq + [alphabet.symb["PAD"]] * (max_target_len - len(target_seq))
        padded_inputs.append(input_seq)
        padded_targets.append(target_seq)

    return padded_inputs, padded_targets, max_input_len, max_target_len + 1  # +1 for SOS

def reconstruct_formula(tokens):
    def helper(index):
        if index >= len(tokens):
            return None, index
        tok = tokens[index]
        if tok in alphabet.symb_reverse:
            if tok == alphabet.symb["LB"]:
                left_formula, next_index = helper(index + 1)
                if next_index >= len(tokens):  # Check if we run out of tokens
                    return None, next_index
                op = tokens[next_index]
                right_formula, next_index = helper(next_index + 1)
                if next_index >= len(tokens) or tokens[next_index] != alphabet.symb["RB"]:  # Extra check for RB
                    return None, next_index  # Instead of assertion, return None
                return [left_formula, op, right_formula], next_index + 1
            elif tok == alphabet.symb["NO"]:
                formula, next_index = helper(index + 1)
                return [alphabet.symb["NO"], formula], next_index
            elif tok == alphabet.symb["RB"] or tok == alphabet.symb["DE"] or tok == alphabet.symb["EOS"]:
                return None, index
            else:
                return tok, index + 1
        else:
            return None, index + 1

    formula, _ = helper(0)
    return formula