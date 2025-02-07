# Import necessary libraries and modules
from imports import *
import alphabet  # Import the alphabet module for symbolic representation

# Function to convert a formula from tokens to a human-readable string 
def formula_to_string(f):
    # If the formula is an integer, return the corresponding symbol from the alphabet
    if isinstance(f, int):
        return alphabet.symb_reverse[f]
    # If the formula is a list, handle compound expressions
    elif isinstance(f, list):
        # Handle negation (e.g., NOT operation)
        if len(f) == 2 and f[0] == alphabet.symb["NO"]:
            return alphabet.symb_reverse[alphabet.symb["NO"]] + formula_to_string(f[1])
        # Handle binary operations, e.g., (left OP right)
        elif len(f) == 3:
            left = formula_to_string(f[0])
            op = alphabet.symb_reverse[f[1]]
            right = formula_to_string(f[2])
            return f'({left} {op} {right})'
        # Recursively handle each sub-formula in the current formula
        else:
            return ''.join(formula_to_string(subf) for subf in f)
    # If formula type is unhandled, convert to string and return
    else:
        return str(f)

# Function to flatten a formula into tokens
def flatten_formula(f):
    # If the formula is an integer (i.e., a single symbol), return it in a list
    if isinstance(f, int):
        return [f]
    # If the formula is a list, it represents a compound expression
    elif isinstance(f, list):
        tokens = []  # To store the resulting tokens
        # Handle negation by adding the negation token and flattening the formula
        if len(f) == 2 and f[0] == alphabet.symb["NO"]:
            tokens.append(alphabet.symb["NO"])
            tokens.extend(flatten_formula(f[1]))
        # Handle binary operations by flattening and enclosing components 
        elif len(f) == 3:
            tokens.append(alphabet.symb["LB"])  # Left bracket for grouping
            tokens.extend(flatten_formula(f[0]))  # Flatten left operand
            tokens.append(f[1])  # Operator token
            tokens.extend(flatten_formula(f[2]))  # Flatten right operand
            tokens.append(alphabet.symb["RB"])  # Right bracket for grouping
        # Handle each part of a complex formula by recursively flattening
        else:
            for subf in f:
                tokens.extend(flatten_formula(subf))
        return tokens
    # If formula type is unhandled, return an empty list
    else:
        return []

# Function to prepare data for training a transformer model
def prepare_data(num_samples=1000):
    data = []
    # Generate a specified number of derivation samples
    for _ in range(num_samples):
        premises, conclusion, derivation_steps = derivations.generate_derivation(derivations.ipl_rules)
        # Only proceed if derivation steps exist
        if derivation_steps:
            input_tokens = []  # To store tokens for input
            # Flatten and combine premise tokens, adding separator token between
            for premise in premises:
                input_tokens.extend(flatten_formula(premise))
            input_tokens.append(alphabet.symb["DE"])  # Separator before the conclusion
            input_tokens.extend(flatten_formula(conclusion))

            target_tokens = []  # To store tokens for target/derivation steps
            for step in derivation_steps:
                target_tokens.extend(flatten_formula(step['conclusion']))

            # Add tuple of input and target tokens to the dataset
            data.append((input_tokens, target_tokens))

    # Calculate maximum lengths of input and target sequences for padding
    max_input_len = max(len(pair[0]) for pair in data)
    max_target_len = max(len(pair[1]) for pair in data)

    # Pad sequences to achieve consistent lengths and prepare data tensors
    padded_inputs = []
    padded_targets = []
    for input_seq, target_seq in data:
        # Pad input sequences with "PAD" token to max length
        input_seq = input_seq + [alphabet.symb["PAD"]] * (max_input_len - len(input_seq))
        # Add a "Start of Sequence" token and pad target sequences
        target_seq = target_seq + [alphabet.symb["PAD"]] * (max_target_len - len(target_seq))
        # Add the padded sequences to their respective lists
        padded_inputs.append(input_seq)
        padded_targets.append(target_seq)

    # Return padded sequences and the maximum lengths
    return padded_inputs, padded_targets, max_input_len, max_target_len + 1  # +1 for SOS

# Function to reconstruct the formula from tokens
def reconstruct_formula(tokens):
    # Helper function to handle recursive reconstruction
    def helper(index):
        # If index is out of range, return None to indicate end
        if index >= len(tokens):
            return None, index
        tok = tokens[index]
        # If token is recognized in the alphabet
        if tok in alphabet.symb_reverse:
            if tok == alphabet.symb["LB"]:
                # Reconstruct left operand/formula
                left_formula, next_index = helper(index + 1)
                if next_index >= len(tokens):  # Ensure presence of necessary tokens
                    return None, next_index
                # Determine operation token
                op = tokens[next_index]
                # Reconstruct right operand/formula
                right_formula, next_index = helper(next_index + 1)
                # Ensure correctness of grouped formula with right bracket
                if next_index >= len(tokens) or tokens[next_index] != alphabet.symb["RB"]:
                    return None, next_index
                return [left_formula, op, right_formula], next_index + 1
            elif tok == alphabet.symb["NO"]:  # Handle negation
                formula, next_index = helper(index + 1)
                return [alphabet.symb["NO"], formula], next_index
            elif tok == alphabet.symb["RB"] or tok == alphabet.symb["DE"] or tok == alphabet.symb["EOS"]:
                return None, index  # End of meaningful tokens
            else:
                return tok, index + 1  # Return simple token
        else:
            return None, index + 1  # Return None for unrecognized tokens

    # Call the helper function starting from the first index
    formula, _ = helper(0)
    return formula
