from imports import * # Import libraries
outp_dict = {}

def create_dataset(iterations = list, calculus = list):
    prem = gen_prem() # Generate random premises
    drvas = generate_derivations(iterations = iterations, calculus = calculus, prem = prem) # Generate all derivations from premises up to max_iterations in length using "calculus"
    sample_conclusions = get_conclusions(premises = prem, # Get sample conclusions
                                         max_iterations = iterations[1],
                                         calculus = calculus)
    drvas = [l for l in tqdm(drvas, desc =  "Checked derivations for sample conclusions") if l[-1] in sample_conclusions] # Filter out derivations with sample conclusions
    max_y_train = torch.tensor(util.lflt(max(drvas, key=len))) 

    inpt, y_train_ordered, max_y_train_len = gen_optimized(drvas) # Generate onehot encoded input dataset "inpt", output dataset "y_train_ordered" and get maximum length of derivations in y as number
    print(f"LENy_t: {len(y_train_ordered)}") # Print number of outputs, the unique derivations in the dataset, which are possibly more than premise conclusion pairs in the input datset

    inpt = np.array(inpt) # Convert input dataset to numpy array
    inpt = torch.tensor(inpt, dtype= torch.float32) # Convert numpy array to torch tensor that can be loaded to GPU


    inpt_2d = inpt.view(inpt.size(0),-1) # Reshape input dataset to 2d shape for FFN
    inpt_2d = torch.cat((inpt_2d[:,:1],inpt_2d[:,14:]), dim = 1) 

    inpt_3d = inpt

    # Print results
    print(f"Number x_train examples: {len(inpt)}")
    print(f"Average number ground truth examples/x_train example: {len(drvas)/len(y_train_ordered)}")
    return inpt_2d, inpt_3d, y_train_ordered, max_y_train_len # Return the two and three dimensional input dataset, the output datset and the maximum length of derivations in the latter

def get_conclusions(premises, max_iterations, calculus):
    drvas = [] # Define list of derivations
    
    # This first part is for the first iteration of rule application to only the premises
    for prem in premises: # Iteration over all premises in the list of premis pairs
        subsets = util.subsets_with_length(prem, 1) + util.subsets_with_length(prem, 2) # Generate all subsets of length 1 and 2 of the premise pairs
        for i in subsets: # Iterate over these (effectivly both single premises and the premise pair)
            for rule in calculus: # Iterate over all rules in the calculus
                if calculi.check(rule, i): # Check whether a rule is applicable to a given subset
                    new = i + [rule(i)] # If it is applicable apply the rule to it
                    drvas.append(new) # Append the result to the list of derivations

    # This second part is for the rest of the iterations
    iterations = 2 # Because rules have been applied once above the iteration counter is set to 2
    prev = 0
    while iterations <= max_iterations: # While the iterations have not exceeded the maximum number set:
        sub_drvas = drvas[prev:]
        prev = len(drvas)
        for drva in tqdm(sub_drvas, desc = f"Processed premises for sample conclusions at iteration {iterations}"): # Iterate over all derivations at current iteration with progress bar indicating processed derivations
            subsets = util.subsets_with_length(drva, 1) + util.subsets_with_length(drva, 2) # The rest of the for loop is the same as in the first part
            subsets = sample(subsets, int(round(len(subsets)/(iterations)))) # Except here: The higher the number of iterations the more derivations are filtered out to keep the dataset small.
            # For given premis-conclusion pairs not all derivations are therefore in the resulting dataset. But this function only produces the sample conclusions, so this is no problem.
            for i in subsets:
                cand = []
                for rule in calculus:
                    if calculi.check(rule, i):
                        cand.append(rule(i))
            new = drva + [choice(cand)] # Only append one of the derived formulas to the previous conclusions to reduce the number of derived conclusions
            drvas.append(new)
        iterations += 1
    conclusions = [x[-1] for x in drvas] # Get only the conclusions as last list elements
    conclusions_proxy = []
    conclusions = [conclusions_proxy.append(x) for x in conclusions if x not in conclusions_proxy] # Create a list by list comprehension of only unique conclusions
    return conclusions_proxy

def generate_derivations(iterations = list, calculus = list, prem = list):
    iter = 1
    number_examples = 0
    inpt = []
    drvas = prem # In the beginning the derivations consist of only the premises
    start_point_next_it = 0
    with Pool() as pool: # Enable multiprocessing on multiple CPU threads
        while iter <= iterations[1]: # While the max iterations of derivation production have not been exceeded derivations are produced
            args = [(i, iter) for i in drvas[start_point_next_it:]]
            drvas_proxy = []

            """
            Description: Implementation of multprocessing and a partial function for my existing code, via new subfunction "process_subsets".
            Generated by: GPT-4
            URL of Service: https://platform.openai.com/playground
            """
            process_subsets_with_calculus = partial(process_subsets, calculus=calculus) # Define a partial function for generating derivations using "process subsets" on multiple CPU threads
            for result in tqdm(pool.imap(process_subsets_with_calculus, args), # Run derivation generation on all avaiable CPU threads 
                            desc=f"Processed at iteration {iter}",
                            total=len(args),
                            miniters=1,
                            mininterval=0.1):
                drvas_proxy.extend(result)

            start_point_next_it = len(drvas)
            if iter == 1:
                drvas = drvas_proxy # At the first iteration the resulting derived formulas are again taken as input - Now the derivation dataset contains only sublists of premises and one derived formula each
            else:
                drvas += drvas_proxy # After the first iteration the resulting formulas with their premises are appended to the list of derivations - Now the derivation dataset is extended by derivations with two or more iterations of rule application
            iter += 1
            
    return drvas # Return the list of all generated derivations

def process_subsets(args, calculus): # Generate derivations like above in get_conclusions but without filtering up to generations of two iterations. In my experiments I never did more than two iterations
    i, iterations = args
    processed_data = []
    if len(i) > 1:
        subsets = util.subsets_with_length(i, 1) + util.subsets_with_length(i, 2)
    else:
        subsets = [i]
    if iterations > 2:
        subsets = sample(subsets, int(round(len(subsets)/(iterations*iterations*iterations))))
    for d in subsets: 
        for rule in calculi.ipl:
            if calculi.check(rule, d):
                new = i + [rule(d)]
                if new not in processed_data:
                    processed_data.append(new)
    return processed_data

def gen_prem(): # Function that generates random premises
    prem, prem_pool = [], []
    while len(prem_pool) < 30000: # Generate premises until 30000 are generated
        form = calculi.gen_wff(randint(1,t_nu), depth=1) # Generate a well formed formula using the function "calculi.wff" with 1 to t_nu different propositional variables and depth 1, meaning that WFF-generation rules are applied up to once on propositional variables
        prem_pool.append(form)
    while len(prem) < 1200: # Get 1200 random subsets of length 2 as unique premise pairs
        new = sample(prem_pool, 2)
        if new not in prem:
            prem.append(new)
    return prem # Return the unique premise pairs

"""
Description: Optimization of my exiting "gen" function tu run fast, via making lists immutable tuples, parts of the comments are therefore also by GPT
Generated by: GPT-4
Date: 2024-04-05
URL of Service: https://platform.openai.com/playground
"""
def to_immutable(obj):
    """
    Recursively converts lists (and nested lists) into tuples
    to make them hashable.
    """
    if isinstance(obj, list):
        return tuple(to_immutable(item) for item in obj)
    return obj


def gen_optimized(drvas):
    y_train_ordered = []
    symb_DE = calculi.symb["DE"]
    unique_inputs = {}
    outp_dict = {}
    onehot_drvas  = [util.lflt(i) for i in drvas] # Until now formulas are encoded in lists of different depths. This converts all brackets within formulas into symbols 
    onehot_drvas  = np.array(onehot_drvas, dtype=object) # Onehot encode the derivations
    maxl = max(len(sub) for sub in onehot_drvas) # Get length of longest derivation
    onehot_drvas = np.array([sub + [0] * (maxl - len(sub)) for sub in tqdm(onehot_drvas, desc =  "Padded x_train entries")]) # Pad all derivation entries with zeros to length of longest derivation
    onehot_drvas = torch.tensor(onehot_drvas) 
    onehot_drvas = [torch.cat([F.one_hot(i[j], num_classes=t_nu + 9) for j in range(len(i))], dim=-1) for i in tqdm(onehot_drvas)]
    max_l = len(onehot_drvas[0])
    n = len(drvas)


    for i, (drv, one_drv) in tqdm(enumerate(zip(drvas, onehot_drvas)), desc = "Processed entries for x_train and y_tdict",total = n):
        # Making sure every element is converted to a hashable type
        in_i = (
            to_immutable(drv[:2]),    # Convert the list and any nested lists into tuples
            to_immutable(symb_DE),    # Convert symb_DE into a tuple if it's a list
            to_immutable(drv[-1])     # Convert the last element into a tuple if it's a list
        )
        
        # Use the immutable representation as a key
        if in_i not in unique_inputs:
            position = len(unique_inputs)
            unique_inputs[in_i] = position
            y_train_ordered.append([one_drv])
        else:
            position = unique_inputs[in_i]
            y_train_ordered[position].append(one_drv)
    

    # Convert the unique_inputs dict keys back to list format if required
    inpt = [[pos] + recursively_convert_to_list(key) for key, pos in unique_inputs.items()]
    max_y = max(len(sub) for sub in y_train_ordered)
    y_padding = [0] * max_l
    max_y_train_len = len(y_train_ordered[0][0])
    padded_y_train_ordered = [sub + [torch.tensor(y_padding)] * (max_y - len(sub)) for sub in tqdm(y_train_ordered, desc="Padded y_train_ordered")]
    y_train_ordered = torch.stack([torch.stack(sub) for sub in padded_y_train_ordered])
    inpt = list_to_onehot(inpt)
    return inpt, y_train_ordered, max_y_train_len # Return input dataset, the output datset and the maximum length of derivations in the latter

def parallel_pad(input):
    num_cores = cpu_count()  # Or set this to a fixed number of cores if desired

    with Pool(processes=num_cores) as pool:
        # Use partial to set the constant parameter `t_nu`
        func = partial(process_item, t_nu=t_nu)
        
        # Distribute the work across the pool and collect results
        # The tqdm wrapper allows for a progress bar
        results = []
        for result in tqdm(pool.imap_unordered(func, input), total=len(input)):
            results.append(result)
    return results
 

def list_to_onehot(list):
    inpt = [util.lflt(i) for i in list] # This converts all brackets within formulas into symbols 
    inpt = np.array(inpt, dtype=object) # Convert to numpy array
    maxl = max(len(sub) for sub in inpt)
    inpt = np.array([sub + [0] * (maxl - len(sub)) for sub in tqdm(inpt, desc =  "Padded x_train entries")]) # Pad with zeros to length of longest derivation
    inpt = torch.tensor(inpt) # Convert to tensor, loadable to GPU
    padding = torch.zeros(t_nu + 8) # Create padding tensor

    # This part was also optimized by GPT by performing onehot encoding and concatenation in one loop
    inpt_processed = []
    for i in inpt:
        ione = F.one_hot(i[1:], num_classes=t_nu + 9)
        index = torch.cat((i[0].unsqueeze(0), padding))
        conc = torch.cat((index.unsqueeze(0), ione), dim=0)
        inpt_processed.append(conc)
    return inpt_processed

def recursively_convert_to_list(obj):
    """
    Recursively convert tuples (and nested tuples) back into lists.
    This is the reverse operation of the to_immutable function.
    """
    if isinstance(obj, tuple):
        return [recursively_convert_to_list(item) for item in obj]
    return obj

# GPT
def process_item(item, t_nu):
    return torch.cat([F.one_hot(i, num_classes=t_nu + 9) for i in item], dim=-1) # Onehot encode and the flatten a list
