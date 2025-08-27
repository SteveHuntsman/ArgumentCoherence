# %%
api_key = '' # INSERT YOUR OPENAI KEY HERE

# %%
import openai

def chat_with_openai(prompt: str, model: str, history: list = None, api_key = api_key):
    client = openai.OpenAI(api_key = api_key)

    if history is None:
        history = []
    
    history.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model = model,
        messages = history,
    )
    
    return response.choices[0].message.content.strip(), history

# %% https://claude.ai/share/2fb655ce-e848-475a-b214-2e2958cdd1ec
import re
import ast

def extract_edgelist(text):
    """
    Extract NetworkX-style edge lists from LLM outputs.
    
    Handles various formats of edge lists like:
    - ('v', 'w', x)
    - ["v", "w", x]
    - Markdown code blocks
    - Lists with or without brackets
    
    Args:
        text (str): The text output from an LLM
        
    Returns:
        list: A list of tuples in the form (vertex1, vertex2, weight)
    """
    # Strip markdown code blocks if present
    clean_text = re.sub(r'```python\s*|\s*```', '', text)
    clean_text = re.sub(r'```\s*|\s*```', '', clean_text)
    
    # Try to find list-like structures in the text
    edges = []
    
    # Pattern for tuples/lists like ('v', 'w', x) or ["v", "w", x]
    # This handles single quotes, double quotes, and no quotes for vertex names
    pattern = r'[\(\[][\s]*[\'"]?([^\'",\[\]\(\)]+)[\'"]?[\s]*,[\s]*[\'"]?([^\'",\[\]\(\)]+)[\'"]?[\s]*,[\s]*([\d\.]+)[\s]*[\)\]]'
    tuples = re.findall(pattern, clean_text)
    
    if tuples:
        for v, w, weight in tuples:
            try:
                # Try to convert weight to float
                edges.append((v.strip(), w.strip(), float(weight.strip())))
            except ValueError:
                continue
        
        return edges
    
    # If no matches with the regex, try using ast.literal_eval to parse Python literals
    try:
        # Look for list-like structures with square brackets
        list_match = re.search(r'\[([^\[\]]*)\]', clean_text)
        if list_match:
            list_content = "[" + list_match.group(1) + "]"
            parsed_list = ast.literal_eval(list_content)
            for item in parsed_list:
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    v, w, weight = item
                    edges.append((str(v), str(w), float(weight)))
            
            return edges
    except (SyntaxError, ValueError):
        pass
    
    # Last resort: try to find list-like structures in the full text
    try:
        # This will look for any valid Python literal in the text
        for line in clean_text.split('\n'):
            line = line.strip()
            if line and (line.startswith('(') or line.startswith('[')):
                try:
                    item = ast.literal_eval(line)
                    if isinstance(item, (list, tuple)) and len(item) == 3:
                        v, w, weight = item
                        edges.append((str(v), str(w), float(weight)))
                except (SyntaxError, ValueError):
                    continue
    except Exception:
        pass
    
    return edges

# %% https://claude.ai/share/72e39460-7229-44e6-b373-e0233f348193
import numpy as np
from collections import defaultdict

def process_edgelists(edgelists):
    """
    Process N NetworkX edge lists according to the specified requirements.
    
    Args:
        edgelists: A list of N edge lists, where each edge list contains entries
                   of the form ('v', 'w', x) with weight x between 0 and 10.
    
    Returns:
        A single edge list with unique edges and weights calculated as m/10,
        where m is the median weight of the corresponding edges in the N lists.
    """
    # i) Sort vertices in individual edges lexicographically
    sorted_edgelists = []
    for edgelist in edgelists:
        sorted_edges = []
        for edge in edgelist:
            v, w, weight = edge
            # Sort vertices lexicographically
            if v > w:
                v, w = w, v
            sorted_edges.append((v, w, weight))
        sorted_edgelists.append(sorted_edges)
    
    # ii) Sort each edge list lexicographically
    for i in range(len(sorted_edgelists)):
        sorted_edgelists[i] = sorted(sorted_edgelists[i], key=lambda x: (x[0], x[1]))
    
    # iii) Compile auxiliary list A of unique unweighted edges
    unique_edges = set()
    for edgelist in sorted_edgelists:
        for edge in edgelist:
            v, w, _ = edge
            unique_edges.add((v, w))
    
    # Sort the unique edges lexicographically
    unique_edges = sorted(list(unique_edges))
    
    # iv) Align edge lists by inserting edges with weight 5 where missing
    aligned_edgelists = []
    for edgelist in sorted_edgelists:
        # Create a dictionary for quick lookup of edges in this list
        edge_dict = {(v, w): weight for v, w, weight in edgelist}
        
        # Create aligned edge list with all unique edges
        aligned_list = []
        for v, w in unique_edges:
            if (v, w) in edge_dict:
                aligned_list.append((v, w, edge_dict[(v, w)]))
            else:
                # Insert edge with weight 5 if it doesn't exist in this list
                aligned_list.append((v, w, 5))
        
        aligned_edgelists.append(aligned_list)
    
    # v) Calculate median weights and create final edge list
    final_edgelist = []
    for i, (v, w) in enumerate(unique_edges):
        # Extract weights of this edge from all aligned edge lists
        weights = [aligned_edgelists[j][i][2] for j in range(len(aligned_edgelists))]
        
        # Calculate the median weight
        median_weight = np.nanmedian(weights)   # nanmedian to be safe
        
        # Calculate the final weight
        # final_weight = (median_weight - 5) / 5
        final_weight = median_weight / 10
        
        # Add to final edge list
        final_edgelist.append((v, w, final_weight))
    
    return final_edgelist

# %% https://claude.ai/share/daa5f67b-f78d-4539-81d0-c36ea250d777
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def plot_weighted_graph(G, node_size=300, node_color='white', 
                         edge_width=2, font_size=10, layout=nx.circular_layout):
    """
    Plot a NetworkX graph with edge colors and alpha based on weights.
    
    Parameters:
    -----------
    G : NetworkX graph
        An undirected graph with weights between 0 and 1 on edges
    node_size : int, optional
        Size of the nodes (default: 300)
    node_color : str, optional
        Color of the nodes (default: 'white')
    edge_width : int, optional
        Width of the edges (default: 2)
    font_size : int, optional
        Size of the node labels (default: 10)
    layout : callable, optional
        Layout function for node positioning (default: nx.spring_layout)
    """
    # Get positions for nodes
    pos = layout(G)
    
    # Create figure and axis
    plt.figure(figsize=(4, 4))
    # plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # Draw edges with colors based on weights
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 0.5)  # Default to 0.5 if no weight specified
        
        # Ensure weight is between 0 and 1
        weight = max(0, min(1, weight))
        
        # Calculate alpha based on weight
        alpha = 2 * abs(weight - 0.5)
        
        # Calculate color (red for 0, blue for 1)
        r = 1 - weight  # Red component decreases as weight increases
        g = 0           # No green component
        b = weight      # Blue component increases as weight increases
        
        # Draw the edge
        if weight < .5:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                color=(r, g, b, alpha), linewidth=edge_width, linestyle='--')
        else:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                color=(r, g, b, alpha), linewidth=edge_width)            
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, 
                          edgecolors='black', ax=ax)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=font_size, font_color='black')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# %% https://claude.site/artifacts/62ad7f5d-2ffa-46e9-8036-e406b14435e9
import re
import time
# import os # for API key if/as necessary
from typing import List, Dict
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

def call_model(messages: List[Dict], model: str = "gpt-4.1", max_retries: int = 3, retry_delay: int = 2) -> str:
    """
    Call the OpenAI model with the given messages.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,  # Lower temperature for more deterministic outputs
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)  # Wait before retrying
            else:
                raise  # Re-raise the exception after max retries

def check_verification(response: str) -> bool:
    """
    Check if the verification response contains a positive confirmation.
    """
    # Check for "YES" anywhere in the response (case insensitive)
    match = re.search(r'\bYES\b', response, re.IGNORECASE)
    # Check for "NO" only at the beginning or end of the response
    no_match = re.search(r'(^NO\b|\bNO$)', response, re.IGNORECASE)    
    # # Make sure "NO" is not in the response (to avoid cases where both YES and NO appear)
    # no_match = re.search(r'\bNO\b', response, re.IGNORECASE)
    return match is not None and no_match is None

def process_with_verification(action_prompt: List[str], verification_prompt: List[str], data: str, model: str = "gpt-4.1", max_retries: int = 3) -> str:
    """
    Process data through a sequence of action and verification steps.
    
    Args:
        action_prompt: List of action prompts to apply sequentially
        verification_prompt: List of verification prompts matching action prompts
        data: The initial data to process
        model: The model to use (default: gpt-4.1)
        max_retries: Maximum number of retry attempts for each action (default: 3)
        
    Returns:
        The final processed output after all steps
    """
    if len(action_prompt) != len(verification_prompt):
        raise ValueError("Action and verification prompt lists must have the same length")
    
    conversation_history = []
    current_output = None
    
    # Process each action-verification pair
    for i in range(len(action_prompt)):
        # For the first action, append data to the prompt
        if i == 0:
            full_prompt = action_prompt[i] + data
        else:
            # For subsequent actions, append the previous output to the prompt
            full_prompt = action_prompt[i] + "\n\n" + current_output
        
        retry_count = 0
        verified = False
        
        # Try the action until verification passes or max retries reached
        while not verified and retry_count < max_retries:
            try:
                # Prepare messages for the model
                messages = conversation_history + [{"role": "user", "content": full_prompt}]
                
                # Call the model with the action prompt
                print(f"Step {i+1}/{len(action_prompt)}: Executing action...")
                action_response = call_model(messages, model)
                
                # Add the action and response to conversation history
                conversation_history.append({"role": "user", "content": full_prompt})
                conversation_history.append({"role": "assistant", "content": action_response})
                
                # Verify the action response
                print(f"Step {i+1}/{len(action_prompt)}: Verifying result...")
                verification_messages = conversation_history + [
                    {"role": "user", "content": verification_prompt[i]}
                ]
                verification_response = call_model(verification_messages, model)
                print(f"Verification response: {verification_response[:100]}...")
                
                # Check if verification passed
                verified = check_verification(verification_response)
                
                if verified:
                    print(f"Step {i+1}/{len(action_prompt)}: Verified ✓")
                    current_output = action_response
                else:
                    print(f"Step {i+1}/{len(action_prompt)}: Verification failed, retrying... ({retry_count+1}/{max_retries})")
                    # Add verification to history for context in next attempt
                    conversation_history.append({"role": "user", "content": verification_prompt[i]})
                    conversation_history.append({"role": "assistant", "content": verification_response})
                    retry_count += 1
            except Exception as e:
                print(f"Error during step {i+1}: {str(e)}")
                retry_count += 1
                time.sleep(2)  # Wait before retrying
            
            # If max retries reached, use last output and continue
            if retry_count >= max_retries and not verified:
                print(f"Warning: Max retries reached for step {i+1}. Moving to next step with current output.")
                if current_output is None and i == 0:
                    # Special case for first step failure
                    print("First step failed. Trying a simplified approach...")
                    try:
                        # Try a direct approach without verification for first step
                        simple_messages = [{"role": "user", "content": full_prompt}]
                        current_output = call_model(simple_messages, model)
                    except Exception as e:
                        print(f"Simplified approach also failed: {str(e)}")
                        return f"Failed to process: {str(e)}"
                else:
                    current_output = action_response
    
    return current_output

# %%

# %%
max_props = 20
action_prompt = []
verification_prompt = []
action_format_suffix = "Keep the lists separated by participant, with each participant indicated in a comment starting with '#' before the corresponding list."
critic_format_q = "Are the propositions separated into lists, with each list preceded by comment starting with '#' to identify the corresponding participant, and with roughly " + str(max_props) + " or fewer total propositions across all lists?"
critic_suffix = " Feel free to explain your answer, but in any event: END YOUR ANSWER WITH A SINGLE WORD: EITHER 'YES' OR 'NO.'"

action_prompt.append("Produce lists of the most important claims or propositions captured in the transcript below after the delimiter '###'. Try to keep the claims/propositions simple and the lists as short as possible without omitting any essential detail, with roughly " + str(max_props) + " or fewer propositions in total." + action_format_suffix + "\n\n###\n\n")
verification_prompt.append(critic_format_q + critic_suffix)

action_prompt.append("Try to simplify the lists of propositions. Do not increase the number of propositions unless it facilitates overall simplification. " + action_format_suffix)
verification_prompt.append("Are the lists of propositions above as simplified as they can be? " + critic_format_q + critic_suffix)

action_prompt.append("Make any references in the lists of propositions above as explicit and self-contained as practical. In particular, try to replace any pronouns with their antecedents." + action_format_suffix)
verification_prompt.append("Are the lists of propositions above as explicit and self-contained as practical, and mostly if not entirely free of pronouns? " + critic_format_q + critic_suffix)

action_prompt.append("Enumerate the propositions in each list as '- p1: ', '- p2: ', '- p3: ', etc., using consecutive numbering across lists, a newline separating lists, and each proposition/claim within a list on its own line. " + action_format_suffix)
verification_prompt.append("Are the propositions in each list enumerated as '- p1: ', '- p2: ', '- p3: ', etc., using consecutive numbering across lists, a newline separating lists, and each proposition/claim within a list on its own line?" + critic_format_q + critic_suffix)

action_prompt.append("For any propositions that are not already bare and/or main clause propositions, transform them accordingly. This includes, but is not limited to, replacing any propositions of the sort 'A believes B,' 'B because A,' 'A says B,' etc. with only 'B.'" + action_format_suffix)
verification_prompt.append("Are the lists of propositions bare and/or main clause propositions? For example, any propositions of the sort 'A believes B,' 'B because A,' 'A says B,' etc. are not bare and/or main clause propositions, while 'B' is (unless it is still of a form along the lines just mentioned)." + critic_format_q + critic_suffix)

# %%
# filename = "/Users/shuntsman/Local/MelianDialogue.txt"
# filename = "MelianDialogue" # (S)Lightly reformatted from the Crawley translation at (variously):
    # https://www.gutenberg.org/files/7142/7142-h/7142-h.htm#link2HCH0017
    # http://academics.wellesley.edu/ClassicalStudies/CLCV102/Thucydides--MelianDialogue.html
    # http://classics.mit.edu/Thucydides/pelopwar.5.fifth.html
# filename = "InheritTheWindAct2Scene2.txt"
# filename = "BrownVsBoard.txt"

with open(filename, "r") as file:
    transcript = file.read()

# %%
action_prompt[0] = action_prompt[0] + transcript

# %%
build_coherence = "Imagine that you are a perfectly objective arbitrator with impeccable judgment and integrity. In response to a prompt of the form 'buildCoherence: ' below followed by a list of labeled propositions, please do the following: First, determine which pairs of propositions are substantively related. Second, for each related pair of propositions, determine their logical relationship, assuming that at least one is true, whether or not either actually is. I want you to ignore the truth, falsity or basis in fact of either claim. Third, based on your determination just above, numerically rate the relative consistency of the two propositions. Do not pay attention to or comment on the truth or basis in fact of either proposition independent of the other. Your rating of relative consistency should be on a scale from 0 to 10, with a value of 0 for a pair of propositions that are not at all consistent and a value of 10 for a pair of propositions that are totally consistent. I cannot emphasize enough that for your rating, I want you to ignore the truth or basis in fact of either proposition, since anything that is not consistent with reality cannot be true. If you determine that propositions are unrelated despite previously determining otherwise, omit that pair. To be clear, a pair of false but consistent claims should also be rated a 10. Meanwhile, a pair of propositions of which one is true and the other is false, should be rated a 0. Finally, construct a networkx graph where propositions are vertices and edges correspond to substantively related pairs of propositions, with weights given by the consistency ratings just above. Only return the edge list with proposition labels for vertices. i.e., return responses in this format (here 'p2', 'p3', 'p4', and 'p5' are labels): [('p2', 'p3', 0), ('p2', 'p5', 10), ('p3', 'p4', 9), ('p3', 'p5', 2)]. Order vertices (in edges) and edges (in the graph) lexicographically.\n\nbuildCoherence: \n\n"

# %% Process the transcript several times
num_summaries = 5
all_separated_lists = []
delimiter = '\n\n###\n\n'
for s in range(num_summaries):
    print('SUMMARY EXTRACTION: ' + str(s+1) + '/' + str(num_summaries) + ':')
    separated_lists = process_with_verification(action_prompt, verification_prompt, transcript, max_retries=3)
    all_separated_lists.append(separated_lists)

# %% Pick the best separated lists that summarize the transcript
all_separated_lists_str = delimiter + delimiter.join(all_separated_lists)

best_summarization_prompt = "Below following the first delimiter '###' is a transcript; subsequent instances of '###' delimit various proposed sets of lists of the most important claims or propositions captured in the transcript. Carefully determine which of the proposals best answers 'YES' to the following questions: \n\t" + "\n\t".join([vp.removesuffix(critic_suffix) for vp in verification_prompt]) + "\nRepeat the best proposal--and only that one--exactly." + delimiter + transcript + all_separated_lists_str

separated_lists, _ = chat_with_openai(prompt=best_summarization_prompt, model="gpt-4.1", api_key=api_key)

# separated_lists = process_with_verification(action_prompt, verification_prompt, transcript, max_retries=10)

print("\nFinal Result:")
print(separated_lists)

# %% For short-circuiting
# filename = "/Users/sman/Local/foo.txt"
# with open(filename, "r") as file:
#     separated_lists = file.read()

# %%
anonymize_lists = lambda text: '\n'.join([line for line in text.split('\n') if not line.strip().startswith('#') and line.strip()])
proposition_list = anonymize_lists(separated_lists)

# %%
N = 30
coherence_graphs = []
for n in range(N):
    print('Building graph ' + str(n+1) + '/' + str(N) + '...')
    response, _ = chat_with_openai(prompt = build_coherence + proposition_list, model = "o3-mini", history = None, api_key = api_key)
    coherence_graph = extract_edgelist(response)
    coherence_graphs.append(coherence_graph)

# %%
import random
import numpy as np
from numpy import linalg 
from scipy.special import comb

C_max = N#int(np.ceil(N / 2))
quality = np.zeros((C_max))    # 0, 1 degenerate
all_l1_distances = {}
for C in range(2, C_max):
    print('Repeatedly sampling ' + str(C) + '/' + str(C_max) + ' graphs...')
    # C = number of coherence_graphs for consensus

    M = int(min(comb(len(coherence_graphs), C), 100))   # number of samples from N

    # M samples of size C from coherence_graphs for analyzing quality of consensus
    sample = []
    for m in range(M):
        E = process_edgelists(random.sample(coherence_graphs, C))
        G = nx.from_edgelist([(u, v, {'weight': w}) for u, v, w in E])
        # G.add_nodes_from(['p'+str(i+1) for i in range(len(proposition_list))])
        G.add_nodes_from(['p'+str(i+1) for i in range(sum(1 for line in proposition_list.split('\n')))])
        sample.append(G)

    # L1/edit distances of samples
    d = np.zeros((M, M))
    for m1 in range(M):
        adj1 = nx.adjacency_matrix(sample[m1]).reshape(-1)
        for m2 in range(M):
            adj2 = nx.adjacency_matrix(sample[m2]).reshape(-1)
            d[m1, m2] = linalg.norm((adj1 - adj2).toarray(), ord=1)

    # Off-diagonal elements are relevant
    l1_distances = d[np.triu_indices(len(d), k=1)]
    quality[C] = np.median(l1_distances)
    all_l1_distances[C] = l1_distances

# %%
# Plot with enhanced visualization in grayscale
plt.figure(figsize=(10, 6))

# Plot the median line
plt.plot(range(2, C_max), quality[2:], color='black', linewidth=2, marker='o', label='Median')

# Create box plots for each sample size
boxplot_data = [all_l1_distances[C] for C in range(2, C_max) if C in all_l1_distances]
box_positions = range(2, 2 + len(boxplot_data))

if boxplot_data:
    bp = plt.boxplot(boxplot_data, positions=box_positions, widths=0.6, 
                    patch_artist=True, showfliers=True)
    
    # Customize box plot appearance - grayscale theme
    for box in bp['boxes']:
        box.set(facecolor='lightgray', alpha=0.7)
    for median in bp['medians']:
        median.set(color='black', linewidth=2)
    for cap in bp['caps']:
        cap.set(color='dimgray', linewidth=1.5)
    for whisker in bp['whiskers']:
        whisker.set(color='dimgray', linewidth=1.5)
    for flier in bp['fliers']:
        flier.set(marker='o', color='gray', markersize=4, alpha=0.5)

# # Add sample size numbers
# for i, C in enumerate(range(2, C_max)):
#     if C in all_l1_distances:
#         data = all_l1_distances[C]
#         plt.text(i+2, np.min(data), f'n={len(data)}', 
#                  verticalalignment='bottom', horizontalalignment='center', fontsize=8,
#                  color='dimgray')

# Improve visualization with grid and better labeling
plt.grid(True, linestyle='--', alpha=0.5, color='lightgray')
plt.xlim(1.5, C_max-0.5)
current_ylim = plt.ylim()
plt.ylim(bottom=0, top=current_ylim[1] * 1.1)  # Add 10% space at top

plt.xlabel('Sample size (n) for consensus coherence graph', fontsize=12)
plt.ylabel('L1 distance between graph realizations', fontsize=12)
# plt.title('Distribution of L1 distances between realizations of consensus graph', fontsize=14)

# Simplified legend in upper right
legend = plt.legend(['Median', 'IQR', 'Range'], loc='upper right')

plt.tight_layout()
# plt.savefig('consensus_quality_boxplot_grayscale.png', dpi=300)
plt.show()

# %% Find N_0 such that quality[N_0] is small relative to, and occurs after, np.argmax(quality)
N_0 = N
# N_0 = np.argmax(quality) + np.argmax(quality[np.argmax(quality):] <= 0.1 * np.max(quality)) + 1 # count from 1

# %%
# consensus_coherence_edgelist = process_edgelists(coherence_graphs)
consensus_coherence_edgelist = process_edgelists(random.sample(coherence_graphs, N_0))

# %%
import networkx as nx
consensus_coherence_graph = nx.from_edgelist([(u, v, {'weight': w}) for u, v, w in consensus_coherence_edgelist])

# %%
# Scale weights to be in the unit interval for plotting
# H = nx.Graph((u, v, {'weight': (w['weight'] + 1) / 2}) for u, v, w in G.edges(data=True))

# %% Nodes in correct order
# consensus_coherence_graph = nx.relabel_nodes(consensus_coherence_graph, {node: node for node in sorted(consensus_coherence_graph.nodes(), key=lambda x: int(x[1:]))})

# %%
# plot_weighted_graph(consensus_coherence_graph, node_color='white', edge_width=2, font_size=10, layout=nx.circular_layout)

# %%
import re

per_list = [(h.strip(), len(re.findall(r'^- p\d+: ', c, re.MULTILINE))) for h, c in re.findall(r'(#[^\n]*\n)((?:(?!#).*(?:\n|$))*)', separated_lists)]
list_label = [re.sub(r'^# ', '', s) for s in [foo[0] for foo in per_list]]
list_length = [foo[1] for foo in per_list]
list_index = [i for i, count in enumerate(list_length) for _ in range(count)]

angle = 2 * np.pi * (.5 + np.arange(len(list_index))) / len(list_index)
# list_angle = [angle[np.array(list_index) == i] for i in range(len(list_length))]
# list_angle_median = np.array([np.median(foo) for foo in list_angle])

# r = 0.5
# pos = {'p'+str(i+1) : np.array([np.cos(angle[i]) + (r * np.cos(list_angle_median[list_index[i]])), np.sin(angle[i]) + (r * np.sin(list_angle_median[list_index[i]]))]) for i in range(len(angle))}

pos = {'p'+str(i+1) : np.array([np.cos(angle[i]), np.sin(angle[i])]) for i in range(len(angle))}

G = consensus_coherence_graph
node_size=300
node_color='white'
edge_width=2
font_size=10

# Create figure and axis
plt.figure(figsize=(4, 4))
ax = plt.gca()

# H = nx.Graph((u, v, {'weight': (w['weight'] + 1) / 2}) for u, v, w in G.edges(data=True))

# Draw edges with colors based on weights
for u, v, data in G.edges(data=True):
    weight = data.get('weight', 0.5)  # Default to 0.5 if no weight specified
    
    # Ensure weight is between 0 and 1
    weight = max(0, min(1, weight))
    
    # Calculate alpha based on weight
    alpha = 2 * abs(weight - 0.5)
    
    # Calculate color (red for 0, blue for 1)
    r = 1 - weight  # Red component decreases as weight increases
    g = 0           # No green component
    b = weight      # Blue component increases as weight increases
    
    # Draw the edge
    if weight < .5:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
            color=(r, g, b, alpha), linewidth=edge_width, linestyle='--')
    else:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
            color=(r, g, b, alpha), linewidth=edge_width)


# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, 
                        edgecolors='black', ax=ax)

# Draw node labels
nx.draw_networkx_labels(G, pos, font_size=font_size, font_color='black')

# Draw proto-cuts based on participant labels
plt.plot([0, 1], [0, 0], color='black', linestyle='dotted', linewidth=2)
ind_0 = np.cumsum([0] + list_length)[1:-1]
ind_1 = (np.cumsum([0] + list_length[:-1]) + 1)[1:]
for i in range(len(list_length) - 1):
    foo = (pos['p'+str(ind_0[i])] + pos['p'+str(ind_1[i])]) / 2
    plt.plot([0, foo[0]], [0, foo[1]], color='black', linestyle='dotted', linewidth=2)

# 
plt.axis('off')
plt.tight_layout()
plt.show()

# %%
import pickle
import datetime

# Get current timestamp for filename
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
pickle_name = filename + "_" + timestamp + ".pkl"

def is_picklable(obj):
    """Test if an object can be pickled"""
    try:
        pickle.dumps(obj)
        return True
    except:
        return False

# Create dictionary with variables that can be pickled
picklable_vars = {}
skipped_vars = []

for key, value in list(globals().items()):
    # Skip special variables and modules
    if (key.startswith('_') or 
        key in ('In', 'Out', 'exit', 'quit', 'get_ipython')):
        continue
    
    # Test if the variable can be pickled
    if is_picklable(value):
        picklable_vars[key] = value
    else:
        skipped_vars.append(key)

# Save to file
with open(pickle_name, 'wb') as file:
    pickle.dump(picklable_vars, file)

print(f"Session saved to {pickle_name}")
print(f"Saved {len(picklable_vars)} variables")

if skipped_vars:
    print(f"Could not save these variables (not picklable): {', '.join(skipped_vars)}")

# %%
# with open("Melian12props.txt_.pkl", 'rb') as file:
#     data = pickle.load(file)

# %%
def calculate_cut_weight(G, S, V_S):
    """
    Calculate the weight of the cut defined by the partition (S, V_S).
    
    Parameters:
    -----------
    G : networkx.Graph
        The input graph.
    S : set
        First set of the partition.
    V_S : set
        Second set of the partition.
        
    Returns:
    --------
    float
        The total weight of edges crossing the cut.
    """
    cut_weight = 0.0
    
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1.0)
        # If the edge crosses the cut (one endpoint in S, the other in V_S)
        if (u in S and v in V_S) or (u in V_S and v in S):
            cut_weight += weight
    
    return cut_weight

# %%
# from copy import deepcopy

# G_neg = deepcopy(G) 
# nx.set_edge_attributes(G_neg, {e: -2 * G.edges[e]['weight'] + 1 for e in G.edges}, 'weight')

# S = {'p7', 'p8', 'p9,' 'p10', 'p11', 'p12', 'p13', 'p14', 'p15'}
# foo = calculate_cut_weight(G_neg, S, G_neg.nodes() - S)
# print(foo)

# # %%
# V_S = G_neg.nodes() - S
# for u, v, data in G.edges(data=True):
#     weight = data.get('weight', 1.0)
#     # If the edge crosses the cut (one endpoint in S, the other in V_S)
#     if (u in S and v in V_S) or (u in V_S and v in S):
#         print((u, v, 2 * weight - 1))


# %%
from copy import deepcopy

G_neg = deepcopy(G) 
nx.set_edge_attributes(G_neg, {e: -2 * G.edges[e]['weight'] + 1 for e in G.edges}, 'weight')
foo = max_cut_local_search(G_neg, seed_bipartition=None, max_moves=10, return_cut_value=True, max_solutions=10, return_full_output=False)
print(foo)

# %%
import networkx as nx
import itertools
from collections import defaultdict

def max_cut_local_search(G, seed_bipartition=None, max_moves=2, return_cut_value=True, max_solutions=10, return_full_output=False):
    """
    A MAX-CUT solver that explores bipartitions within a certain number of 
    element exchanges from a seed bipartition and returns all optimal solutions up to max_solutions.
    
    Parameters:
    -----------
    G : networkx.Graph
        An undirected graph with weighted edges
    seed_bipartition : list, optional
        Initial bipartition (subset of nodes). If None, an empty set is used.
    max_moves : int
        Maximum number of element moves to consider
    return_cut_value : bool
        If True, return the cut value along with each bipartition.
        If False, only return the bipartitions.
    max_solutions : int, default=10
        Maximum number of optimal solutions to return
    
    Returns:
    --------
    If return_cut_value is True:
        list of tuples : [(list, list, float), ...]
            The bipartitions of nodes and their maximum cut values
    If return_cut_value is False:
        list of tuples : [(list, list), ...]
            Just the bipartitions of nodes
    If return_full_output is True:
        verbose : [(list, list, float), ...]
            The bipartitions of nodes and their cut values
    """
    nodes = list(G.nodes())
    n = len(nodes)
    
    # Initialize seed bipartition if not provided
    if seed_bipartition is None:
        seed_bipartition = []
    
    # Convert to sets for efficient operations
    part_1 = set(seed_bipartition)
    part_0 = set(nodes) - part_1
    
    # Function to calculate cut value between two parts
    def calculate_cut_value(p0, p1):
        cut = 0
        for i in p0:
            for j in p1:
                if G.has_edge(i, j):
                    cut += G[i][j].get('weight', 1)
        return cut
    
    # Store cut values and corresponding bipartitions
    cut_to_bipartitions = defaultdict(list)
    
    # Calculate initial cut value
    initial_cut_value = calculate_cut_value(part_0, part_1)
    cut_to_bipartitions[initial_cut_value].append((list(part_0), list(part_1)))
    
    if return_full_output:
        verbose = []

    # For each number of moves from 1 to max_moves
    for k in range(1, max_moves + 1):
        # Generate all possible k-combinations from each part
        for p0_elements in itertools.combinations(part_0, min(k, len(part_0))):
            for p1_elements in itertools.combinations(part_1, min(k, len(part_1))):
                
                # Create new bipartitions by exchanging elements
                new_p0 = (part_0 - set(p0_elements)) | set(p1_elements)
                new_p1 = (part_1 - set(p1_elements)) | set(p0_elements)
                
                # Calculate new cut value
                cut_value = calculate_cut_value(new_p0, new_p1)
                
                # Store the bipartition with its cut value
                # Create a canonical representation to avoid duplicates
                # Sort both parts and use a frozenset to ensure uniqueness
                canonical_bipartition = (
                    frozenset(sorted(new_p0)),
                    frozenset(sorted(new_p1))
                )

                if return_full_output:
                    verbose.append([list(new_p0), list(new_p1), cut_value])
                
                # Add to the dictionary only if we haven't seen this bipartition before
                is_new = True
                for existing_bipartition in cut_to_bipartitions[cut_value]:
                    existing_p0, existing_p1 = existing_bipartition
                    if (frozenset(existing_p0) == canonical_bipartition[0] and 
                        frozenset(existing_p1) == canonical_bipartition[1]) or \
                       (frozenset(existing_p0) == canonical_bipartition[1] and 
                        frozenset(existing_p1) == canonical_bipartition[0]):
                        is_new = False
                        break
                
                if is_new:
                    cut_to_bipartitions[cut_value].append((list(new_p0), list(new_p1)))
    
    # Find the maximum cut value
    max_cut = max(cut_to_bipartitions.keys())
    
    # Get all bipartitions with the maximum cut value (up to max_solutions)
    best_bipartitions = cut_to_bipartitions[max_cut][:max_solutions]
    
    if return_cut_value:
        # Return bipartitions with their cut value
        if return_full_output:
            return [(bp[0], bp[1], max_cut) for bp in best_bipartitions], verbose
        else:
            return [(bp[0], bp[1], max_cut) for bp in best_bipartitions]
    else:
        # Return just the bipartitions
        if return_full_output:
            return best_bipartitions, verbose
        else:
            return best_bipartitions


# %%
# import networkx as nx
# import numpy as np

# def generate_tikz_code(G, pos, scale_factor, node_size=0.8, node_color='white', font_size='normalsize'):
#     """
#     Generate TikZ code from a NetworkX graph with edge colors and styles based on weights.
    
#     Parameters:
#     -----------
#     G : NetworkX graph
#         An undirected graph with weights between 0 and 1 on edges
#     pos : dict
#         Dictionary with nodes as keys and positions as values (x,y coordinates)
#     scale_factor: float
#         TikZ scale factor
#     node_size : float, optional
#         Size of the nodes (default: 0.8)
#     node_color : str, optional
#         Color of the nodes (default: 'white')
#     font_size : str, optional
#         Size of the node labels (default: 'normalsize')
#     """
#     # Start TikZ code
#     tikz_code = [
#         r"\begin{tikzpicture}",
#         r"  % Define styles",
#         r"  \tikzset{",
#         f"    node/.style={{circle, draw=black, fill={node_color}, minimum size={node_size}cm}},",
#         r"  }",
#         r""
#     ]
    
#     # Add nodes
#     tikz_code.append(r"  % Nodes")
#     for node in G.nodes():
#         if node not in pos:
#             raise ValueError(f"Position for node {node} not found in pos dictionary")
#         x = pos[node][0] * scale_factor
#         y = pos[node][1] * scale_factor
#         tikz_code.append(f"  \\node[node] (node{node}) at ({x:.4f}, {y:.4f}) {{${node}$}};")
    
#     # Add edges
#     tikz_code.append(r"")
#     tikz_code.append(r"  % Edges")
#     for u, v, data in G.edges(data=True):
#         weight = data.get('weight', 0.5)  # Default to 0.5 if no weight specified
        
#         # Ensure weight is between 0 and 1
#         weight = max(0, min(1, weight))
        
#         # Calculate alpha based on weight (convert to opacity in TikZ)
#         alpha = 2 * abs(weight - 0.5)
#         opacity = min(1.0, alpha)  # TikZ opacity must be between 0 and 1
        
#         # Calculate color components (values between 0 and 1)
#         r = 1 - weight  # Red component decreases as weight increases
#         g = 0           # No green component
#         b = weight      # Blue component increases as weight increases
        
#         # Create proper RGB color string for TikZ
#         color_str = f"color={{rgb,1:red,{r:.3f};green,{g:.3f};blue,{b:.3f}}}"
        
#         # Set line style based on weight
#         line_style = "dashed" if weight < 0.5 else "solid"
        
#         # Add edge to TikZ code
#         tikz_code.append(f"  \\draw[{line_style}, {color_str}, opacity={opacity:.2f}, line width=1pt] (node{u}) -- (node{v});")
    
#     # Close TikZ environment
#     tikz_code.append(r"\end{tikzpicture}")
    
#     return "\n".join(tikz_code)

# foo = generate_tikz_code(G, pos, scale_factor=6, node_size=0.8, node_color='white', font_size='normalsize')
# print(foo)

# %% Example usage
if __name__ == "__main__":
    import networkx as nx
    from copy import deepcopy

    G_neg = deepcopy(G) 
    nx.set_edge_attributes(G_neg, {e: -G.edges[e]['weight'] for e in G.edges}, 'weight')

    # Solve MAX-CUT
    seed_bipartition = ['p'+str(i+1) for i in range(list_length[0])]
    # optimal_cuts = max_cut_local_search(G_neg, seed_bipartition=seed_bipartition, max_moves=min(3,len(seed_bipartition)))
    optimal_cuts = max_cut_local_search(G_neg, seed_bipartition=seed_bipartition, max_moves=int(len(G.nodes()) / 2) + 1)
    print(f"Found {len(optimal_cuts)} optimal cuts with value {optimal_cuts[0][2]}:")
    for i, (part0, part1, cut_value) in enumerate(optimal_cuts, 1):
        foo = sorted(part0, key=lambda x: int(x[1:]))
        bar = sorted(part1, key=lambda x: int(x[1:]))
        print(f"Solution {i}: Part A {foo}, Part B {bar}, Cut Value: {cut_value}")

# %% https://claude.site/artifacts/f4cc0b2b-8814-4963-8fb1-549797e3479b
# Cf. https://doi.org/10.1007/978-0-387-30162-4_219
import networkx as nx
import numpy as np
from scipy.optimize import linprog
import itertools

def max_cut_linear_relaxation(G):
    """
    Solve the linear relaxation of the MAX-CUT problem on a weighted graph.
    
    Parameters:
    -----------
    G : networkx.Graph
        A weighted undirected graph. Edges should have a 'weight' attribute.
        If no weight is specified, a default weight of 1 is assumed.
    
    Returns:
    --------
    dict
        A dictionary mapping edge tuples (i,j) to their relaxed x_ij values.
    float
        The optimal objective value.
    """
    # Get list of nodes and edges
    nodes = list(G.nodes())
    n = len(nodes)
    
    # Create a mapping from node to index
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    
    # Initialize the objective function coefficients (all zeros initially)
    c = np.zeros((n * (n-1)) // 2)
    
    # Create a mapping from edge to its position in the flattened variable array
    # Create variables for ALL pairs of vertices, not just existing edges
    edge_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_to_idx[(i, j)] = idx
            edge_to_idx[(j, i)] = idx  # Both directions map to the same variable
            idx += 1
    
    # Populate the objective function coefficients
    # For pairs that aren't edges in G, w_ij is set to 0 (as per the problem statement)
    for i in range(n):
        for j in range(i+1, n):
            u, v = idx_to_node[i], idx_to_node[j]
            idx = edge_to_idx[(i, j)]
            
            # Check if edge exists and get its weight
            if G.has_edge(u, v):
                weight = G[u][v].get('weight', 1.0)  # Default weight is 1.0
            else:
                weight = 0.0  # Non-edges have weight 0
                
            # Maximize sum(w_ij * x_ij), or minimize sum(-w_ij * x_ij)
            c[idx] = -weight
    
    # Create constraints
    A_ub = []
    b_ub = []
    
    # For each 3-cycle i,j,k:
    # 1. x_ij + x_jk + x_ki <= 2
    # 2. x_ij + x_jk - x_ki >= 0 => -x_ij - x_jk + x_ki <= 0
    for i, j, k in itertools.combinations(range(n), 3):
        # First constraint: x_ij + x_jk + x_ki <= 2
        constraint = np.zeros(len(c))
        constraint[edge_to_idx[(i, j)]] = 1
        constraint[edge_to_idx[(j, k)]] = 1
        constraint[edge_to_idx[(k, i)]] = 1
        A_ub.append(constraint)
        b_ub.append(2)
        
        # Second constraint: x_ij + x_jk - x_ki >= 0 => -x_ij - x_jk + x_ki <= 0
        constraint = np.zeros(len(c))
        constraint[edge_to_idx[(i, j)]] = -1
        constraint[edge_to_idx[(j, k)]] = -1
        constraint[edge_to_idx[(k, i)]] = 1
        A_ub.append(constraint)
        b_ub.append(0)
        
        # Cyclic permutations for the second constraint
        constraint = np.zeros(len(c))
        constraint[edge_to_idx[(j, k)]] = -1
        constraint[edge_to_idx[(k, i)]] = -1
        constraint[edge_to_idx[(i, j)]] = 1
        A_ub.append(constraint)
        b_ub.append(0)
        
        constraint = np.zeros(len(c))
        constraint[edge_to_idx[(k, i)]] = -1
        constraint[edge_to_idx[(i, j)]] = -1
        constraint[edge_to_idx[(j, k)]] = 1
        A_ub.append(constraint)
        b_ub.append(0)
    
    # Linear relaxation: 0 <= x_ij <= 1 (bounds in linprog)
    bounds = [(0, 1) for _ in range(len(c))]
    
    # Solve the linear relaxation
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    # Check if the optimization was successful
    if result.success:
        # Convert the solution back to a dictionary mapping edges to values
        solution = {}
        idx_to_edge = {idx: (min(i, j), max(i, j)) for (i, j), idx in edge_to_idx.items() if i < j}
        
        for idx, val in enumerate(result.x):
            if idx in idx_to_edge:
                i, j = idx_to_edge[idx]
                u, v = nodes[i], nodes[j]
                
                # Only include variables for actual edges in the result
                # (non-edges have weight 0 and don't contribute to objective)
                if G.has_edge(u, v) or abs(val) > 1e-6:  # Include if significant value or is an edge
                    solution[(u, v)] = val

    # if result.success:
    #     # Prepare to return x_ij values and the solution
    #     x_values = {}
    #     for i in range(n):
    #         for j in range(i+1, n):
    #             idx = edge_to_idx[(i, j)]
    #             u, v = idx_to_node[i], idx_to_node[j]
    #             x_values[(u, v)] = result.x[idx]
    #             x_values[(v, u)] = result.x[idx]  # Add both directions for convenience
        
    # return x_values, -result.fun, result.x, edge_to_idx, node_to_idx, idx_to_node

        return solution, -result.fun, result.x, edge_to_idx, node_to_idx, idx_to_node  # Negate because we minimized -objective
    else:
        raise ValueError(f"Optimization failed: {result.message}")

# %%
import random

def construct_partition_deterministic(G, x_values, seed_node=None):
    """
    Construct a partition of the graph nodes based on the LP relaxation solution.
    This uses a deterministic approach starting from a seed node.
    
    Parameters:
    -----------
    G : networkx.Graph
        The input graph.
    x_values : dict
        A dictionary mapping edge tuples (i,j) to their relaxed x_ij values.
    seed_node : any, optional
        The node to start the partition from. If None, the first node is used.
        
    Returns:
    --------
    tuple
        A tuple (S, V_S) representing the bipartition of the vertices.
    """
    nodes = list(G.nodes())
    if not nodes:
        return set(), set()
    
    if seed_node is None:
        seed_node = nodes[0]
    
    # Initialize the partition with the seed node
    S = {seed_node}
    V_S = set()
    
    # Assign remaining nodes based on their x_ij values relative to the seed node
    for node in nodes:
        if node == seed_node:
            continue
        
        # If x_seed,node is closer to 1, they should be on opposite sides
        if x_values.get((seed_node, node), 0) >= 0.5:
            V_S.add(node)
        else:
            S.add(node)
    
    return S, V_S

def construct_partition_randomized(G, x_values, seed=None):
    """
    Construct a partition of the graph nodes based on the LP relaxation solution.
    This uses a randomized approach.
    
    Parameters:
    -----------
    G : networkx.Graph
        The input graph.
    x_values : dict
        A dictionary mapping edge tuples (i,j) to their relaxed x_ij values.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns:
    --------
    tuple
        A tuple (S, V_S) representing the bipartition of the vertices.
    """
    if seed is not None:
        random.seed(seed)
    
    nodes = list(G.nodes())
    n = len(nodes)
    
    # Calculate the bias term epsilon for each node
    epsilon = {}
    for node in nodes:
        # Average tendency of this node to be on different side than others
        sum_x = sum(x_values.get((node, other), 0) for other in nodes if other != node)
        epsilon[node] = (sum_x / (n-1 if n > 1 else 1)) - 0.5
    
    # Assign nodes to partitions based on randomized rounding
    S = set()
    V_S = set()
    
    for node in nodes:
        # Probability of putting node in S: 0.5 - epsilon
        # (If epsilon > 0, node tends to be on different sides with others, so less likely in S)
        if random.random() >= 0.5 + epsilon[node]:
            S.add(node)
        else:
            V_S.add(node)
    
    return S, V_S

def construct_best_partition(G, x_values, num_trials=10):
    """
    Try multiple random partitions and return the one with highest cut weight.
    
    Parameters:
    -----------
    G : networkx.Graph
        The input graph.
    x_values : dict
        A dictionary mapping edge tuples (i,j) to their relaxed x_ij values.
    num_trials : int
        Number of random partitions to try.
        
    Returns:
    --------
    tuple
        A tuple (S, V_S) representing the best bipartition of the vertices.
    float
        The cut weight of the best partition.
    """
    best_partition = None
    best_weight = -float('inf')
    
    # Try the deterministic partition
    S, V_S = construct_partition_deterministic(G, x_values)
    weight = calculate_cut_weight(G, S, V_S)
    if weight > best_weight:
        best_weight = weight
        best_partition = (S, V_S)
    
    # Try multiple random partitions
    for i in range(num_trials):
        S, V_S = construct_partition_randomized(G, x_values, seed=i)
        weight = calculate_cut_weight(G, S, V_S)
        if weight > best_weight:
            best_weight = weight
            best_partition = (S, V_S)
    
    return best_partition, best_weight

def calculate_cut_weight(G, S, V_S):
    """
    Calculate the weight of the cut defined by the partition (S, V_S).
    
    Parameters:
    -----------
    G : networkx.Graph
        The input graph.
    S : set
        First set of the partition.
    V_S : set
        Second set of the partition.
        
    Returns:
    --------
    float
        The total weight of edges crossing the cut.
    """
    cut_weight = 0.0
    
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1.0)
        # If the edge crosses the cut (one endpoint in S, the other in V_S)
        if (u in S and v in V_S) or (u in V_S and v in S):
            cut_weight += weight
    
    return cut_weight

# %%
if __name__ == "__main__":

    from copy import deepcopy

    G_neg = deepcopy(G) 
    nx.set_edge_attributes(G_neg, {e: -G.edges[e]['weight'] for e in G.edges}, 'weight')

    # Solve the LP relaxation
    x_values, obj_value, res_x, edge_to_idx, node_to_idx, idx_to_node = max_cut_linear_relaxation(G_neg)
    
    print(f"Relaxed solution objective value: {obj_value:.4f}")
    print("\nRelaxed x_ij values:")
    for edge, value in sorted(x_values.items()):
        if edge[0] < edge[1]:  # Only print each edge once
            print(f"x_{edge} = {value:.4f}")

    # Generate partitions
    S_det, V_S_det = construct_partition_deterministic(G_neg, x_values)
    weight_det = calculate_cut_weight(G_neg, S_det, V_S_det)
    print(f"\nDeterministic partition: S = {S_det}, V\\S = {V_S_det}")
    print(f"Cut weight: {weight_det}")
    
    S_rand, V_S_rand = construct_partition_randomized(G_neg, x_values, seed=42)
    weight_rand = calculate_cut_weight(G_neg, S_rand, V_S_rand)
    print(f"\nRandomized partition: S = {S_rand}, V\\S = {V_S_rand}")
    print(f"Cut weight: {weight_rand}")
    
    best_partition, best_weight = construct_best_partition(G_neg, x_values, num_trials=1000)
    print(f"\nBest partition: S = {best_partition[0]}, V\\S = {best_partition[1]}")
    print(f"Best cut weight: {best_weight}")
    
    # Calculate approximation ratio
    approx_ratio = best_weight / obj_value if obj_value != 0 else float('inf')
    print(f"\nApproximation ratio: {approx_ratio:.4f}")

# %%
import pickle
import datetime

# Get current timestamp for filename
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
pickle_name = filename + "_" + timestamp + ".pkl"

def is_picklable(obj):
    """Test if an object can be pickled"""
    try:
        pickle.dumps(obj)
        return True
    except:
        return False

# Create dictionary with variables that can be pickled
picklable_vars = {}
skipped_vars = []

for key, value in list(globals().items()):
    # Skip special variables and modules
    if (key.startswith('_') or 
        key in ('In', 'Out', 'exit', 'quit', 'get_ipython')):
        continue
    
    # Test if the variable can be pickled
    if is_picklable(value):
        picklable_vars[key] = value
    else:
        skipped_vars.append(key)

# Save to file
with open(pickle_name, 'wb') as file:
    pickle.dump(picklable_vars, file)

print(f"Session saved to {pickle_name}")
print(f"Saved {len(picklable_vars)} variables")

if skipped_vars:
    print(f"Could not save these variables (not picklable): {', '.join(skipped_vars)}")

# %%
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_edge_weight_distribution(G):
    # Extract edge weights from the graph
    weights = [data['weight'] for _, _, data in G.edges(data=True) if 'weight' in data]
    
    if not weights:
        print("No weighted edges found in the graph.")
        return
    
    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(weights, bins=30, alpha=0.7, color='blue', edgecolor='black')
    
    # Add a kernel density estimate
    if len(weights) > 1:  # KDE requires at least 2 data points
        density = True
        plt.hist(weights, bins=30, density=density, alpha=0.4, color='blue', edgecolor='black')
        
        # Add KDE curve
        x = np.linspace(min(weights), max(weights), 1000)
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(weights)
        plt.plot(x, kde(x), 'r-', lw=2)
    
    # Add labels and title
    plt.xlabel('Edge Weight')
    plt.ylabel('Frequency')
    plt.title('Distribution of Edge Weights')
    plt.grid(alpha=0.3)
    
    # Show some statistics as text
    stats_text = f"n = {len(weights)}\nμ = {np.mean(weights):.2f}\nσ = {np.std(weights):.2f}"
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

# %% Example usage
if __name__ == "__main__":

    plot_edge_weight_distribution(G)


# %%
from copy import deepcopy

lo = 0.4
hi = 0.6
# Create a deep copy of the graph and remove/filter edges with weights between lo and hi
G_neg_filtered = deepcopy(G)
edges_to_remove = [(u, v) for u, v, data in G_neg_filtered.edges(data=True) if 'weight' in data and lo <= data['weight'] <= hi]
G_neg_filtered.remove_edges_from(edges_to_remove)
nx.set_edge_attributes(G_neg_filtered, {e: -G.edges[e]['weight'] for e in G.edges}, 'weight')

# %%
if __name__ == "__main__":
    # Solve the LP relaxation
    x_values, obj_value, _, _, _, _ = max_cut_linear_relaxation(G_neg_filtered) # G_neg???
    
    print(f"Relaxed solution objective value: {obj_value:.4f}")
    print("\nRelaxed x_ij values:")
    for edge, value in sorted(x_values.items()):
        if edge[0] < edge[1]:  # Only print each edge once
            print(f"x_{edge} = {value:.4f}")
    
    # Generate partitions
    S_det, V_S_det = construct_partition_deterministic(G_neg_filtered, x_values)
    weight_det = calculate_cut_weight(G_neg_filtered, S_det, V_S_det)
    print(f"\nDeterministic partition: S = {S_det}, V\\S = {V_S_det}")
    print(f"Cut weight: {weight_det}")
    
    S_rand, V_S_rand = construct_partition_randomized(G_neg_filtered, x_values, seed=42)
    weight_rand = calculate_cut_weight(G_neg_filtered, S_rand, V_S_rand)
    print(f"\nRandomized partition: S = {S_rand}, V\\S = {V_S_rand}")
    print(f"Cut weight: {weight_rand}")
    
    best_partition, best_weight = construct_best_partition(G_neg_filtered, x_values, num_trials=10)
    print(f"\nBest partition: S = {best_partition[0]}, V\\S = {best_partition[1]}")
    print(f"Best cut weight: {best_weight}")
    
    # Calculate approximation ratio
    approx_ratio = best_weight / obj_value if obj_value != 0 else float('inf')
    print(f"\nApproximation ratio: {approx_ratio:.4f}")

# %%
