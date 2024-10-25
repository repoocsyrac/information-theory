import numpy as np
import pandas as pd

# Step 0: Load the data from the CSV files into dictionaries
def load_file(path):
    letter_count_data = pd.read_csv(path, names=['letters', 'count'])
    letter_count_dict = letter_count_data.set_index('letters')['count'].to_dict()
    return letter_count_dict

single_letter_counts_dict = load_file('single_counts.csv')
double_letter_counts_dict = load_file('double_counts.csv')
triple_letter_counts_dict = load_file('triple_counts.csv')
quadruple_letter_counts_dict = load_file('quadruple_counts.csv')

# Step 1: Calculate the probabilities
def calculate_probabilities(counts_dict):
    total_letter_count = sum(counts_dict.values())
    probabilities = {}
    for letters, count in counts_dict.items():
        probabilities[letters] = count / total_letter_count
    return probabilities

single_letter_probabilities = calculate_probabilities(single_letter_counts_dict)
double_letter_probabilities = calculate_probabilities(double_letter_counts_dict)
triple_letter_probabilities = calculate_probabilities(triple_letter_counts_dict)
quadruple_letter_probabilities = calculate_probabilities(quadruple_letter_counts_dict)

# Step 2: Calculate the entropies

def calculate_entropy(probabilities):
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

single_letter_entropy = calculate_entropy(single_letter_probabilities.values())
double_letter_entropy = calculate_entropy(double_letter_probabilities.values())
triple_letter_entropy = calculate_entropy(triple_letter_probabilities.values())
quadruple_letter_entropy = calculate_entropy(quadruple_letter_probabilities.values())

# H(X_1)
H_X1 = single_letter_entropy
print(f"H(X_1): {H_X1}")

# H(X_2 | X_1) = H(X_2, X_1) - H(X_1)
H_X2 = double_letter_entropy - H_X1
print(f"H(X_2): {H_X2}")

# H(X_3 | X_2, X_1) = H(X_3, X_2, X_1) - H(X_2, X_1)
H_X3 = triple_letter_entropy - double_letter_entropy
print(f"H(X_3): {H_X3}")

# H(X_4 | X_3, X_2, X_1) = H(X_4, X_3, X_2, X_1) - H(X_3, X_2, X_1)
H_X4 = quadruple_letter_entropy - triple_letter_entropy
print(f"H(X_4): {H_X4}")