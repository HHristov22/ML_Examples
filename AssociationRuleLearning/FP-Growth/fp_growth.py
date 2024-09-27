# FP-Growth Algorithm

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('./AssociationRuleLearning/FP-Growth/Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training the FP-Growth model on the dataset
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

# Encoding the transactions into a format suitable for FP-Growth
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Applying FP-Growth algorithm
frequent_itemsets = fpgrowth(df, min_support=0.003, use_colnames=True)

# Generating association rules
from mlxtend.frequent_patterns import association_rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=3)

# Visualising the results

## Displaying the first results coming directly from the output of the FP-Growth function
# print(frequent_itemsets.head())

## Putting the results well organised into a Pandas DataFrame
def inspect_fp(results):
    lhs = [list(result['antecedents'])[0] for result in results]
    rhs = [list(result['consequents'])[0] for result in results]
    supports = [result['support'] for result in results]
    lifts = [result['lift'] for result in results]
    return list(zip(lhs, rhs, supports, lifts))

rules_filtered = rules[(rules['antecedents'].apply(lambda x: len(x)) == 1) & (rules['consequents'].apply(lambda x: len(x)) == 1)]

## Displaying the results sorted by descending supports
results_in_DataFrame = pd.DataFrame(inspect_fp(rules_filtered.to_dict('records')), columns=['Product 1', 'Product 2', 'Support', 'Lift'])
print(results_in_DataFrame.nlargest(n=10, columns='Lift'))


## Filtering to ensure unique combinations of products
def inspect_fp_unique(results):
    seen = set()  # Keep track of seen combinations
    unique_results = []
    for result in results:
        lhs = list(result['antecedents'])[0]
        rhs = list(result['consequents'])[0]
        pair = tuple(sorted([lhs, rhs]))  # Sort to ensure that (A, B) and (B, A) are treated as the same pair
        if pair not in seen:
            seen.add(pair)
            supports = result['support']
            lifts = result['lift']
            unique_results.append((lhs, rhs, supports, lifts))
    return unique_results

rules_filtered_unique = rules[(rules['antecedents'].apply(lambda x: len(x)) == 1) & (rules['consequents'].apply(lambda x: len(x)) == 1)]

## Displaying the results without duplicates
unique_results_in_DataFrame = pd.DataFrame(inspect_fp_unique(rules_filtered_unique.to_dict('records')), columns=['Product 1', 'Product 2', 'Support', 'Lift'])
print(unique_results_in_DataFrame.nlargest(n=10, columns='Lift'))