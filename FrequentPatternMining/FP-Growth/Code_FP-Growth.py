

#Generating Frequent itemsets using FP-Growth Algorithm

#Using oragnge3-associate library
from orangecontrib.associate.fpgrowth import * 

#dataset
data = [['A','B','C'        ],
		[    'B',    'D','E'],
		['A','B',    'D','E'],
		['A'                ],
		['A','B','C','D','E'],
		['A',            'E'],
		['A',    'C'        ]]

#Generating Frequent itemsets
itemsets = list(frequent_itemsets(data,0.4))
print('\n\nFrequent Items:---------\n')
print('\nOutput Format\n(frozenset(Frequent Itemset), Support Count)\n\n')

print(*itemsets, sep="\n")

#Generating Association Rules
itemsets = dict(frequent_itemsets(data, .4))
rules =list(association_rules(itemsets, .9))
print('\n\nAsscoiation Rule:---------\n')
print('\nOutput Format\n(frozenset(Antecedent), frozenset(Consequent), Support Count, Confidence)\n\n')
print(*rules, sep="\n")