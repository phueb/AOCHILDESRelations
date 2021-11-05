

from TwoProcessTheory.analysis.aggregator import Aggregator

ag = Aggregator()
df = ag.make_df()

# clean df
df.drop(df[df['neg_pos_ratio'] == 0.0].index, inplace=True)
df.drop(df[df['embedder'] == 'random_normal'].index, inplace=True)


avg_nov_score = df[df['regime'] == 'novice']['score'].mean()
std_nov_score = df[df['regime'] == 'novice']['score'].std()
avg_exp_score = df[df['regime'] == 'expert']['score'].mean()
std_exp_score = df[df['regime'] == 'expert']['score'].std()

print(avg_nov_score)
print(std_nov_score)
print()
print(avg_exp_score)
print(std_exp_score)