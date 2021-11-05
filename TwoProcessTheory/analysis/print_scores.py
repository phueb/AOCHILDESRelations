from TwoProcessTheory.analysis.aggregator import Aggregator

CORPUS = 'childes-20180319'
NUM_VOCAB = 4096
EMBEDDER = 'lstm'  # rnd_normal
EMBED_SIZE = 200
TASK = 'cohyponyms_semantic'
PROCESS = 'expert'

ag = Aggregator()
df = ag.make_df()
print('Columns:')
for col in df.columns:
    print(col)

# filter
filtered_df = df[(df['embedder'] == EMBEDDER) &
                 (df['task'] == TASK) &
                 (df['corpus'] == CORPUS) &
                 (df['num_vocab'] == NUM_VOCAB) &
                 (df['regime'] == PROCESS) &
                 (df['embed_size'] == EMBED_SIZE)]

print()
print('Length of complete data={}'.format(len(df)))
print('Trained embedders={}'.format(df['embedder'].unique()))
print()
print('Data for given criteria:')
print(filtered_df[['job_name', 'score']])