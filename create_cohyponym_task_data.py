import numpy as np

from src import config

CORPUS_NAME = 'childes-20180319'
# CORPUS_NAME = 'tasa-20181213'


if __name__ == '__main__':
    for vocab_size in config.Corpus.vocab_sizes:
        # vocab
        p = config.Dirs.vocab / '{}_{}_vocab.txt'.format(config.Corpus.name, config.Corpus.num_vocab)
        if not p.exists():
            raise RuntimeError('{} does not exist'.format(p))
        vocab = np.loadtxt(p, 'str').tolist()
        #
        for data_name1 in ['semantic', 'syntactic']:
            # load all probes
            in_path = config.Dirs.data / 'categories' / data_name1 / '{}_complete.txt'.format(CORPUS_NAME)
            probes, probe_cats = np.loadtxt(in_path, dtype='str').T
            cat2probes = {cat: probes[probe_cats == cat].tolist() for cat in probe_cats}
            # write probes if in vocab
            out_fname = '{}_{}.txt'.format(CORPUS_NAME, vocab_size)
            out_path = config.Dirs.relations / 'cohyponyms' / data_name1 / out_fname
            print('Writing {}'.format(out_path))
            with out_path.open('w') as f:
                for probe, cat in zip(probes, probe_cats):
                    if probe in vocab:
                        cohyponyms = ' '.join([cohyponym for cohyponym in cat2probes[cat]
                                              if cohyponym != probe and cohyponym in vocab])
                        if not cohyponyms:
                            continue
                        f.write('{} {}\n'.format(probe, cohyponyms))