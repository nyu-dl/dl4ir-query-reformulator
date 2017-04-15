import parameters as prm
import numpy as np

def run(q, d_input, wordss, metric_base, options, D_gt=[]):
    out = np.zeros((prm.max_feedback_docs, prm.max_terms_per_doc), dtype=np.float32)
    
    for i, words in enumerate(wordss):

        queries = [q.decode('ascii', 'ignore') + ' ' + word.decode('ascii', 'ignore') for word in words]
        candss = options['engine'].get_candidates(queries, prm.max_candidates + 1)
        for j, cands in enumerate(candss):
                    
            # don't add the input document in case it was returned.
            if d_input in cands:
                del cands[d_input]
            else:
                # remove the last doc so the max number of candidates is prm.max_candidates
                if len(cands) > 0:
                    cands.popitem(last=True)

            cands_set = set(cands.keys())
            intersec = len(set(D_gt) & cands_set)
            metric_new = intersec / max(1., float(len(D_gt)))
            
            if (metric_new - metric_base) / max(0.0001, metric_base) > prm.supervised_threshold:
                out[i,j] = 1.

    return out
