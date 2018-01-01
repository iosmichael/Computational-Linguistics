'''
Created on Sep 30, 2015
Modified on Oct 11, 2017
@author: tvandrun, michael liu
'''

from langmod import *
from gtmod import *

test_file_name = "test-smallish.txt"

raw_test_words = re.findall(r'[a-z][a-z\']*|\d+|[!$%*()\-:;\"\',.?]', file(test_file_name).read().lower())
test_text = [transform(w) for w in raw_test_words]

models = [ConstantLanguageModel(),
          UnigramGTModel(lm_data),
          UnigramCutoffModel(lm_data)]
# interpolated between constant and unigram
models.append(InterpolatedLanguageModel([UnigramCutoffModel(lm_data),BigramLanguageModel(lm_data),TrigramLanguageModel(lm_data)]))
# interpolated among uni-, bi-, and trigram each with Laplace smoothing
#models.append(InterpolatedLanguageModel([models[2],models[4], models[5]]))



infinity = float('inf')

for model in models :
    print "==%s==" % model.kind_of_model()

    vocab_prob = model.p("NUM", []) + model.p("OOV", []) + model.p("PNCT", [])
    for w in vocab :
        vocab_prob += model.p(w, [])
    print "total probability of vocab: ", vocab_prob

    total_log_prob = 0 
    history = []
    for w in test_text:
        prob = model.p(w, history)
        #print w, lm_data.fd_unigrams[w], prob
        if prob == 0 :
            log_prob = infinity
        else :
            log_prob = math.log(prob) 
        total_log_prob += log_prob
        history.append(w)
    if total_log_prob != infinity :
        print "perplexity: %s" % math.exp(-total_log_prob/len(test_text))
    else :
        print "infinite perplexity"
        
