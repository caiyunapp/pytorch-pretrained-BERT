import os
import json
import itertools
from itertools import product, permutations
from random import sample

from pytorch_pretrained_bert.tokenization import BertTokenizer
from child_lib import *


BERT_DIR = '/nas/pretrain-bert/pretrain-pytorch/bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained('/nas/pretrain-bert/pretrain-pytorch/bert-base-uncased-vocab.txt')


def assert_in_bert_vocab(tokens):
    for token in tokens:
        if isinstance(token, str):  # entities
            assert token.lower() in tokenizer.vocab, token + '->' + str(tokenizer.tokenize(token))
        elif isinstance(token, tuple):  # relations
            assert len(token) == 2, str(token)
            for rel in token:
                rel = rel.split('..')[0]
                assert rel in tokenizer.vocab, rel + '->' + str(tokenizer.tokenize(rel))


male_names = ['James', 'John', 'Robert', ]#'Michael', 'David', 'Paul', 'Jeff', 'Daniel', 'Charles', 'Thomas']
female_names = ['Mary', 'Linda', 'Jennifer', ]#'Maria', 'Susan', 'Lisa', 'Sandra', 'Barbara', 'Patricia', 'Elizabeth']
people_names = (male_names, female_names)
assert_in_bert_vocab(male_names)
assert_in_bert_vocab(female_names)

people_adj_relations = (
    ('taller..than', 'shorter..than'), 
#     ('thinner..than', 'fatter..than'),   # fatter not in BERT vocab
    ('younger..than', 'older..than'), 
#     ('stronger..than', 'weaker..than'), 
#     ('faster..than', 'slower..than'),
#     ('richer..than', 'poorer..than')
)

rel2entypes = {
#     spatial_relations: [fruits, animals, people_names],
    people_adj_relations: [people_names],
#     animal_adj_relations: [animals],
#     object_adj_relations: [fruits, animals]
}


def comparative2superlative(comparative_form, structured=False):
    assert comparative_form.endswith('er'), comparative_form
    superlative_form = 'the ' + comparative_form[:-2] + 'est' \
        if not structured else 'the ' + comparative_form + ' st'
    return superlative_form


def make_relational_atoms(relational_template, entities, relations):
    neg_relations = ["isn't " + r for r in relations]
    relations = ["is " + r for r in relations]
    atoms = [relational_template.format(ent0=ent0, ent1=ent1, rel=rel) 
             for ent0, ent1, rel in [entities + relations[:1], reverse(entities) + reverse(relations)[:1]]]
    atoms += [relational_template.format(ent0=ent0, ent1=ent1, rel=rel) 
              for ent0, ent1, rel in [entities + reverse(neg_relations)[:1], reverse(entities) + neg_relations[:1]]]
    return atoms


transitive_P_template = '{ent0} {rel} {ent1} .'
transitive_wh_QA_template = '{which} is {pred} ? {ent} .'
transitive_yesno_QA_template = 'is {ent0} {rel} {ent1} ? {ans} .'

def make_transitive(P_template, wh_QA_template, yesno_QA_template, join_template,
                   index=-1, orig_sentence='', entities=["John", "Mary", "Susan"], entity_substitutes=None, determiner="", 
                   relations=('taller..than', 'shorter..than'), maybe=True, structured=False,
                   packed_predicates=["pred0/~pred0", "pred1/~pred1"], predicate_substitutes=None,
                   predicate_dichotomy=True, reverse_causal=False):
    if entities[0].islower():
        entities = ['the ' + e for e in entities]
#     print('relations =', relations)
    relations, predicates = ([r.replace('..', ' ') for r in relations], [r.split('..')[0] for r in relations]) \
        if '..' in relations[0] else ([r.split('/')[0] for r in relations], [r.split('/')[-1] for r in relations])
#     print('relations =', relations, 'predicates =', predicates)
    predicates = [comparative2superlative(p, structured=structured) for p in predicates]
    
    P0_entities, P1_entities = ([entities[0], entities[1]], [entities[1], entities[2]]) \
        if not maybe else ([entities[0], entities[1]], [entities[0], entities[2]])
    P0 = make_relational_atoms(P_template, P0_entities, relations)
    P1 = make_relational_atoms(P_template, P1_entities, relations)
        
    wh_pronoun = 'which' if entities[0].startswith('the') else 'who'
    wh_QA = [wh_QA_template.format(which=wh_pronoun, pred=pred, ent=ent) 
             for pred, ent in [(predicates[0], mask(entities[0])), (predicates[-1], mask(entities[-1] if not maybe else 'unknown'))]]
    
    def _maybe(s):
         return s if not maybe else 'maybe'
    yesno_entities = (entities[0], entities[-1]) if not maybe else (entities[1], entities[-1])
    yesno_QA = [yesno_QA_template.format(ent0=ent0, ent1=ent1, rel=rel, ans=ans) 
                for ent0, ent1, rel, ans in [
                    (yesno_entities[0], yesno_entities[-1], relations[0], mask(_maybe('yes'))), 
                    (yesno_entities[0], yesno_entities[-1], relations[-1], mask(_maybe('no'))),
                    (yesno_entities[-1], yesno_entities[0], relations[-1], mask(_maybe('yes'))),
                    (yesno_entities[-1], yesno_entities[0], relations[0], mask(_maybe('no')))]]
    
    Ps = [(p0, p1) for p0, p1 in list(product(P0, P1)) + list(product(P1, P0))]
    QAs = wh_QA + yesno_QA
    
    def get_rel(atom):
        for rel in relations:
#             assert rel.startswith('is')
            rel = rel.split()[0]  # "taller than" -> "taller"
            if rel in atom:
                return rel
        assert False
    sentences = [p0 + ' ' + p1 + ' ||| ' + qas for (p0, p1), qas in product(Ps, QAs)
                 if not structured or get_rel(p0) == get_rel(p1) == get_rel(qas)]
#     sentences = [s.replace('er st ', 'est ') for s in sentences]
    return sentences


def make_sentences(maybe=True, structured=False):
    sentence_groups = []
    maybe = False
    for relations, entity_types in rel2entypes.items():
        sentences = []
        ent_tuples = []
        for entities in entity_types:
            if isinstance(entities, list):
                ent_tuples += permutations(entities, 3)
            else:
                assert isinstance(entities, tuple) and len(entities) == 2  # people_names
                ent_tuples += permutations(entities[0] + entities[1], 3)
        for (rel, ent_tuple) in product(relations, ent_tuples):
            sentences += make_transitive(transitive_P_template, transitive_wh_QA_template, transitive_yesno_QA_template, None, 
                                entities=list(ent_tuple), relations=rel, maybe=False, structured=True)
            if maybe:
                sentences += make_transitive(transitive_P_template, transitive_wh_QA_template, transitive_yesno_QA_template, None, 
                                    entities=list(ent_tuple), relations=rel, maybe=True, structured=True)
        # sample(sentences, 20)
        # logger.info('num_sent = %d -> %d' % (len(sentences), len(set(sentences))))
        sentence_groups.append(sentences)
    return sentences
