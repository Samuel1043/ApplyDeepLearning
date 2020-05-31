import ckiptagger
import collections
from tqdm import tqdm
import os


class Tokenizer:
    def __init__(self, model_dir):
        print(f'[*] Creating CKIP tokenizer from {model_dir}...', end='', flush=True)
        self._ws = ckiptagger.WS(model_dir)
        self._pos = ckiptagger.POS(model_dir)
        self._pos_punc_class_suffix = 'CATEGORY'

        print('done')

    def __call__(self, text, remove_punc=False):
        tokens = self._ws([text])[0]
        if not remove_punc:
            return tokens

        pos = self._pos([tokens])[0]
        tokens = [t for t, p in zip(tokens, pos)
                  if not p.endswith(self._pos_punc_class_suffix)]

        return tokens


def compute_em(ans, pred):
    def em(a, p):
        return int(''.join(a) == ''.join(p))

    return max([em(a, pred) for a in ans])


def compute_f1(ans, pred):
    def f1(a, p):
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        if len(a) == 0 or len(p) == 0:
            return int(''.join(a) == ''.join(p))

        common = collections.Counter(a) & collections.Counter(p)
        tp = sum(common.values())
        if tp == 0:
            return 0
        precision = tp / len(p)
        recall = tp / len(a)

        return (2 * precision * recall) / (precision + recall)

    return max([f1(a, pred) for a in ans])


def compute_metric(ans, pred, tokenizer):
    ans = [tokenizer(a, remove_punc=True) for a in ans]
    pred = tokenizer(pred, remove_punc=True)

    return {
        'em': compute_em(ans, pred),
        'f1': compute_f1(ans, pred)
    }


def compute_metrics(answers, predictions, tokenizer):
    metrics = []
    for id_ in tqdm(list(answers.keys()), desc='[*] Evaluating', dynamic_ncols=True,position=0,leave=True):
        if id_ not in predictions:
            print(f'[!] Cannot find answer for id {id_} in model predictions')
            continue
        answerable = answers[id_]['answerable']
        prediction = predictions[id_]
        metric = compute_metric(answers[id_]['answers'], prediction, tokenizer)
        metrics.append({
            **metric,
            'answerable': answerable,
            'answerable_acc': int(answerable ^ (prediction == ''))
        })

    n_total = len(metrics)
    n_answerable = len([m for m in metrics if m['answerable']])
    n_unanswerable = n_total - n_answerable
    result = {
        'overall': {
            'count': n_total,
            'em': sum([m['em'] for m in metrics]) / n_total,
            'f1': sum([m['f1'] for m in metrics]) / n_total
        },
        'answerable': {
            'count': n_answerable,
            'em': sum([m['em'] for m in metrics if m['answerable']]) / n_answerable,
            'f1': sum([m['f1'] for m in metrics if m['answerable']]) / n_answerable
        },
        'unanswerable': {
            'count': n_unanswerable,
            'em': sum([m['em'] for m in metrics if not m['answerable']]) / n_unanswerable,
            'f1': sum([m['f1'] for m in metrics if not m['answerable']]) / n_unanswerable
        },
        'answerable accuracy': sum(m['answerable_acc'] for m in metrics) / n_total
    }

    return result

def collect_answers(data):
    answers = {}
    for d in data:
        for p in d['paragraphs']:
            for qa in p['qas']:
                answers[qa['id']] = {
                    'answerable': qa['answerable'],
                    'answers': [a['text'] for a in qa['answers']]
                }

    return answers
def collect_answers_fromfeat(data):
    answers = {}
    for d in data:        
      answers[d.qa_id] = {
          'answerable': d.answerable,
          'answers': [d.answer['text']]
      }

    return answers
