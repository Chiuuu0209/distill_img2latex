import numpy as np
# import os
import sys
from pathlib import Path

from nltk.translate.bleu_score import corpus_bleu

from model import get_all_formulas, get_split

def BLEU(predict_path):
    formula_file = Path(__file__).resolve().parents[0] / "data" / "im2latex_formulas.norm.new.lst"
    test_filter = Path(__file__).resolve().parents[0] / "data" / "im2latex_test_filter.lst"
    all_formulas = get_all_formulas(formula_file)
    _ , test_formulas = get_split(all_formulas, test_filter)

    candidates, references = [], []
    lines = open(predict_path, 'r').readlines()
    for i in range(len(lines)):
        candidates.append(lines[i].rstrip().split(" "))
        references.append([test_formulas[i]])
    score = corpus_bleu(references, candidates)
    return score

if __name__ == '__main__':
    predict_path = sys.argv[1]
    score = BLEU(predict_path)
    print('BLEU score: ', score)
