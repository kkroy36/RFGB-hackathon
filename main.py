from GradientBoosting import GradientBoosting
import random

def main():
    '''main method that runs boosting'''

    bk = ["smokes(+Person)","friends(+Person,-Person)","friends(-Person,+Person)","cancer(Person)"]
    facts = []
    pos = []
    neg = []

    with open("Toy-Cancer/train/train_facts.txt") as f:
        facts = f.read().splitlines()
        facts = [item[:-1] for item in facts]

    with open("Toy-Cancer/train/train_pos.txt") as p:
        pos = p.read().splitlines()
        pos = [item[:-1] for item in pos]

    with open("Toy-Cancer/train/train_neg.txt") as n:
        neg = n.read().splitlines()
        neg = [item[:-1] for item in neg]

    ratio = len(neg)/float(len(pos))

    if ratio > 1:
        prob = 2*len(pos)/float(len(neg))
        neg = [item for item in neg if random.random() < prob]

    clf = GradientBoosting(treeDepth = 2, trees = 3)
    clf.setTargets(["cancer"])
    clf.learn_clf(facts,pos,neg,bk)

main()
