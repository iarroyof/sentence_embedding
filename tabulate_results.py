import os
from itertools import chain
import subprocess
import sys
# global-tfidf-binary-ost-dep2vec-300d-bmk_dev-avg-cosine.out

printfile = True
results = "results_revision/"
<<<<<<< HEAD
DATA = "/almac/ignacio/data"
#DATA = "../data"
=======
#DATA = "/almac/ignacio/data"
DATA = "../data/sts_all"
>>>>>>> 64115a9e1a185fec4c4716a0fc7e2613a0efde00
out = "out_tab.csv"
# cat out_tab.csv | parallel -- > out_all.out

#filters = ["local", "trts"]
filters = []
with open(out, "w") as of:
    names = os.listdir(results)
    if filters != []:
        filtered = list(
                chain.from_iterable(
                    [[n for n in names if filtro in n]
                         for filtro in filters]) )
    else:
        filtered = names

    for j in filtered:
        f = j.split("-")[6]
        if f == "bmk_dev" and not "bmk" in filters: # and not "dev" in filters:
            gs = DATA + "/benchmark_2017/sts-gs.dev.txt"
        elif f == "bmk_train" and not "bmk" in filters: # and not "train" in filters:
            gs = DATA + "/benchmark_2017/sts-gs.train.txt"
        elif f == "bmk_test" and not "bmk" in filters: # and not "test" in filters:
            gs = DATA + "/benchmark_2017/sts-gs.test.txt"
        elif f == "sick_trial" and not "sick" in filters: # and not "trial" in filters:
            gs = DATA + "/sick/sts.gs.sick-trial.txt"
        elif f == "sick_trts" and not "sick" in filters: # and not "trts" in filters:
            gs = DATA + "/sick/sts.gs.zick.txt"
        elif f == "answer-answer" and not "answer" in filters: # and not "trts" in filters:
<<<<<<< HEAD
            gs = DATA + "/sts/sts2016-english-with-gs-v1.0/STS2016.gs.answer-answer.txt"
        elif f == "headlines" and not "headlines" in filters: # and not "trts" in filters:
            gs = DATA + "/sts/sts2016-english-with-gs-v1.0/STS2016.gs.headlines.txt"
        elif f == "plagiarism" and not "plagiarism" in filters: # and not "trts" in filters:
            gs = DATA + "/sts/sts2016-english-with-gs-v1.0/STS2016.gs.plagiarism.txt"
        elif f == "postediting" and not "postediting" in filters: # and not "trts" in filters:
            gs = DATA + "/sts/sts2016-english-with-gs-v1.0/STS2016.gs.postediting.txt"
        elif f == "question-question" and not "question" in filters: # and not "trts" in filters:
            gs = DATA + "/sts/sts2016-english-with-gs-v1.0/STS2016.gs.question-question.txt"
=======
            gs = DATA + "/sts2016-english-with-gs-v1.0/STS2016.gs.answer-answer.txt"
        elif f == "headlines" and not "headlines" in filters: # and not "trts" in filters:
            gs = DATA + "/sts2016-english-with-gs-v1.0/STS2016.gs.headlines.txt"
        elif f == "plagiarism" and not "plagiarism" in filters: # and not "trts" in filters:
            gs = DATA + "/sts2016-english-with-gs-v1.0/STS2016.gs.plagiarism.txt"
        elif f == "postediting" and not "postediting" in filters: # and not "trts" in filters:
            gs = DATA + "/sts2016-english-with-gs-v1.0/STS2016.gs.postediting.txt"
        elif f == "question-question" and not "question" in filters: # and not "trts" in filters:
            gs = DATA + "/sts2016-english-with-gs-v1.0/STS2016.gs.question-question.txt"
>>>>>>> 64115a9e1a185fec4c4716a0fc7e2613a0efde00
        else:
            gs = ""

        if gs != "":
            command=("gs={0}; sys={1};" #if [[ $(wc -l $gs | cut -d' ' -f1) = $(wc -l $sys | cut -d' ' -f1)  ]];then"
                    " p=$(perl correlation-noconfidence.pl $gs $sys | cut -d' ' -f2); name=$(echo $(basename $sys) | "
                    "sed 's/-/ /g'); echo ${{name%.out}} ${{p//-}};\n") #fi\n")
            if printfile:
                of.write (command.format(gs, results + j ))
            else:
                print(command[:-1].format(gs, results + j ))
        else:
            continue
