import numpy as np
from pdb import set_trace as st

files="sick/sts.input.sick.txt train-2013/STS.input.FNWN.txt train-2013/STS.input.OnWN.txt sts2016-english-with-gs-v1.0/STS2016.input.answer-answer.txt sts2016-english-with-gs-v1.0/STS2016.input.headlines.txt sts2016-english-with-gs-v1.0/STS2016.input.plagiarism.txt sts2016-english-with-gs-v1.0/STS2016.input.postediting.txt sts2016-english-with-gs-v1.0/STS2016.input.question-question.txt".split()

print("STS datasets statistics...")
#print(r"Dataset & Mean & Median & std. dev & Mean diff & Median diff & std. dev. \\")
print("Dataset \tMean \tMedian \tstd. \tMean diff \tMedian diff \tstd. \\\\")

for filename in files:
    with open(filename) as f:
        double=f.readlines()
        lengths = []
        diffs = []
        for line in double:
            s = line.split("\t")
            a, b = s[0], s[1]
            #lengths.append((len(a.split()), len(b.split())))
            lengths.append(len(a.split()))
            lengths.append(len(b.split()))
            diffs.append(abs(len(a.split()) - len(b.split())))

    #print("%s & %d & %d & %f & %d & %d & %f \\\\" % (filename.split(".")[-2], 
    print("%s \t %d \t %d \t %f \t %d \t %d \t %f \\\\" % (filename.split(".")[-2],
                                        np.mean(lengths), 
                                        np.median(lengths),
                                        np.std(lengths)/np.mean(lengths),
                                        np.mean(diffs),
                                        np.median(diffs),
                                        np.std(diffs)/np.mean(diffs))
          )
