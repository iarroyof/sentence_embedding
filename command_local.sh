parallel 'in=$(echo {5}|cut -d'.' -f 2); python sts.py --stop {3} --format wisse --idfmodel local --tfidf {1} --localw {2} --embedmodel {4} --input {5} --comb {6} --dist {7} --output results_revision/$(echo {1}-{2}-{3}-{4/}-${in}-{6}-{7}.out | sed "s/--/-wst-/g")' ::: `cat tfidf.l` ::: `cat localw.l` ::: `cat stop.l` ::: `cat embedmodel.l` ::: `cat input.l` ::: `cat comb.l` ::: `cat dist.l`