#parallel 'in=$(echo $DATA/sts/benchmark_2017/sts-input.train.txt |cut -d'.' -f 2); python sts.py --stop {3} --format wisse --idfmodel local --tfidf {1} --localw {2} --embedmodel {4} --input $DATA/sts/benchmark_2017/sts-input.train.txt --comb {5} --dist {6} --output results_revision/local_$(echo {1}-{2}-{3}-{4/}-${in}-{5}-{6}.out | sed "s/--/-wst-/g")' ::: `cat tfidf.l` ::: `cat localw.l` ::: `cat stop.l` ::: `cat embedmodel.l` ::: `cat comb.l` ::: `cat dist.l`
# STS benchmark and sick
#parallel 'in=$(echo {5}|cut -d'.' -f 2); python sts.py --stop {3} --format wisse --idfmodel local --tfidf {1} --localw {2} --embedmodel {4} --input {5} --comb {6} --dist {7} --output results_revision/local-$(echo {1}-{2}-{3}-{4/}-${in}-{6}-{7}.out | sed "s/--/-wst-/g")' ::: `cat tfidf.l` ::: `cat localw.l` ::: `cat stop.l` ::: `cat embedmodel.l` ::: `cat input.l` ::: `cat comb.l` ::: `cat dist.l`
# STS benchmark unweighted
#parallel 'in=$(echo {3}|cut -d'.' -f 2); python sts.py --stop {1} --format wisse --idfmodel none --tfidf none --localw none --embedmodel {2} --input {3} --comb {4} --dist {5} --output results_revision/unweighted-$(echo {1}-{2/}-${in}-{4}-{5}.out | sed "s/--/-wst-/g")' ::: `cat stop.l` ::: `cat embedmodel.l` ::: `cat input.l` ::: `cat comb.l` ::: `cat dist.l`
# STS 2016
#parallel 'in=$(echo {5}|cut -d'.' -f 4); python sts.py --stop {3} --format wisse --idfmodel local --tfidf {1} --localw {2} --embedmodel {4} --input {5} --comb {6} --dist {7} --output results_revision/local-$(echo {1}-{2}-{3}-{4/}-${in}-{6}-{7}.out | sed "s/--/-wst-/g")' ::: `cat tfidf.l` ::: `cat localw.l` ::: `cat stop.l` ::: `cat embedmodel.l` ::: `cat _input.l` ::: `cat comb.l` ::: `cat dist.l`
# STS 2016 unweighted
#parallel 'in=$(echo {3}|cut -d'.' -f 4); python sts.py --stop {1} --format wisse --idfmodel none --tfidf none --localw none --embedmodel {2} --input {3} --comb {4} --dist {5} --output results_revision/unweighted$(echo -{1}-{2/}-${in}-{4}-{5}.out | sed "s/--/-wst-/g")' ::: `cat stop.l` ::: `cat embedmodel.l` ::: `cat _input.l` ::: `cat comb.l` ::: `cat dist.l`
# STS 2016 final
parallel 'in=$(echo {3}|cut -d'.' -f 4|sed 's/-/_/g'); python3 sts.py --stop {2} --format wisse --idfmodel local --tfidf tfidf --localw {1} --embedmodel links/fstx-300d --input {3} --comb sum --dist cosine --output results_revision/local-$(echo tfidf-{1}-{2}-fstx-300d-${in}-sum-cosine.out | sed "s/--/-wst-/g")' ::: `cat _localw.l` ::: `cat stop.l` ::: `cat _input.l`
