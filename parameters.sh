# Help sts.py
#  --idfmodel IDFMODEL   Input file containing IDF pre-trained weights. If not
#                        provided, all word vector weights will be set to 1.0.
#                        If 'local' tf-idf weights will be computed locally
#                        from the input file (pickled sklearn object).
#  --embedmodel EMBEDMODEL
#                        Input file containing word embeddings model (binary
#                        and text are allowed).
#  --output OUTPUT       Output file containing the sentence embeddings.
#  --input INPUT         Input file containing a sentence by row.
#  --comb COMB           Desired word vector combination for sentence
#                        representation {sum, avg}. (default = 'sum').
#  --suffix [SUFFIX]     A suffix to be added to the output file (default =
#                        '').
#  --tfidf TFIDF         To predict TFIDF complete weights ('tfidf') or use
#                        only partial IDFs ('idf'). (default = 'tfidf').
#  --localw LOCALW       TFIDF word vector weights computed locally from the
#                        input file of sentences {freq, binary, sublinear}
#                        (default='none').
#  --stop                Toggles stripping stop words in locally computed word
#                        vector weights.
#  --format FORMAT       The format of the embedding model file: {binary, text,
#                        wisse}. default = 'binary'.
#  --ngrams NGRAMS       The n-gram limit specified as, e.g., 3 for 1-grams,
#                        2-grams and 3-grams, considered to obtain TF-IDF
#                        weights. Default = 1.
#  --njobs NJOBS         The number of jobs to compute similarities of the
#
# python sts.py --format wisse --input $DATA/sts/benchmark_2017/sts-input.dev.txt --idfmodel local --embedmodel {1} \
#               --localw binary --ngrams {2} --output results_revision/sts_{1/}_{2}.out' ::: ../data/word2vec/indexed_w2v_En_vector_space_H300 ../data/fastText/fstx_300d_indexed ::: 1-1 1-2 1-3 2-2 2-3 2-4
#$input=$DATA/sts/benchmark_2017/sts-input.dev.txt


# --tfidf
for t in `echo "tfidf idf None"`; do echo $t >> tfidf.l; done

# --localw
for t in `echo "freq binary sublinear"`; do echo $t >> localw.l; done

# --dist
for t in `echo "cosine euclid manha"`; do echo $t >> dist.l; done

# --stop
echo >> stop.l
echo "--stop" >> stop.l

# --idfmodel
ls $DATA/INEXQA2012corpus/*.pk > idfmodel.l

# --embedmodel
for d in `ls -d $DATA/fastText/*/`; do echo $d >> embedmodel.l; done
for d in `ls -d $DATA/word2vec/*/`; do echo $d >> embedmodel.l; done
for d in `ls -d $DATA/glove/*/`; do echo $d >> embedmodel.l; done
ls -d $DATA/dependency_word2vec/*/  >> embedmodel.l

# --input
echo $DATA/sts/benchmark_2017/sts-input.dev.txt >> input.l
echo $DATA/sts/benchmark_2017/sts-input.test.txt >> input.l
echo $DATA/sts/sick/sts.input.sick-trial.txt >> input.l

# --comb
echo "sum" >> comb.l
echo "avg" >> comb.l

# --output

# local output name
parallel 'st=$(echo {3}|sed "s/--//g"); in=$(echo {5}|cut -d'.' -f 2); echo {1}-{2}-${st}-{4/}-${in}-{6}-{7}.out | sed "s/--/-wst-/g"' ::: `cat tfidf.l` ::: `cat localw.l` ::: `cat stop.l` ::: `cat embedmodel.l` ::: `cat input.l` ::: `cat comb.l` ::: `cat dist.l`
# local command
parallel 'in=$(echo {5}|cut -d'.' -f 2); python sts.py --stop {3} --format wisse --idfmodel local --tfidf {1} --localw {2} --embedmodel {4} --input {5} --comb {6} --dist {7} --output results_revision/$(echo {1}-{2}-{3}-{4/}-${in}-{6}-{7}.out | sed "s/--/-wst-/g")' ::: `cat tfidf.l` ::: `cat localw.l` ::: `cat stop.l` ::: `cat embedmodel.l` ::: `cat input.l` ::: `cat comb.l` ::: `cat dist.l`
# global output name
parallel 'idf=$(echo {1/}|cut -d'.' -f1|cut -d'_' -f 6-); in=$(echo {3}|cut -d'.' -f2); echo ${idf}-{2/}-${in}-{4}-{5}.out | sed 's/^-/freq-/g' ' ::: `cat idfmodel.l` ::: `cat embedmodel.l` ::: `cat input.l` ::: `cat comb.l` ::: `cat dist.l`
# global command
parallel 'in=$(echo {3}|cut -d'.' -f 2); idf=$(echo {1/}|cut -d'.' -f1|cut -d'_' -f 6-); python sts.py --format wisse --idfmodel {1} --embedmodel {2} --input {3} --comb {4} --dist {5} --output results_revision/global_${idf}-{2/}-${in}-{4}-{5}.out' ::: `cat idfmodel.l` ::: `cat embedmodel.l` ::: `cat input.l` ::: `cat comb.l` ::: `cat dist.l`

