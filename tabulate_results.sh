for j in `ls results_revision/`; do
    lg=$(echo ${j} | cut -d'_' -f2 | cut -d'/' -f2)

    if [ ${lg} = "local" ]; then
        f=$(echo ${j} | cut -d'-' -f5)
    else if [ ${lg} = "global" ]; then
        f=$(echo ${j} | cut -d'-' -f3)
    fi
#    if [ ${f} = "dev"  ]; then
#        gs=$DATA/sts/benchmark_2017/sts-gs.dev.txt
#    else if [ ${f} = "train"  ]; then
#        gs=$DATA/sts/benchmark_2017/sts-gs.train.txt
#    else if [ ${f} = "test"  ]; then
#        gs=$DATA/sts/benchmark_2017/sts-gs.test.txt
#    else if [ ${f} = "input"  ]; then
        gs=$DATA/sts/sick/sts.gs.sick-trial.txt
#    fi
    #perl correlation-noconfidence.pl ${gs} ${j};
    echo ${gs} ${j}
done
