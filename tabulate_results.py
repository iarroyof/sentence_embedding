import os
import subprocess
import sys
from pdb import set_trace as st

results = "results_revision/"
DATA = "/almac/ignacio/data"
out = "out_tab.csv"

with open(out, "w") as of:
    for j in os.listdir(results):
        lg = j.split("_")[0]    
        if lg == "local":
            f = j.split("-")[4] #$(echo ${j} | cut -d'-' -f5)
        elif lg == "global":
            f = j.split("-")[2]

        if f == "dev":
            gs = DATA + "/sts/benchmark_2017/sts-gs.dev.txt"
        elif f == "train":
            gs = DATA + "/sts/benchmark_2017/sts-gs.train.txt"
        elif f == "test":
            gs = DATA + "/sts/benchmark_2017/sts-gs.test.txt"
        elif f == "input":
            gs = DATA + "/sts/sick/sts.gs.sick-trial.txt"

    #perl_script = subprocess.check_output(["perl", 
    #                                        "correlation-noconfidence.pl", gs, 
    #                                        "/almac/ignacio/sentence_embedding/" + results + j])
        command="""gs={0}; sys={1}; if [[ $(wc -l $gs | cut -d' ' -f1) = $(wc -l $sys | cut -d' ' -f1)  ]];then p=$(perl correlation-noconfidence.pl $gs $sys | cut -d' ' -f2); printf "%s %s\\n" $sys ${{p//-}}; fi\n""" 
        of.write (command.format(gs, results + j ))
#    with open(out, "w") as of:
#        if isinstance(perl_script, bytes):
#            try:
#                of.write ("%s\t%s" % (j, perl_script.decode('utf-8').split[1].strip()))
#            except TypeError:
#                pass
#        else:
#            st()

    
                                                
