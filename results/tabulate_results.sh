pair=$1 # File of pairs dataset
gs=$2   # File of labels associated to pairs
c=$3
r=$4

printf "Embedding Size Transform Combine TFIDF Cosine Euclidean Manhattan\n" > all_results
export LC_NUMERIC="en_US.UTF-8"
paste <(for r in `ls "$pair".output*`; do printf "%s\\t%s\\t%f\\n" "${r##*output_}" $(perl $NLP/correlation-noconfidence.pl "$gs" \
      <(awk -F '\\t' '{print match($1, /[^ ]/) ? $1 : "0.1"}' "$r")); done) \
      <(for r in `ls "$pair".output*`; do printf "%s\\t%s\\t%f\\n" "${r##*output_}" $(perl $NLP/correlation-noconfidence.pl "$gs" \
      <(awk -F '\\t' '{print match($2, /[^ ]/) ? $2 : "0.1"}' "$r")); done) \
      <(for r in `ls "$pair".output*`; do printf "%s\\t%s\\t%f\\n" "${r##*output_}" $(perl $NLP/correlation-noconfidence.pl "$gs" \
      <(awk -F '\\t' '{print match($3, /[^ ]/) ? $3 : "0.1"}' "$r")); done) | awk '{ print $1, $3, $6, $9 }' >> all_results

#read -p "Press key..."

sed -i -- 's/ -0./ 0./g' all_results
awk '{for(i=NF;i>=1;i--) printf "%s ", $i;print ""}' all_results > all_resultss
if [ ! -z "$c" ]; then
    if [ ! -z "$r" ]; then
        cat all_resultss | perl -pe 's/_/ /g'| sort -nrk "$c" | awk '{for(i=NF;i>=1;i--) printf "%s ", $i;print ""}'
    #cat all_results | sort -nrk "$c"
    else
        cat all_resultss | perl -pe 's/_/ /g'| sort -nk "$c" | awk '{for(i=NF;i>=1;i--) printf "%s ", $i;print ""}'
        #cat all_results | sort -nk "$c"
    fi
else
    cat all_results| perl -pe 's/_/ /g'| head
fi
