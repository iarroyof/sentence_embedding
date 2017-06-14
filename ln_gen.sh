DATA=$1
paste -d' ' <(for d in `cat embeddings.list`; do ls $DATA"${d##*data}"; done) <(cat links) | while read first second; do ln -s $first $second; done
