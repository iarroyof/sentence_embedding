# System modules
import codecs
from sys import stderr as err
from Queue import Queue
from threading import Thread
import time
from sklearn.datasets import load_files
import numpy as np
# Local modules

# Set up some global variables
num_fetch_threads = 20
vec_file="sample.uniq.vec"
enclosure_queue = Queue()

def load_word_vectors(data_dir, word):
    #populations=load_obj(pop_file)
    #taken_words=[k for k in populations
    #                    if (k != "/*max_freq*/" and k != "<number>" and
    #                        k != "<NUMBER>" and int(populations[k]) > f_min and
    #                        (int(populations[k]) < f_max if f_max > 0 else True)
    #                       )
    #                ]
    # Return the queried word and its vectors as a list of numpy arrays
    return load_files(data_dir, categories=[word]).data

def average_embeddings(y, data_dir, q):
    """This is the worker thread function.
    It processes items in the queue one after
    another.  These daemon threads go into an
    infinite loop, and only exit when
    the main thread ends.
    """
    while True:
        try:
            word = q.get()
            #err.write('%s: Looking for the next enclosure\n' % y)
            word_vec_strs=load_word_vectors(data_dir, word)
            err.write('%s: Averaging... %s\n' % (y, word))
        # instead of really downloading the URL,
        # we just pretend and sleep
        #time.sleep(i + 2)
            word_matrix=[]
            for word_vec in word_vec_strs:
                word_matrix.append(np.asarray(word_vec.strip().split()).astype(float))

            word_matrix=np.vstack(word_matrix).mean(axis=0)
            with codecs.open(vec_file, 'a', 'latin-1') as f:
                #with open(filename, 'w') as f:
                f.write("%s\n" % " ".join(np.hstack([np.array([word]), word_matrix.astype('str')])))
        except Exception as ex:
            template = "\nAn exception of type {0} occurred. Arguments:\n{1!r}\n"
            message = template.format(type(ex).__name__, ex.args)
            err.write(message)
            exit()

        err.write('Written to file ::: %s\n' % word)
        q.task_done()

windows_dir="/almac/ignacio/data/windows/sample_dataset"
feed_words=load_files(windows_dir, load_content=False).target_names

# Set up some threads to fetch the enclosures
for i in range(num_fetch_threads):
    worker = Thread(target=average_embeddings, args=(i, windows_dir, enclosure_queue,))
    worker.setDaemon(True)
    worker.start()

# Download the feed(s) and put the enclosure URLs into
# the queue.
for word in feed_words:
    #response = feedparser.parse(url, agent='fetch_podcasts.py')
    #for entry in word.data:
        #for enclosure in entry.get('enclosures', []):
    err.write('Queuing: %s\n' % word)
    enclosure_queue.put(word)
        
# Now wait for the queue to be empty, indicating that we have
# processed all of the downloads.
err.write('\n*** Main thread waiting\n')
enclosure_queue.join()
err.write('*** Done')

