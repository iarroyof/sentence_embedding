# Author: Ignacio Arroyo-Fernandez (UNAM)

from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument, LabeledSentence
import os
from argparse import ArgumentParser as ap
import sys
from time import localtime, strftime

def clean_Ustring_fromU(string):
  if not string.strip().isspace():      
    from unicodedata import name, normalize
    gClean = ''
    accepted = ['SPACE', 'HYPHEN', 'LOW LINE']
    for ch in u''.join(string.decode('utf-8', 'ignore')):
        try:
            if name(ch).startswith('LATIN') or name(ch) in accepted:
                gClean = gClean + ch
            else: # Remove non-latin characters and change them by spaces
                gClean = gClean + ' '
        except ValueError: # In the case name of 'ch' does not exist in the unicode database.
            gClean = gClean + ' '
    
    try: # Trying different cases for bad input documents.
        normalized_string = normalize('NFKC', gClean.lower())
    except TypeError:
        #sys.stderr.write("\nBadly formed string at the first attempt\n-- Document sample: '"+gClean[0:49]+"'\n")
        try:
            range_error = 999
            normalized_string = normalize('NFKC', gClean[0:range_error].lower()) # One thousand of characters are written if available. 
        except TypeError:
            #sys.stderr.write('\nThe wrong string at the second attempt: before %s words\n' % range_error)
            try:
                range_error = 99
                normalized_string = normalize('NFKC', gClean[0:range_error].lower())
            except TypeError:
                #sys.stderr.write('\nThe wrong string at the third attempt: before %s words' % range_error)
                try:
                    range_error = 49
                    normalized_string = normalize('NFKC', gClean[0:range_error].lower())
                except TypeError:    
                 #   sys.stderr.write('\nIt was not possible forming output file after three attempts. Fatally bad file\n')
                    normalized_string = gClean.lower()
                    pass
    if not normalized_string.strip().isspace():
        return  normalized_string.split() # Return the unicode normalized document.
    return None
  else:
    return None

class yield_line_documents(object):
    def __init__(self, dirname, d2v=False, single=False, dirty=None):
        self.dirname = dirname
        self.d2v = d2v
        self.single = single
        self.dirty = dirty
        self.get_string = {True: str.split, False: clean_Ustring_fromU}
    def __iter__(self):
        if self.d2v:
            for fname in os.listdir(self.dirname):
                l = -1; pair = 0
                for line in open(os.path.join(self.dirname, fname)):
                    l += 1
                    #cs = clean_Ustring_fromU(line)
                    cs = self.get_string[self.dirty](line)
                    if not self.single:
                        if (l + 1) % 2: 
                            pair = pair + 1
                        tag = str(pair)+"_"+str(l)+"_snippet" # pair_sentence index tag
                    else:
                        tag = str(l)+"_snippet"                # sentence index tag                          
                    if cs:
                        yield LabeledSentence(cs, [tag])
                    else:
                        #sys.stderr.write("Empty string at line %s.\n" % l)
                        yield None
                    
        else:
            for fname in os.listdir(self.dirname):
                for line in open(os.path.join(self.dirname, fname)):
                    #yield clean_Ustring_fromU(line)
                    yield self.get_string[self.dirty](line)

if __name__ == "__main__":
    parser = ap(description='Trains and saves a word2vec model into a file for mmap\'ing. Tokenization is performed un utf-8 an for Python 2.7. Non-latin characters are replaced by spaces. The model is saved into a given directory. All options are needed.')    
    parser.add_argument('-i', type=str, dest = 'indir_file_name', help='Specifies the directory containing files to be processed. No sub-directories are allowed. In the case doc2vec is used, a file name must be specified. This file must contain a a sentence/document by line.')
    parser.add_argument('-o', type=str, dest = 'outfile', help='Specifies the file where to be stored the model.')
    parser.add_argument('-t', type=int, dest = 'threads', help='Specifies the number of threads the training will be divided.')
    parser.add_argument('-H', type=int, dest = 'hidden', help='Specifies the number of hidden units the model going to have.')
    parser.add_argument('-m', type=int, dest = 'minc', help='Specifies the minimum frequency a word should have in the corpus to be considered.')
    parser.add_argument('-d', default=False, action="store_true", dest = 'd2v', help='Toggles the doc2vec model, insted of the w2v one.')
    parser.add_argument('-s', default=False, action="store_true", dest = 'single', help='Toggles the pair or single tags.')
    parser.add_argument('-c', default=False, action="store_true", dest = 'update', help='Toggles if you want loading a pretrained model and continue training it with new input files.')
    parser.add_argument('-D', default=False, action="store_true", dest = 'dirty', help='Toggles if you do not want to process clean strings (i.e. raw file, including any symbol).')
    parser.add_argument('-w', type=int, dest = 'window', default=8, help='Specifies the number of words in the cooccurrence window.')
    args = parser.parse_args()

    if args.d2v:
        if not args.update:
            sys.stderr.write("\n>> [%s] Articles generator unpacking... %s\n" % (strftime("%Y-%m-%d %H:%M:%S", localtime()), args.outfile))
            arts = yield_line_documents(args.indir_file_name, d2v = True, single = args.single, dirty=args.dirty)
            sys.stderr.write("\n>> [%s] Articles generator unpacked... Training begins.\n" % (strftime("%Y-%m-%d %H:%M:%S", localtime())))

            try:
                d2v_model = Doc2Vec([a for a in arts if a], min_count = args.minc, workers = args.threads, size = args.hidden, window = int(args.window))    
                sys.stderr.write("\n>> [%s] Model successfully trained...\n" % (strftime("%Y-%m-%d %H:%M:%S", localtime())))
                d2v_model.save(args.outfile, separately = None)
                sys.stderr.write("\n>> [%s] Model successfully saved... %s\n" % (strftime("%Y-%m-%d %H:%M:%S", localtime()),args.outfile))
            except IOError:
                sys.stderr.write("\n>> [%s] Error caught while model saving...\n" % (strftime("%Y-%m-%d %H:%M:%S", localtime())))
                exit()
            #except:
                #sys.stderr.write("\n>> [%s] Error caught while model instantiation...\n" % (strftime("%Y-%m-%d %H:%M:%S", localtime())))
                #exit()
        else: ## If args.update:
            from os.path import exists
            from os import makedirs, getcwd, remove
            from time import sleep

            try:
                #trials = 10; to = 0
                #while(exists("%s/blocked" % getcwd()) and to < trials):
                #    if to > 0: sleep(1)
                #    to += 1
                #if not to < trials:
                #    sys.stderr.write("\n>> [%s] ERROR -- Unlearned: %s...\n" % (strftime("%Y-%m-%d %H:%M:%S", localtime()),args.indir_file_name))
                #    exit()
                
                #makedirs("%s/blocked" % getcwd())
                d2v_model = Doc2Vec.load(args.outfile)
                #try:
                #    remove("%s/blocked" % getcwd())
                #except:
                #    pass
                d2v_model.workers = args.threads
                sys.stderr.write("\n>> [%s] Model successfully loaded...\n" % (strftime("%Y-%m-%d %H:%M:%S", localtime())))
                sys.stderr.write("\n>> [%s] Articles generator unpacking... %s\n" % (strftime("%Y-%m-%d %H:%M:%S", localtime()), args.outfile))
                arts = yield_line_documents(args.indir_file_name, d2v = True, single = args.single, dirty=args.dirty)

                sys.stderr.write("\n>> [%s] Articles generator unpacked... Training begins.\n" % (strftime("%Y-%m-%d %H:%M:%S", localtime())))

                d2v_model.train([a for a in arts if a])
                sys.stderr.write("\n>> [%s] Model successfully trained...\n" % (strftime("%Y-%m-%d %H:%M:%S", localtime())))
                
                #to = 0
                #while(exists("%s/blocked" % getcwd()) and to < trials):
                 #   if to > 0: sleep(1)
                 #   to += 1
                #if not to < trials:
                 #   sys.stderr.write("\n>> [%s] ERROR -- Unlearned: %s...\n" % (strftime("%Y-%m-%d %H:%M:%S", localtime()),args.indir_file_name))
                 #   exit()
                
                #makedirs("%s/blocked" % getcwd())
                d2v_model.save(args.outfile, separately = None)
                #remove("%s/blocked" % getcwd())

                sys.stderr.write("\n>> [%s] Model successfully saved... %s\n" % (strftime("%Y-%m-%d %H:%M:%S", localtime()), args.outfile))
                sys.stderr.write("\n>> [%s] Model successfully saved...\n%s\n" % (strftime("%Y-%m-%d %H:%M:%S", localtime()), args.outfile))
            except IOError:
                sys.stderr.write("\n>> [%s] Error caught while model saving...\n" % (strftime("%Y-%m-%d %H:%M:%S", localtime())))
                exit()
            #except:
                #sys.stderr.write("\n>> [%s] Error caught while model instantiation...\n" % (strftime("%Y-%m-%d %H:%M:%S", localtime())))
                #exit()
        if not args.update:
            model = Doc2Vec.load(args.outfile)
            del(model)
        sys.stderr.write("\n>> [%s] Successful reload and Finished !! %s\n" % (strftime("%Y-%m-%d %H:%M:%S", localtime()), args.outfile))
    else:
        sys.stderr.write("\n>> [%s] Articles generator unpacking...\n" % (strftime("%Y-%m-%d %H:%M:%S", localtime())))        
        articles = yield_line_documents(args.indir_file_name)
        sys.stderr.write("\n>> [%s] Articles generator unpacked... Training begins.\n" % (strftime("%Y-%m-%d %H:%M:%S", localtime())))
        w2v_model = Word2Vec(articles, min_count = args.minc, workers = args.threads, size = args.hidden)
        sys.stderr.write("\n>> [%s] Model successfully trained...\n" % (strftime("%Y-%m-%d %H:%M:%S", localtime())))
        w2v_model.save(args.outfile, separately = None)
        sys.stderr.write("\n>> [%s] Model successfully saved...\n" % (strftime("%Y-%m-%d %H:%M:%S", localtime()), args.outfile))
        model = Word2Vec.load(args.outfile)
        del(model)
        sys.stderr.write("\n>> [%s] Finished !!\n" % (strftime("%Y-%m-%d %H:%M:%S", localtime())))
    
        
