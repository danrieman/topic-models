from re import sub
import time
from sqlalchemy import create_engine
from dlatk.lib.happierfuntokenizing import Tokenizer
from langid import langid
from .configs import (
    MYSQL_HOST,
    MYSQL_PASS,
    MYSQL_PORT,
    MYSQL_USER,
)

# Data location
DB = 'empathy_qualtrics'
MSGS = 'statuses_fin'
LABS = 'survey_combined_fin'

# Parse settings
MIN_TOK_AFTER_STOPWORD = 5

# Connect to DB
db_eng = create_engine('mysql://{0}:{1}@{2}:{3}/{4}'.format(
    MYSQL_USER, MYSQL_PASS, MYSQL_HOST, MYSQL_PORT, DB))


# Get outcomes
sql = sub('[\t\n ]+', ' ', """
    SELECT DISTINCT user_id, gender 
    FROM {}.{} 
    WHERE gender IN ('1','2') AND 
        num_complete_surveys_rid=1 AND 
        age_fixed IS NOT NULL 
    """.format(DB, LABS))

labs = dict()
for i in db_eng.execute(sql):
    labs[i[0]] = {'gender':i[1]}

# Label summary
lab_summ = {1:0, 2:0}
for i in labs.values():
    lab_summ[i['gender']] += 1

print(lab_summ[2], 'female')
print(lab_summ[1], 'male')


# Load mallet stopwords
stoplist = list()
with open('/home/rieman/mallet/mallet-2.0.8RC3/stoplists/en.txt', 'r') as doc:
    for word in doc:
        stoplist.append(sub('\n', '', word))
    doc.close()

stoplist_set = set(stoplist)

# Create tokenizer
tokenizer = Tokenizer(preserve_case=False, use_unicode=True)


# Get messages, classify language, tokenize, remove stopwords, and join to label
msgs = list()
n_msgs = [i[0] for i in db_eng.execute('SELECT COUNT(*) FROM {}.{}'.format(DB, MSGS))][0]
sql = 'SELECT user_id, message_id, message FROM {}.{}'.format(DB, MSGS)
k = 0
t0 = time.time()
for i in db_eng.execute(sql):
    k += 1
    if k % 10000 == 0:
        print('{} of {}\t{} min'.format(k, n_msgs, round((time.time() - t0)/60., 2)))
    if langid.classify(i[2])[0] == 'en':
        try:
            tmp = [tok for tok in tokenizer.tokenize(i[2]) if tok not in stoplist_set]
            if len(tmp) >= MIN_TOK_AFTER_STOPWORD:
                msgs.append((i[0], i[1], labs[i[0]]['gender'], 
                    len(tmp), ' '.join(tmp)))
                try:
                    labs[i[0]]['n'] += 1
                except KeyError:
                    labs[i[0]]['n'] = 1
        except KeyError:
            pass


print('Total time: {} min'.format(round(time.time() - t0)/60., 2))

# 2072125 messages without langid





# Tokenize messages
bag_of_words_list = list()
sql = 'SELECT user_id, message_id, message FROM {}.{}'.format(DB, MSGS)
for i in db_eng.execute(sql):
    tmp = dict()
    for j in tokenizer.tokenize(i[2]):
        try:
            tmp[j] += 1
        except KeyError:
            tmp[j] = 1
    bag_of_word_list.append((i[0], i[1], tmp))
