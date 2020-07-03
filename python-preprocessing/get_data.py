from re import sub
import sys

sys.path.insert(0, '/home/rieman/DLATK_23/dlatk')

from dlatk.mysqlMethods.mysql_iter_funcs import get_db_engine

# Set parameters
DB = 'empathy_qualtrics'
FEAT_FB = 'feat$1gram$statuses_fin$user_id$16to16'
FEAT_TWT = 'feat$1gram$statuses_repullT_en$user_id$16to16'
OUTCOME_TBL = ''
FOLDS = 1
TEMP_TABLE = 'TMP_TBL'

# Connect to MySQL
db_eng = get_db_engine(DB, host='127.0.0.1')


# Format data in MySQL
sql = sub('[\n\t ]+', ' ', 
    '''CREATE TABLE {db}.{tmp} (
        group_id VARCHAR(250), 
        feat VARCHAR(36), 
        fb_value INT(10) DEFAULT 0, 
        twt_value INT(10) DEFAULT 0, 
        KEY (group_id), 
        KEY (feat)) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin;'''.format(db=DB, tmp=TEMP_TABLE))
db_eng.execute(sql)

sql = sub('[\n\t ]+', ' ', 
    '''INSERT INTO {db}.{tmp} (group_id, feat, fb_value, twt_value) 
        SELECT group_id, feat, value AS fb_value, 0 AS twt_value FROM {db}.{fb} UNION ALL 
        SELECT group_id, feat, 0 AS fb_value, value AS twt_value FROM {db}.{twt}
    '''.format(db=DB, tmp=TEMP_TABLE, fb=FEAT_FB, twt=FEAT_TWT))
db_eng.execute(sql)


# Get Data
sql = sub('[\n\t ]+', ' ', 
    '''SELECT group_id, feat, SUM(fb_value) AS fb_value, SUM(twt_value) AS twt_value 
        FROM {db}.{tmp} 
        GROUP BY group_id, feat
    '''.format(db=DB, tmp=TEMP_TABLE))
df = pd.read_sql(con=db_eng, sql=sql)



