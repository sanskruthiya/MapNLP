import pandas as pd
import os
import sys
import codecs
import csv

file_name = input('Input csv file >> ')
if os.path.exists(file_name):
    with codecs.open(file_name, 'r', 'utf-8', 'ignore') as f:
        df = pd.read_csv(f)
    base_name = os.path.splitext(os.path.basename(file_name))[0] #file name without extension
else:
    print('Processing was cancelled due to an invalid input. Please enter a correct file name')
    sys.exit()

print('Please review the data information')
print(df.info())

next_input = input('Creating a summary data? Press 1 to cancell, Enter-key or any to continue. >> ')
if next_input == "1":
    print('Program cancelled.')
    sys.exit()
else:
    pass

df['title'] = df['研究課題名'].astype(str)
df['doc_id'] = df['研究課題/領域番号'].astype(str)
df['project_period'] = df['研究期間 (年度)'].astype(str)
df['research_keywords'] = df['キーワード'].astype(str)
df['main_researcher'] = df['研究代表者'].astype(str)
df['co_researcher'] = df['研究分担者'].astype(str)
df['member_01'] = df['連携研究者'].astype(str)
df['member_02'] = df['研究協力者'].astype(str)
df['member_03'] = df['特別研究員'].astype(str)
df['member_04'] = df['外国人特別研究員'].astype(str)
df['member_05'] = df['受入研究者'].astype(str)
df['text_01'] = df['研究開始時の研究の概要'].astype(str)
df['text_02'] = df['研究概要'].astype(str)
df['text_03'] = df['研究成果の概要'].astype(str)
df['text_04'] = df['研究実績の概要'].astype(str)
df['text_05'] = df['今後の研究の推進方策'].astype(str)

#combine target columns
df['description'] = df['text_01'].str.cat(df.loc[:,['text_02', 'text_03', 'text_04', 'text_05']], sep=' / ', na_rep=' - ')
df['other_members'] = df['member_01'].str.cat(df.loc[:,['member_02', 'member_03', 'member_04', 'member_05']], sep=' / ', na_rep=' - ')

df2 = pd.DataFrame()
df2['num_id'] = pd.RangeIndex(start=1, stop=len(df) + 1, step=1)
df2['doc_id'] = df['doc_id']
df2['title'] = df['title'].replace('[\r\n\t]', '', regex=True)
df2['description'] = df['description'].replace('(nan / )|(^nan$)|(/ nan)|(nan$)|[\r\n\t]', '', regex=True) #nanがString型で認識されるため置換削除する
df2['research_keywords'] = df['research_keywords'].replace('(nan / )|(^nan$)|(/ nan)|(nan$)|[\r\n\t]', '', regex=True)
df2['project_period'] = df['project_period'].replace('[\r\n\t]', '', regex=True)
df2['start_year'] = df['project_period'].str[:4]
df2['main_researcher'] = df['main_researcher'].replace('(nan / )|(^nan$)|(/ nan)|(nan$)|[\r\n\t]', '', regex=True)
df2['co_researcher'] = df['co_researcher'].replace('(nan / )|(^nan$)|(/ nan)|(nan$)|[\r\n\t]', '', regex=True)
df2['other_members'] = df['other_members'].replace('(nan / )|(^nan$)|(/ nan)|(nan$)|[\r\n\t]', '', regex=True)
df2['affiliation'] = df['研究機関'].replace('[\r\n\t]', '', regex=True)
df2['review_section'] = df['審査区分'].replace('[\r\n\t]', '', regex=True)
df2['research_section'] = df['研究分野'].replace('[\r\n\t]', '', regex=True)
df2['research_category'] = df['研究種目'].replace('[\r\n\t]', '', regex=True)
df2['total_grant'] = df['総配分額'].replace('[\r\n\t]', '', regex=True)
df2['rating'] = df['評価記号'].replace('[\r\n\t]', '', regex=True)

df2.to_csv(base_name + "_summary.csv", encoding="utf-8", index=False, quotechar='"', quoting=csv.QUOTE_ALL)

print("Done. Please see: " + base_name + "_summary.csv")
