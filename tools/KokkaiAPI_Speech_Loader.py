import requests
import time
import sys
import csv
import re

base_url = "https://kokkai.ndl.go.jp/api/speech"

#URLパラメータ用の辞書を空の状態で用意し、後から順次格納する。01はヒット総数の確認用、02はデータ取得用。
params_01 = {} 
params_02 = {}

#パラメータを対話的に入力する
text_input = input('検索する文字列を入力（Enterキーでスキップ） >> ')
if text_input != "":
    params_01['any'] = str(text_input)
    params_02['any'] = str(text_input)
else:
    pass

house_input = input('検索する院名を入力（Enterキーでスキップ） >> ')
if house_input != "":
    params_01['nameOfHouse'] = str(house_input)
    params_02['nameOfHouse'] = str(house_input)
else:
    pass

speaker_input = input('検索する発言者名を入力（Enterキーでスキップ） >> ')
if speaker_input != "":
    params_01['speaker'] = str(speaker_input)
    params_02['speaker'] = str(speaker_input)
else:
    pass

from_input = input('開始日を入力 (e.g. 2020-09-01) >> ')
if re.match(r'[0-9]{4}-[0-1][0-9]-[0-3][0-9]', from_input): #正規表現によるパターンマッチングにて入力値が有効か判定
    params_01['from'] = str(from_input)
    params_02['from'] = str(from_input)
else:
    params_01['from'] = "2020-09-01"
    params_02['from'] = "2020-09-01"
    print("'From' date is set to 2020-09-01 due to invalid input")

until_input = input('終了日を入力 (e.g. 2020-11-30) >> ')
if re.match(r'[0-9]{4}-[0-1][0-9]-[0-3][0-9]', until_input): #正規表現によるパターンマッチングにて入力値が有効か判定
    params_01['until'] = str(until_input)
    params_02['until'] = str(until_input)
else:
    params_01['until'] = "2020-09-30"
    params_02['until'] = "2020-09-30"
    print("'Until' date is set to 2020-09-30 due to invalid input")

#ヒット件数確認用のパラメータを設定
params_01['maximumRecords'] = 1
params_01['recordPacking'] = "json"

response_01 = requests.get(base_url, params_01) #URLをエンコードしてAPIへリクエスト
jsonData_01 = response_01.json() #APIからのレスポンスをJSON形式で取得

#レスポンスに含まれているヒット件数を確認（レスポンスのJSONにレコード数の項目がない場合はクエリに問題ありと判断しエラー終了）
try:
    total_num = jsonData_01["numberOfRecords"]
except:
    print("クエリエラーにより取得できませんでした。")
    sys.exit()

#件数を表示し、データ取得を続行するか確認
next_input = input("検索結果は " + str(total_num) + "件です。\nキャンセルする場合は 1 を、データを取得するにはEnterキーまたはその他を押してください。 >> ")
if next_input == "1":
    print('プログラムをキャンセルしました。')
    sys.exit()
else:
    pass

max_return = 100 #発言内容は一回のリクエストにつき100件まで取得可能なため、その上限値を取得件数として設定
pages = (int(total_num) // int(max_return)) + 1 #ヒットした全件を取得するために何回リクエストを繰り返すか算定

#全件取得用のパラメータを設定
params_02['maximumRecords'] = max_return
params_02['recordPacking'] = "json"

Records = [] #取得データを格納するための空リストを用意

#全件取得するためのループ処理
i = 0
while i < pages:
    i_startRecord = 1 + (i * int(max_return))
    params_02['startRecord'] = i_startRecord
    response_02 = requests.get(base_url, params_02)
    jsonData_02 = response_02.json()
    #JSONデータ内の各発言データから必要項目を指定してリストに格納する
    for list in jsonData_02['speechRecord']:
        list_sid = list['speechID']
        list_mid = list['issueID']
        list_kind = list['imageKind']
        list_house = list['nameOfHouse']
        list_topic = list['nameOfMeeting']
        list_issue = list['issue']
        list_date = list['date']
        list_order = list['speechOrder']
        list_speaker = list['speaker']
        list_group = list['speakerGroup']
        list_position = list['speakerPosition']
        list_role = list['speakerRole']
        list_speech = list['speech'].replace('\r\n', ' ').replace('\n', ' ') #発言内容の文中には改行コードが含まれるため、これを半角スペースに置換
        list_url01 = list['speechURL']
        list_url02 = list['meetingURL']
        Records.append([list_sid, list_mid, list_kind, list_house, list_topic, list_issue, list_date, list_order, list_speaker, list_group, list_position, list_role, list_speech, list_url01, list_url02])

    sys.stdout.write("\r%d/%d is done." % (i+1, pages)) #進捗状況を表示する
    i += 1
    time.sleep(0.5) #リクエスト１回ごとに若干時間をあけてAPI側への負荷を軽減する

#CSVへの書き出し
with open("kokkai_speech_" + str(total_num) + ".csv", 'w', newline='') as f:
    csvwriter = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC) #CSVの書き出し方式を適宜指定
    csvwriter.writerow(['発言ID', '会議録ID', '種別', '院名', '会議名', '号数', '日付', '発言番号', '発言者名', '発言者所属会派', '発言者肩書き', '発言者役割', '発言内容', '発言URL', '会議録URL'])
    for record in Records:
        csvwriter.writerow(record)
