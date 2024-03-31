import numpy as np
import datatable as dt
from datetime import datetime
from datatable import f,join,sort
import pandas as pd
import sys
import os

n = len(sys.argv)

if n == 1:
    print("No input path")
    sys.exit()

inPath = sys.argv[1]
outPath = os.path.dirname(inPath) + "/formatted_transactions.csv"

# Extract top1000 bank transactions
df = pd.read_csv(inPath)
grouped_from_bank = df.groupby('From Bank').size().reset_index(name='Total_Count_From')
grouped_to_bank = df.groupby('To Bank').size().reset_index(name='Total_Count_To')
top_1000_from_banks = grouped_from_bank.sort_values('Total_Count_From', ascending=False).head(1000)
top_1000_to_banks = grouped_to_bank.sort_values('Total_Count_To', ascending=False).head(1000)
df = df[df['From Bank'].isin(top_1000_from_banks['From Bank']) & df['To Bank'].isin(
    top_1000_to_banks['To Bank'])]
df.to_csv(inPath, index=False)

raw = dt.fread(inPath, columns = dt.str32)

currency = dict()
paymentFormat = dict()
bankAcc = dict()
account = dict()

def get_dict_val(name, collection):
    if name in collection:
        val = collection[name]
    else:
        val = len(collection)
        collection[name] = val
    return val

header = "EdgeID,from_id,to_id,Timestamp,\
Amount Sent,Sent Currency,Amount Received,Received Currency,\
Payment Format,Is Laundering, From_bank, To_bank\n"

firstTs = -1

with open(outPath, 'w') as writer:
    writer.write(header)
    for i in range(raw.nrows):
        datetime_object = datetime.strptime(raw[i,"Timestamp"], '%Y/%m/%d %H:%M')
        ts = datetime_object.timestamp()
        day = datetime_object.day
        month = datetime_object.month
        year = datetime_object.year
        hour = datetime_object.hour
        minute = datetime_object.minute

        if firstTs == -1:
            startTime = datetime(year, month, day)
            firstTs = startTime.timestamp() - 10

        ts = ts - firstTs

        cur1 = get_dict_val(raw[i,"Receiving Currency"], currency)
        cur2 = get_dict_val(raw[i,"Payment Currency"], currency)

        fmt = get_dict_val(raw[i,"Payment Format"], paymentFormat)

        fromAccIdStr = raw[i,"From Bank"] + raw[i,2]
        fromId = get_dict_val(fromAccIdStr, account)

        toAccIdStr = raw[i,"To Bank"] + raw[i,4]
        toId = get_dict_val(toAccIdStr, account)

        amountReceivedOrig = float(raw[i,"Amount Received"])
        amountPaidOrig = float(raw[i,"Amount Paid"])

        isl = int(raw[i,"Is Laundering"])
        From_bank = raw[i, "From Bank"]
        To_bank = raw[i, "To Bank"]

        line = '%d,%d,%d,%d,%f,%d,%f,%d,%d,%d, %s, %s\n' % \
                    (i,fromId,toId,ts,amountPaidOrig,cur2, amountReceivedOrig,cur1,fmt,isl, \
                     From_bank, To_bank)

        writer.write(line)

formatted = dt.fread(outPath)
formatted = formatted[:,:,sort(3)]

formatted.to_csv(outPath)