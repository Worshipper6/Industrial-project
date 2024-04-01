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


# Extract top1000 bank transactions
df = pd.read_csv(inPath)
common_ids = list(set(df['From Bank']).intersection(df['To Bank']))
bank_id = 0
for i in common_ids:
    par_path = os.path.dirname(inPath) + "/bank/bank" + str(bank_id)
    by_bank = df[(df['From Bank']==i) | (df['To Bank']==i)]
    # Convert to datatable
    #raw = dt.fread(inPath, columns = dt.str32)
    raw = dt.Frame(by_bank)

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
    Payment Format,Is Laundering\n"

    firstTs = -1

    with open(par_path, 'w') as writer:
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

            fromAccIdStr = str(raw[i,"From Bank"]) + raw[i,2]
            fromId = get_dict_val(fromAccIdStr, account)

            toAccIdStr = str(raw[i,"To Bank"]) + raw[i,4]
            toId = get_dict_val(toAccIdStr, account)

            amountReceivedOrig = float(raw[i,"Amount Received"])
            amountPaidOrig = float(raw[i,"Amount Paid"])

            isl = int(raw[i,"Is Laundering"])

            line = '%d,%d,%d,%d,%f,%d,%f,%d,%d,%d\n' % \
                        (i,fromId,toId,ts,amountPaidOrig,cur2, amountReceivedOrig,cur1,fmt,isl)

            writer.write(line)

    formatted = dt.fread(par_path)
    formatted = formatted[:,:,sort(3)]

    formatted.to_csv(par_path)
    bank_id += 1