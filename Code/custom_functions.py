def process_dfs(train_df, test_df):
    print("Processing dfs...")
    print("Dropping repeated columns...")
    columns = [col for col in train_df.columns if train_df[col].nunique() > 1]

    train_df = train_df[columns]
    test_df = test_df[columns]

    trn_len = train_df.shape[0]

    merged_df = pd.concat([train_df, test_df])

    merged_df['diff_visitId_time'] = merged_df['visitId'] - merged_df['visitStartTime']
    merged_df['diff_visitId_time'] = (merged_df['diff_visitId_time'] != 0).astype(int)
    del merged_df['visitId']

    print("Generating date columns...")
    format_str = '%Y%m%d'
    merged_df['formated_date'] = merged_df['date'].apply(lambda x: datetime.strptime(str(x), format_str))
    merged_df['WoY'] = merged_df['formated_date'].apply(lambda x: x.isocalendar()[1])
    merged_df['month'] = merged_df['formated_date'].apply(lambda x:x.month)
    merged_df['quarter_month'] = merged_df['formated_date'].apply(lambda x:x.day//8)
    merged_df['weekday'] = merged_df['formated_date'].apply(lambda x:x.weekday())

    del merged_df['date']
    del merged_df['formated_date']

    merged_df['formated_visitStartTime'] = merged_df['visitStartTime'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
    merged_df['formated_visitStartTime'] = pd.to_datetime(merged_df['formated_visitStartTime'])
    merged_df['visit_hour'] = merged_df['formated_visitStartTime'].apply(lambda x: x.hour)

    del merged_df['visitStartTime']
    del merged_df['formated_visitStartTime']

    print("Encoding columns with pd.factorize()")
    for col in merged_df.columns:
        if col in ['fullVisitorId', 'month', 'quarter_month', 'weekday', 'visit_hour', 'WoY']: continue
        if merged_df[col].dtypes == object or merged_df[col].dtypes == bool: merged_df[col], indexer = pd.factorize(merged_df[col])

    print("Splitting back...")
    train_df = merged_df[:trn_len]
    test_df = merged_df[trn_len:]
    print("Done!")

    return train_df, test_df

def preprocess():
    train_df = pd.read_csv('train-flattened.csv', dtype = {'fullVisitorId' : np.str})
    test_df = pd.read_csv('test-flattened.csv', dtype = {'fullVisitorId' : np.str})

    target = train_df['totals.transactionRevenue'].fillna(0).astype(float)
    target = target.apply(lambda x: np.log1p(x))

    del train_df['totals.transactionRevenue']

    train_df, test_df = process_dfs(train_df, test_df)
    train_df.to_csv('train-flat-clean.csv', index=False)
    test_df.to_csv('test-flat-clean.csv', index=False)
    target.to_csv('target.csv', index=False)
