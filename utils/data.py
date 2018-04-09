import numpy as np
import sys
import ibis
import pandas


def padding(response_df: pandas.DataFrame, user_count=2, item_count=2):
    if 'user_id' not in response_df.columns or 'item_id' not in response_df.columns or 'answer' not in response_df.columns:
        raise ValueError("input dataframe have no user_id or item_id  or answer")
    print("=" * 20 + "padding" + "=" * 20, file=sys.stderr)
    _response_df = response_df[['user_id', 'item_id', 'answer']]
    _user_ids = _response_df['user_id'].unique()
    _user_count = len(_user_ids)
    _item_ids = _response_df['item_id'].unique()
    _item_count = len(_item_ids)
    # print("before", file=sys.stderr)

    print(
        '[before] user:%d item:%d record:%d' % (_user_count, _item_count, len(_response_df['user_id'])),
        file=sys.stderr)
    _padding = {'user_id': [], 'item_id': [], 'answer': []}
    # padding user
    for i in range(user_count):
        _padding['user_id'].extend(['_u_%d' % i] * _item_count)
        _padding['answer'].extend([0] * _item_count)
        _padding['item_id'].extend(list(_item_ids))

    # padding item
    for i in range(item_count):
        _padding['item_id'].extend(['_i_%d' % i] * _user_count)
        _padding['answer'].extend([0] * _user_count)
        _padding['user_id'].extend(list(_user_ids))

    assert len(_padding['user_id']) == len(_padding['item_id']) == len(_padding['answer'])
    _response_df = pd.concat([_response_df, pandas.DataFrame(_padding)]).sample(frac=1)

    print('[padding] user:%d item:%d record:%d' % (user_count, item_count, len(_padding['answer'])),
          file=sys.stderr)
    print('[after] user:%d item:%d record:%d' % (len(_response_df['user_id'].unique()),
                                                 len(_response_df['item_id'].unique()),
                                                 len(_response_df)), file=sys.stderr)
    print("=" * 20 + "padding end" + "=" * 20, file=sys.stderr)
    return _response_df


def split_data(df):
    """
    把数据分成训练集、测试集两部分,每个人3条作答记录
    要保证每个题有足够多的人答，每个人有足够多的题
    :param df:
    :return:
    """
    item_count = len(df['item_id'].unique())
    user_count = len(df['user_id'].unique())
    user_stats = {index: row['item_id'] for index, row in df.groupby(['user_id']).count().iterrows()}
    item_stats = {index: row['user_id'] for index, row in df.groupby(['item_id']).count().iterrows()}

    test_selected = []
    user_flag = {}
    item_flag = {}
    for index, row in df.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        if user_stats[user_id] > item_count * 0.5 \
                and item_stats[item_id] >= user_count * 0.5:
            # and user_flag.get(user_id, 0) <= 5 \
            # and item_flag.get(item_id, 0) <= 5:
            test_selected.append(True)
            user_stats[user_id] -= 1
            item_stats[item_id] -= 1
            # 记录学生、题目被选中的次数
            user_flag[user_id] = user_flag.get(user_id, 0) + 1
            item_flag[user_id] = item_flag.get(item_id, 0) + 1
        else:
            test_selected.append(False)

    return df[[not x for x in test_selected]], df[test_selected]


sys.path.append("./")


def load_logs(cache_file="logs.pickle", from_cache=True):
    _sql = """
            select
                sa.sa_stu_id as user_id,
                lq.lq_origin_id as item_id,
                sa.sa_answer_status as answer,
                lq.lq_qst_difct as difficult

            from odata.ods_ips_tb_stu_answer sa
            join odata.ods_ips_tb_level_question lq on lq.lq_id=sa.sa_lq_id
            where
                sa.sa_year="2017"
                and sa.sa_city_code="028"
                and sa.sa_term_id='3'
                and sa.sa_subj_id='ff80808127d77caa0127d7e10f1c00c4'
                and sa.sa_lev_id='ff80808145707302014582f9d9dc3658'
                and sa.sa_grd_id="7"
                and lq.lq_library_id='5'
                and sa.sa_is_fixup=0
                and sa.sa_answer_status in (1,2)
        """

    if from_cache:
        # print >> sys.stderr, "从缓存读取题目画像数据"
        print("从缓存读取题目画像数据", file=sys.stderr)
        return pandas.read_pickle(cache_file)
    # print("从impala读取题目画像数据", file=sys.stderr)
    print("从impala读取题目画像数据", file=sys.stderr)
    # 默认情况下会限制只返回10000条数据
    ibis.options.sql.default_limit = None
    impala_client = ibis.impala.connect(host='192.168.23.236', port=21050, user='app_bi')
    df_question = impala_client.sql(_sql).execute()
    df_question.to_pickle(cache_file)
    impala_client.close()
    print("count:", len(df_question), file=sys.stderr)
    return df_question


if __name__ == "__main__":
    df = load_logs(from_cache=True)
    train_df, test_df = split_data(df)
    print(len(df), len(train_df), len(test_df))
