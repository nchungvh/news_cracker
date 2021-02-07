
import pandas as pd
from torchtext import data

class DataFrameDataset(data.Dataset):
    def __init__(self, df, fields, is_test=False, **kwargs):
        examples = []
        for i, row in df.iterrows():
            label = row.label if not is_test else None
            text = row.text
            weight = row.weight
            examples.append(data.Example.fromlist([text, label, weight], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, fields, train_df, val_df=None, test_df=None, **kwargs):
        train_data, val_data, test_data = (None, None, None)
        data_field = fields

        if train_df is not None:
            train_data = cls(train_df.copy(), data_field, **kwargs)
        if val_df is not None:
            val_data = cls(val_df.copy(), data_field, **kwargs)
        if test_df is not None:
            test_data = cls(test_df.copy(), data_field, True, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)

def get_df_articles(_from,_to,_lang):

    query = {
        "query": {
          "bool": {
            "must": [
              {
                "range": {
                  "publish_datetime": {
                    "gte": _from,
                    "lt": _to
                  }
                }
              },
              {
                "match": {
                    "lang": _lang
                }
              }
            ]
          },
        }
    }

    res = helpers.scan(
                    client = es,
                    scroll = '2m',
                    query = query,
                    index = "articles")

    articles = []
    for a in res:
        art = a["_source"]
        h   = art['handle']
        url = art['url']
        tit = art['title']
        txt = art['text']
        lan = art['lang']

        fav = art['favorite_count'] if 'favorite_count' in art else 0
        rep = art['reply_count'] if 'reply_count' in art else 0
        quo = art['quote_count'] if 'quote_count' in art else 0
        ret = art['retweet_count'] if 'retweet_count' in art else 0

        articles.append((h,lan,tit,txt,fav,rep,quo,ret))

    cols = ['handle','language','title','text','favorite','reply','quote','retweet']
    return pd.DataFrame(articles, columns=cols)