import json
import pandas as pd


def predict_tags(v_tags, song_list, meta):
    t = []
    for s in song_list:
        if 'tags' in meta[s]:
            t += meta[s]['tags']
    tags = list(pd.Series(t).value_counts(ascending=False).index)
    for rm in v_tags:
        if rm in tags:
            tags.remove(rm)
    return tags[:10]


def predict_songs(song_list):
    df = pd.read_pickle('cos_sim.pkl')
    group_list = []
    for s in song_list:
        group_list.append(df.loc[s, 'group_idx'])
    group_list = list(pd.Series(t).value_counts(ascending=True).index)
    result = []
    while len(result) < 100:
        g_idx = group_list.pop()
        result += list(df[df['group_idx'] == g_idx].index)
        if len(group_list) == 0:
            break
    return result[:100]


if __name__ == '__main__':
    print('load song meta data')
    with open('song_meta_with_tags.json', 'r', encoding='utf-8') as f:
        song_meta = json.load(f)

    print('load target data')
    with open('val.json', 'r', encoding='utf-8') as f:
        val = json.load(f)

    result = []
    pred = {}
    for v in val[:5]:
        pred['id'] = v['id']
        pred['songs'] = predict_songs(song_list=v['songs'])
        pred['tags'] = predict_tags(v_tags=v['tags'], song_list=v['songs'], meta=song_meta)
        result.append(pred)

    print('result file writing...')
    with open('result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent='\t')
