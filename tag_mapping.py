import json
import pandas as pd

if __name__ == '__main__':

    with open('train.json', encoding='utf-8') as f:
        train = json.load(f)

    with open('song_meta.json', encoding='utf-8') as f:
        song_meta = json.load(f)

    check_list = []
    cnt = 1
    length = len(train)
    for t in train:
        print('*'*int((cnt/length)*10))
        tags = t['tags']
        song_list = t['songs']
        check_list += song_list
        for s in song_list:
            if 'tags' not in song_meta[s]:
                song_meta[s]['tags'] = tags
            else:
                song_meta[s]['tags'] += tags

    cnt = 1
    length = len(song_meta)
    for sm in song_meta:
        print('*'*int((cnt/length)*10))
        if 'tags' in sm:
            sm['tags'] = list(pd.Series(sm['tags']).value_counts(ascending=False)[:20].index)
            print(sm['tags'])

    print('start file writing...')
    with open('song_meta_with_tags.json', 'w', encoding='utf-8') as f:
        json.dump(song_meta, f, ensure_ascii=False, indent='\t')
