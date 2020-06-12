import arena_util
import numpy as np
from collections import defaultdict
import pandas as pd
import tqdm
import fire


class CfModel:
    def run(self):
        print('load song_meta')
        song_meta = arena_util.load_json('./arena_data/meta/song_meta_with_tags.json')
        print('load train data')
        train_data = arena_util.load_json('train.json')

        print('곡간 플레이 리스트 연관도 추출')
        song_map = defaultdict(dict)
        tag_map = defaultdict(dict)
        for td in tqdm.tqdm(train_data[:25000]):
            id_list = td['songs']
            tags = td['tags']
            for i in id_list:
                for j in id_list:
                    if i != j:
                        if j not in song_map[i]:
                            song_map[i][j] = td['like_cnt']
                        else:
                            song_map[i][j] += td['like_cnt']
            for i in tags:
                for j in tags:
                    if i != j:
                        if j not in tag_map[i]:
                            tag_map[i][j] = td['like_cnt']
                        else:
                            tag_map[i][j] += td['like_cnt']

        print('연관도 정규화를 통한 확률 추출')
        for sm in tqdm.tqdm(song_map):
            song_list = pd.Series(song_map[sm]).value_counts()
            song_list = song_list/np.sum(song_list)
            song_map[sm] = [(i, song_list[i]) for i in song_list.index]

        print('song matrix writing start')
        arena_util.write_json(song_map, 'meta/song_map_test.json')
        print('finish')

        print('tag 연관도 정규화를 통한 확률 추출')
        for tm in tqdm.tqdm(tag_map):
            _sum = sum(tag_map[tm].values())
            if _sum != 0:
                for tag in tag_map[tm]:
                    tag_map[tm][tag] /= _sum

        print('tag matrix writing start')
        arena_util.write_json(tag_map, 'meta/tag_map_test.json')
        print('finish')


if __name__ == '__main__':
    # fire.Fire()
    model = CfModel()
    model.run()
