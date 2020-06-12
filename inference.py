# import fire
import tqdm
import pandas as pd
import numpy as np
import multiprocessing
from collections import Counter, defaultdict
from arena_util import load_json
from arena_util import write_json

import time

class Inference:

    def _load_meta(self):
        print('load song meta data')
        self.song_meta = load_json('./arena_data/meta/song_meta_with_tags.json')
        print('load song relation map')
        self.song_map = load_json('./arena_data/meta/song_map_test.json')
        print('load tag relation map')
        self.tag_map = load_json('./arena_data/meta/tag_map.json')
        print('load tag dict')
        self.tag_dict = load_json('./arena_data/meta/tag_dict.json')
        print('load kmeans group table')
        self.kmeans = pd.read_pickle('./arena_data/meta/kmeans.pkl')
        self.kmeans.index = self.kmeans.index.map(int)

    def _load_val(self):
        self.val = load_json('./arena_data/questions/val.json')

    def _relation_probability(self, song_probability, vs):
        print('test ok', vs)
        song_probability[vs] = -1e9
        if vs in self.song_map:
            df = pd.DataFrame(self.song_map[vs])
            df.set_index(keys=0, inplace=True)
        g_id = self.kmeans.loc[vs, 'group']
        return df, g_id

    def _colaborate_probability(self, manager, vs_list):

        relation_songs = pd.DataFrame()

        for vs in vs_list:
            manager[vs] = -1e9

            e = False
            try:
                sm = self.song_map[str(vs)]
            except:
                e = True

            if not e:
                if len(relation_songs) == 0:
                    relation_songs = pd.DataFrame(self.song_map[str(vs)])
                    relation_songs.set_index(keys=0, inplace=True)
                else:
                    df = pd.DataFrame(self.song_map[str(vs)])
                    df.set_index(keys=0, inplace=True)
                    relation_songs = relation_songs.add(df, fill_value=0)


        if len(relation_songs.index) > 0:
            relation_songs.fillna(value=0, inplace=True)
            for i in relation_songs.index:
                try:
                    manager[i] += float(relation_songs.loc[i])
                except IndexError:
                    manager[i] = float(relation_songs.loc[i])

    def _group_probability(self, manager, vs_list):
        group_ids = []
        for vs in vs_list:
            g_id = self.kmeans.loc[vs, 'group']
            group_ids.append(str(g_id))

        group_ids_cnt = Counter(group_ids)
        g_sum = len(group_ids)
        group_list = [(gid, group_ids_cnt[gid]/g_sum) for gid in group_ids_cnt]

        for g, p in group_list:
            kp = p
            for i in self.kmeans[self.kmeans['group'] == int(g)].index:
                try:
                    manager[i] += p
                except IndexError:
                    manager[i] = p

    def _from_tag_probability(self, manager, vt_list):
        if len(vt_list) != 0:
            tp = 1/len(vt_list)
            for tag in vt_list:
                if tag in self.tag_dict:
                    for i in self.tag_dict[tag]:
                        try:
                            manager[i] += tp
                        except IndexError:
                            manager[i] = tp

    def _predict_songlist_multi(self, vt_list, vs_list):
        p_manager = multiprocessing.Manager()
        song_probability = p_manager.dict()

        cp = multiprocessing.Process(target=self._colaborate_probability, args=(song_probability, vs_list))
        gp = multiprocessing.Process(target=self._group_probability, args=(song_probability, vs_list))
        tp = multiprocessing.Process(target=self._from_tag_probability, args=(song_probability, vt_list))

        cp.start()
        gp.start()
        tp.start()

        print('process start')

        cp.join()
        gp.join()
        tp.join()

        print('process join')

        sp = pd.Series(song_probability).sort_values(ascending=False)

        cp.close()
        gp.close()
        tp.close()

        return list(sp.index[:100])


    def _predict_songlist(self, vt_list, vs_list):
        # song_probability = defaultdict(float)
        song_probability = np.zeros(len(self.song_meta))
        # relation_songs = pd.DataFrame()
        # group_ids = []

        for vs in vs_list:
            song_probability[vs] = -1e9

            e = False
            try:
                sm = self.song_map[str(vs)]
            except:
                e = True

            if not e:
                for s in sm:
                    song_probability[s[0]] += s[1]

            g_id = self.kmeans.loc[vs, 'group']
            group_ids.append(str(g_id))

        if len(relation_songs.index) > 0:
            relation_songs.fillna(value=0, inplace=True)
            for i in relation_songs.index:
                song_probability[int(i)] += float(relation_songs.loc[i])

        group_ids_cnt = Counter(group_ids)
        g_sum = len(group_ids)
        group_list = [(gid, group_ids_cnt[gid]/g_sum) for gid in group_ids_cnt]

        for g, p in group_list:
            # kp = p
            for i in self.kmeans[self.kmeans['group'] == int(g)].index:
                song_probability[int(i)] += p

        if len(vt_list) != 0:
            tp = 1/len(vt_list)
            for tag in vt_list:
                if tag in self.tag_dict:
                    for i in self.tag_dict[tag]:
                        song_probability[int(i)] += tp
        sp = pd.Series(song_probability).sort_values(ascending=False)
        return list(sp.index[:100])

    def _predict_taglist(self, vt_list, vs_list):
        relation_tags = pd.Series()
        tag_probability = defaultdict(float)
        for vt in vt_list:
            tag_probability[vt] = -1e9
            if vt in self.tag_map:
                relation_tags.add(pd.Series(self.tag_map[vt]), fill_value=0)
        if len(relation_tags) > 0:
            relation_tags.fillna(value=0, inplace=True)
            for i in relation_tags.index:
                tag_probability[i] += relation_tags[i]

        tags = []
        for vs in vs_list:
            if 'tags' in self.song_meta[vs]:
                tags += self.song_meta[vs]['tags']
        tag_cnt = Counter(tags)
        _sum = sum(tag_cnt.values())
        for tag in tag_cnt:
            tag_probability[tag] += tag_cnt[tag]/_sum
        tag_probability = pd.Series(tag_probability).sort_values(ascending=False)
        return list(tag_probability.index)[:10]

    def run(self):
        print('load meta data')
        self._load_meta()

        print('load val data')
        self._load_val()

        print('predict song list and tag list')
        result = []
        for v in tqdm.tqdm(self.val):
            predict = dict()
            predict['id'] = v['id']
            predict['songs'] = self._predict_songlist(v['tags'], v['songs'])
            predict['tags'] = self._predict_taglist(v['tags'], predict['songs'])
            result.append(predict)

        print('write result file')
        write_json(result, 'result.json')


if __name__ == '__main__':
    # fire.Fire()
    inference = Inference()
    inference.run()
