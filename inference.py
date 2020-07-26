from arena_util import load_json, write_json
from tensorflow.keras import models
from scipy.spatial.distance import cosine
from tqdm import tqdm
import numpy as np
from collections import Counter


class Inference:
    def _load_meta(self):
        self.tag_list = load_json('./arena_data/meta/AE_tag_list.json')
        self.test = load_json('./test.json')

    def run(self):
        print('loading meta data...')
        self._load_meta()

        model = models.load_model('./models/auto_encoder.h5')
        result = []

        for t in tqdm(self.test):
            input_song_vec = np.zeros((1, 707989))
            input_tag_vec = np.zeros((1, len(self.tag_list)))
            predict = dict()
            predict['id'] = t['id']
            for song in t['songs']:
                input_song_vec[0][song] = 1
            for tag in t['tags']:
                if tag in self.tag_list:
                    input_tag_vec[0][self.tag_list.index(tag)] = 1
            song_vec, tag_vec = model([input_song_vec, input_tag_vec])

            song_vec = np.array(song_vec[0])
            tag_vec = np.array(tag_vec[0])
            song_rank = song_vec.argsort()
            tag_rank = tag_vec.argsort()
            pred_songs = []
            pred_tags = []
            i = -1
            while len(pred_songs) < 100:
                if song_rank[i] not in t['songs']:
                    pred_songs.append(song_rank[i])
                i -= 1
            i = -1
            while len(pred_tags) < 10:
                if self.tag_list[tag_rank[i]] not in t['tags']:
                    pred_tags.append(self.tag_list[tag_rank[i]])
                i -= 1
            predict['songs'] = pred_songs
            predict['tags'] = pred_tags
            result.append(predict)
        write_json(result, 'result.json')


if __name__ == '__main__':
    ifrc = Inference()
    ifrc.run()
