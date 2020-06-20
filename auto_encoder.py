from arena_util import load_json, write_json
from tensorflow.keras import layers, models, metrics
from tqdm import tqdm
import random
import numpy as np


class AutoEncoder:
    def _load_train(self):
        data = load_json('./train.json')
        self.train = []
        self.song_list = set()
        self.tag_list = set()
        print('train data filtering...')
        for t in tqdm(data):
            # if t['like_cnt'] > 100:
            self.train.append(t)
            self.song_list.update(t['songs'])
            self.tag_list.update(t['tags'])
        self.song_list = list(self.song_list)
        self.tag_list = list(self.tag_list)

        write_json(self.tag_list, 'meta/AE_tag_list.json')

    def _build_model(self):
        input_song_tag = layers.Input(shape=(707989+len(self.tag_list)), name='input')
        latent = layers.Dense(128, activation='relu')(input_song_tag)
        drop = layers.Dropout(0.5)(latent)
        latent = layers.Dense(1024, activation='relu')(drop)
        drop = layers.Dropout(0.5)(latent)
        latent = layers.Dense(128, activation='relu')(drop)
        output = layers.Dense(707989+len(self.tag_list), activation='softmax')(latent)

        model = models.Model(inputs=input_song_tag, outputs=output)

        model.compile(optimizer='adam', loss='cosine_similarity', metrics=[metrics.RootMeanSquaredError()])

        return model

    def _generate_batch(self, batch_size):
        batch_x = np.zeros((batch_size, 707989+len(self.tag_list)))
        batch_y = np.zeros((batch_size, 707989+len(self.tag_list)))
        while True:
            for idx, t in enumerate(random.sample(self.train, batch_size)):
                y_vec = np.zeros(707989+len(self.tag_list))
                song_vec = np.zeros(707989)
                for i, song in enumerate(t['songs']):
                    if i%2 == 1:
                        song_vec[song] = 1
                    y_vec[song] = 1

                tag_vec = np.zeros(len(self.tag_list))
                for i, tag in enumerate(t['tags']):
                    if i%2 == 1:
                        tag_vec[self.tag_list.index(tag)] = 1
                    y_vec[707989+self.tag_list.index(tag)] = 1

                batch_x[idx, :707989] = song_vec
                batch_x[idx, 707989:] = tag_vec
                batch_y[idx, :] = y_vec
            yield batch_x, batch_y

    def run(self):
        self._load_train()
        model = self._build_model()
        print(model.summary())
        model.fit_generator(self._generate_batch(64), steps_per_epoch=len(self.train)//64, epochs=20)

        model.save('./models/auto_encoder.h5')

    def inference(self):
        model = models.load_model('./models/auto_encoder.h5')
        val = load_json('./arena_data/questions/val.json')
        tag_list = load_json('./arena_data/meta/AE_tag_list.json')
        result = []
        for v in tqdm(val):
            input_vec = np.zeros((1, 707989+len(tag_list)))
            predict = dict()
            predict['id'] = v['id']
            for s in v['songs']:
                input_vec[0][s] = 1
            for t in v['tags']:
                if t in tag_list:
                    input_vec[0][707989+tag_list.index(t)] = 1
            ans = model(input_vec)
            ans = np.array(ans)
            song_vec = ans[0][:707989]
            tag_vec = ans[0][707989:]
            song_rank = song_vec.argsort()
            tag_rank = tag_vec.argsort()
            pred_songs = []
            pred_tags = []
            i = -1
            while len(pred_songs) < 100:
                if song_rank[i] not in v['songs']:
                    pred_songs.append(song_rank[i])
                i -= 1
            i = -1
            while len(pred_tags) < 10:
                if tag_list[tag_rank[i]] not in v['tags']:
                    pred_tags.append(tag_list[tag_rank[i]])
                i -= 1
            predict['songs'] = pred_songs
            predict['tags'] = pred_tags
            result.append(predict)
        write_json(result, 'AE_results_0619.json')


if __name__ == '__main__':
    ae = AutoEncoder()
    # ae.run()
    ae.inference()
