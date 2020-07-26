from arena_util import load_json, write_json
from tensorflow.keras import layers, models, metrics, optimizers, regularizers
from tqdm import tqdm
import random
import numpy as np
from multiprocessing import Pool, cpu_count


class AutoEncoder:
    def _load_train(self):
        data = load_json('./arena_data/orig/train.json')
        self.train = []
        self.song_list = set()
        self.tag_list = set()
        print('train data filtering...')
        for t in tqdm(data):
            if t['like_cnt'] > 50:
                self.train.append(t)
                self.song_list.update(t['songs'])
                self.tag_list.update(t['tags'])
        self.song_list = list(self.song_list)
        self.tag_list = list(self.tag_list)
        self.total_song_num = 707989

        write_json(self.tag_list, 'meta/AE_tag_list.json')

    def _build_model(self):
        input_songs = layers.Input(shape=(self.total_song_num,), name='in_songs')
        input_tags = layers.Input(shape=(len(self.tag_list),), name='in_tags')

        song_dense = layers.Dense(32, kernel_regularizer=regularizers.l2(0.0001))(input_songs)
        batch_norm = layers.BatchNormalization()(song_dense)
        song_dense = layers.Activation(activation='relu')(batch_norm)

        tag_dense = layers.Dense(32, kernel_regularizer=regularizers.l2(0.0001))(input_tags)
        batch_norm = layers.BatchNormalization()(tag_dense)
        tag_dense = layers.Activation(activation='relu')(batch_norm)

        input_song_tag = layers.Concatenate(axis=1)([song_dense, tag_dense])

        dense = layers.Dense(1024, kernel_regularizer=regularizers.l2(0.0001))(input_song_tag)
        batch_norm = layers.BatchNormalization()(dense)
        dense = layers.Activation(activation='relu')(batch_norm)

        song_dense = layers.Dense(64, kernel_regularizer=regularizers.l2(0.0001))(dense)
        batch_norm = layers.BatchNormalization()(song_dense)
        song_dense = layers.Activation(activation='relu')(batch_norm)

        tag_dense = layers.Dense(64, kernel_regularizer=regularizers.l2(0.0001))(dense)
        batch_norm = layers.BatchNormalization()(tag_dense)
        tag_dense = layers.Activation(activation='relu')(batch_norm)

        output_songs = layers.Dense(self.total_song_num, activation='sigmoid', name='out_songs')(song_dense)
        output_tags = layers.Dense(len(self.tag_list), activation='sigmoid', name='out_tags')(tag_dense)

        model = models.Model(inputs=[input_songs, input_tags], outputs=[output_songs, output_tags])

        model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=[metrics.RootMeanSquaredError()])

        return model

    def _generate_batch(self, batch_size):
        # y_songs = np.zeros((batch_size, self.total_song_num))
        # y_tags = np.zeros((batch_size, len(self.tag_list)))
        while True:
            x_songs = np.zeros((batch_size, self.total_song_num))
            x_tags = np.zeros((batch_size, len(self.tag_list)))
            y_songs = np.zeros((batch_size, self.total_song_num))
            y_tags = np.zeros((batch_size, len(self.tag_list)))
            # y = np.zeros((batch_size, 740000))
            for idx, t in enumerate(random.sample(self.train, batch_size)):
                # y_song_vec = np.zeros(self.total_song_num)
                # y_tag_vec = np.zeros(len(self.tag_list))
                # song_vec = np.zeros(707989)
                for i, song in enumerate(t['songs']):
                    if i%2 == 0:
                        x_songs[idx, song] = 1
                    # y_song_vec[song] = 1
                    y_songs[idx, song] = 1
                # tag_vec = np.zeros(len(self.tag_list))
                for i, tag in enumerate(t['tags']):
                    if i%2 == 0:
                        x_tags[idx, self.tag_list.index(tag)] = 1
                    # y_tag_vec[self.tag_list.index(tag)] = 1
                    y_tags[idx, self.tag_list.index(tag)] = 1
                # x_songs[idx, :] = song_vec
                # x_tags[idx, :] = tag_vec
                # y_songs[idx, :] = y_song_vec
                # y_tags[idx, :] = y_tag_vec
            yield {'in_songs': x_songs, 'in_tags': x_tags}, {'out_songs': y_songs, 'out_tags': y_tags}

    def run(self):
        self._load_train()
        model = self._build_model()
        # model = models.load_model('./models/auto_encoder.h5')
        print(model.summary())
        model.fit_generator(self._generate_batch(64), steps_per_epoch=len(self.train)//64, epochs=5)

        model.save('./models/auto_encoder.h5')

    def val_inference(self):
        model = models.load_model('./models/auto_encoder.h5')
        val = load_json('./arena_data/questions/val.json')
        tag_list = load_json('./arena_data/meta/AE_tag_list.json')
        result = []

        for v in tqdm(val):
            input_song_vec = np.zeros((1, 707989))
            input_tag_vec = np.zeros((1, len(tag_list)))
            predict = dict()
            predict['id'] = v['id']
            for s in v['songs']:
                input_song_vec[0][s] = 1
            for t in v['tags']:
                if t in tag_list:
                    input_tag_vec[0][tag_list.index(t)] = 1
            song_vec, tag_vec = model([input_song_vec, input_tag_vec])

            song_vec = np.array(song_vec[0])
            tag_vec = np.array(tag_vec[0])
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
        write_json(result, 'AE_results.json')


if __name__ == '__main__':
    ae = AutoEncoder()
    ae.run()
    while True:
        val_flag = input('validation inference start? : (yes|no)')
        if val_flag == 'yes':
            ae.val_inference()
        elif val_flag == 'no':
            print('exit process')
            break
