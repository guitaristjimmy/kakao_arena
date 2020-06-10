from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import os
import json
from arena_util import load_json
from arena_util import write_json


class Mel_Preprocessing:
    def load_npy(self, _dir):
        return np.load(_dir)

    def convolution_model(self, in_shape=(48, 1024)):
        model = models.Sequential()
        model.add(layers.Conv1D(filters=256, kernel_size=7, padding='causal', input_shape=in_shape))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Conv1D(filters=256, kernel_size=3, padding='causal'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Conv1D(filters=256, kernel_size=3, padding='causal'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Conv1D(filters=256, kernel_size=3, padding='causal'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Conv1D(filters=256, kernel_size=3, padding='causal'))
        model.add(layers.MaxPooling1D(pool_size=3))
        model.add(layers.Flatten())

        return model

    def make_model(self):
        model = self.convolution_model()
        print(model.summary())
        model.save('./conv.h5', include_optimizer=False)

    def cos_similarity(self, vec_01, vec_02):
        return np.dot(vec_01, vec_02) / (np.linalg.norm(vec_01) * np.linalg.norm(vec_02))


if __name__ == '__main__':
    m_pre = Mel_Preprocessing()

    # Model 생성 및 저장 --------------------------------------------
    # model = m_pre.convolution_model()
    # print(model.summary())
    # model.save('./conv.h5', include_optimizer=False)
    #
    # del model

    # 저장된 Model 불러오기 -----------------------------------------
    model = models.load_model('./conv.h5')

    # Convolution을 통한 특징 추출 및 저장 ---------------------------
    root_path = 'D:/Kakao_Arena'

    folder_path = os.listdir(root_path)
    song_meta = load_json('./song_meta.json')

    conv_dict = {}
    for fp in folder_path:
        mels = []
        folder = os.path.join(root_path, fp)
        files = os.listdir(folder)
        for file in files:
            mels.append(np.load(os.path.join(folder, file))[:, :1024])
        mels = np.array(mels)
        print(int(fp)*1000, '~', int(fp)*1000+999, 'Convolution Vector 추출 완료 / shape: ', mels.shape)
        conv_vec = model(mels)

        for name, dat in zip(files, conv_vec):
            conv_dict[name[:-4]] = np.array(dat)
        conv_df = pd.DataFrame(conv_dict).T
        pd.to_pickle(conv_df, './conv_vector.pkl')
        print('conv_df shape : ', conv_df.shape)

    # 특징 유사도 추출 -------------------------------------------------
    print('grouping...')
    conv_df = pd.read_pickle('./conv_vector.pkl')
    random_sample = conv_df.sample(n=500, random_state=83)
    conv_df['group_idx'] = 0
    group_idx = 0
    cnt = 1
    _length = len(conv_df.index)
    for conv_idx in conv_df.index:
        print('*'*int((cnt/_length)*10))
        sim = []
        for sample_idx in random_sample.index:
            sim.append(np.array(random_sample.loc[sample_idx])[:256], np.array(conv_df.loc[conv_idx])[:256])
        conv_df.loc[conv_idx, 'group'] = sim.index(max(sim))
    print(conv_df.head())
    pd.to_pickle(conv_df, './cos_sim.pkl')
