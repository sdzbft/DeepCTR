import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names


# tf_1.15.4 没问题，但是tf2不行了
# # 可用  开启后显存占用从 10769MiB => 803MiB
# import os
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# # 指定使用哪块GPU训练
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config = tf.ConfigProto()
# # 设置最大占有GPU不超过显存的70%
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
# # 重点：设置动态分配GPU
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))   #


## tf2 用这个 https://blog.csdn.net/weixin_44885180/article/details/116377820
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


if __name__ == "__main__":
    data = pd.read_csv('./criteo_sample.txt')   #200条数据，39维特征，1维label

    sparse_features = ['C' + str(i) for i in range(1, 27)] # 特征连续型的有13个，类别型的26个
    dense_features = ['I' + str(i) for i in range(1, 14)]  # 同上

    data[sparse_features] = data[sparse_features].fillna('-1', )   # dataframe[list]时也是一个dataframe
    data[dense_features] = data[dense_features].fillna(0, )  # arange返回的是一个ndarray(np.arange)；而range返回一个list。arange允许步长为小数，而range不允许
    target = ['label']

    # 1.Label Encoding for sparse features, and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()   # object变为int了
        data[feat] = lbe.fit_transform(data[feat])  # fit_transform: https://blog.csdn.net/weixin_38278334/article/details/82971752
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field, and record dense feature field name
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)   # "data[feat].max() + 1" 用 data[feat].nunique() 代替也行
                              for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
                                                                            for feat in dense_features]  # list加法类似extend

    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    # generate feature columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    # print("feature_names: ", feature_names)

    # 3.generate input data for model
    train, test = train_test_split(data, test_size=0.2, random_state=2020)  # *args 表示任何多个无名参数，它本质是一个 tuple。**kwargs 表示关键字参数，它本质上是一个 dict
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    # print("futong_test: ", linear_feature_columns[0])  # SparseFeat(name='C1', vocabulary_size=27, embedding_dim=4, use_hash=False, vocabulary_path=None, dtype='int32', embeddings_initializer=<tensorflow.python.keras.initializers.RandomNormal object at 0x7f6489f54b38>, embedding_name='C1', group_name='default_group', trainable=True)
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=100, verbose=2, validation_split=0.2, )
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

