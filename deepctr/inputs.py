# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,weichenswc@163.com

"""

from collections import defaultdict
from itertools import chain

from tensorflow.python.keras.layers import Embedding, Lambda
from tensorflow.python.keras.regularizers import l2

from .layers.sequence import SequencePoolingLayer, WeightedSequenceLayer
from .layers.utils import Hash


def get_inputs_list(inputs):
    return list(chain(*list(map(lambda x: x.values(), filter(lambda x: x is not None, inputs)))))

cnt = 0
def create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, seed, l2_reg,
                          prefix='sparse_', seq_mask_zero=True):
    sparse_embedding = {}
    global cnt
    cnt += 1
    for feat in sparse_feature_columns:  # 对每个维度都做一个embedding层
        # print("ft_sparse_feature_columns_____test", " cnt: ", str(cnt), "type: ", type(feat), "  e.g: ", feat)  # SparseFeat(name='C1', vocabulary_size=27, embedding_dim=1, use_hash=False, vocabulary_path=None, dtype='int32', embeddings_initializer=<tensorflow.python.ops.init_ops.Zeros object at 0x7fcbcbab62b0>, embedding_name='C1', group_name='default_group', trainable=True)
        emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                        embeddings_initializer=feat.embeddings_initializer,
                        embeddings_regularizer=l2(l2_reg),
                        name=prefix + '_emb_' + feat.embedding_name)  # https://blog.csdn.net/sinat_22510827/article/details/90727435#commentBox
        # print("ft_emb's type: ", type(emb))  # <class 'tensorflow.python.keras.layers.embeddings.Embedding'>
        emb.trainable = feat.trainable  # trainable参数哪里来的?  slot?
        sparse_embedding[feat.embedding_name] = emb

    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            # if feat.name not in sparse_embedding:
            emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                            embeddings_initializer=feat.embeddings_initializer,
                            embeddings_regularizer=l2(l2_reg),
                            name=prefix + '_seq_emb_' + feat.name,
                            mask_zero=seq_mask_zero)
            emb.trainable = feat.trainable
            sparse_embedding[feat.embedding_name] = emb
    return sparse_embedding


def get_embedding_vec_list(embedding_dict, input_dict, sparse_feature_columns, return_feat_list=(), mask_feat_list=()):
    embedding_vec_list = []
    for fg in sparse_feature_columns:
        feat_name = fg.name
        if len(return_feat_list) == 0 or feat_name in return_feat_list:
            if fg.use_hash:
                lookup_idx = Hash(fg.vocabulary_size, mask_zero=(feat_name in mask_feat_list), vocabulary_path=fg.vocabulary_path)(input_dict[feat_name])
            else:
                lookup_idx = input_dict[feat_name]

            embedding_vec_list.append(embedding_dict[feat_name](lookup_idx))

    return embedding_vec_list


def create_embedding_matrix(feature_columns, l2_reg, seed, prefix="", seq_mask_zero=True):
    from . import feature_column as fc_lib

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.VarLenSparseFeat), feature_columns)) if feature_columns else []
    # print("varlen_sparse_feature_columns: ", varlen_sparse_feature_columns)  # []
    # print("prefix: ", prefix)
    sparse_emb_dict = create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, seed,
                                            l2_reg, prefix=prefix + 'sparse', seq_mask_zero=seq_mask_zero)  # emb_dict  {name: Embedding(), ...}
    return sparse_emb_dict


def embedding_lookup(sparse_embedding_dict, sparse_input_dict, sparse_feature_columns, return_feat_list=(),
                     mask_feat_list=(), to_list=False):
    group_embedding_dict = defaultdict(list)  # 为一个不存在的键提供默认值，从而避免KeyError异常  https://blog.csdn.net/u014248127/article/details/79338543
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if (len(return_feat_list) == 0 or feature_name in return_feat_list):
            if fc.use_hash:
                lookup_idx = Hash(fc.vocabulary_size, mask_zero=(feature_name in mask_feat_list), vocabulary_path=fc.vocabulary_path)(
                    sparse_input_dict[feature_name])
            else:
                lookup_idx = sparse_input_dict[feature_name]  # Input()   shape: (?, 1)
                # print("ft test type(lookup_idx) is: ", type(lookup_idx), "    content: ", lookup_idx)  # <class 'tensorflow.python.framework.ops.Tensor'>     content:  Tensor("C1_3:0", shape=(?, 1), dtype=int32)

            # input() tensor作为Embedding层的输入      input层拼接上embedding层的一个值
            # tmp_emb = sparse_embedding_dict[embedding_name]
            # tmp_final = sparse_embedding_dict[embedding_name](lookup_idx)
            group_embedding_dict[fc.group_name].append(sparse_embedding_dict[embedding_name](lookup_idx))  # {"default_group": [Embedding()(Input()), ...], }
    if to_list:
        return list(chain.from_iterable(group_embedding_dict.values()))  # https://docs.python.org/zh-cn/3/library/itertools.html#itertools.chain.from_iterable
    return group_embedding_dict


def varlen_embedding_lookup(embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            lookup_idx = Hash(fc.vocabulary_size, mask_zero=True, vocabulary_path=fc.vocabulary_path)(sequence_input_dict[feature_name])
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)
    return varlen_embedding_vec_dict


def get_varlen_pooling_list(embedding_dict, features, varlen_sparse_feature_columns, to_list=False):
    pooling_vec_list = defaultdict(list)
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        combiner = fc.combiner
        feature_length_name = fc.length_name
        if feature_length_name is not None:
            if fc.weight_name is not None:
                seq_input = WeightedSequenceLayer(weight_normalization=fc.weight_norm)(
                    [embedding_dict[feature_name], features[feature_length_name], features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=False)(
                [seq_input, features[feature_length_name]])
        else:
            if fc.weight_name is not None:
                seq_input = WeightedSequenceLayer(weight_normalization=fc.weight_norm, supports_masking=True)(
                    [embedding_dict[feature_name], features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=True)(
                seq_input)
        pooling_vec_list[fc.group_name].append(vec)
    if to_list:
        return chain.from_iterable(pooling_vec_list.values())
    return pooling_vec_list


def get_dense_input(features, feature_columns):
    """
    features: OrderedDict()     {name: Input}
    feature_columns: list       [SparseFeat, DenseFeat]
    """
    from . import feature_column as fc_lib
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.DenseFeat), feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        if fc.transform_fn is None:
            dense_input_list.append(features[fc.name])  # 获取input() tensor
        else:
            transform_result = Lambda(fc.transform_fn)(features[fc.name])
            dense_input_list.append(transform_result)
    return dense_input_list


def mergeDict(a, b):
    c = defaultdict(list)
    for k, v in a.items():
        c[k].extend(v)
    for k, v in b.items():
        c[k].extend(v)
    return c
