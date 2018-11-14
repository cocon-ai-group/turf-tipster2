import os
import csv
import sys
import argparse
import itertools
import random
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from xgboost import DMatrix
from collections import Counter, defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR

parser = argparse.ArgumentParser(description='LambdaRank example: Prefigure horse rase')
parser.add_argument('--train', '-t', default='',
					help='Train file')
parser.add_argument('--history', '-i', default='',
					help='Horse history file')
parser.add_argument('--test', '-e', default='',
					help='Test file')
parser.add_argument('--race', '-r', default='',
					help='Race data (File name or [Horse|Jockey]* list)')
parser.add_argument('--meta', '-m', default='',
					help='Race Meta data if Race data is list')
parser.add_argument('--num', '-n', type=int, default=18,
					help='Max number of gates')
parser.add_argument('--position', '-p', type=int, default=7,
					help='Count number of gates')
parser.add_argument('--horses', '-o', type=int, default=8000,
					help='Max number of horses')
parser.add_argument('--algorizm', '-a', default='xgb',
					help='Algorizm (xgb|lgbm|ensemble)')
args = parser.parse_args()

train_src = args.train
history_file = args.history
test_src = args.test
in_data = args.race  # 入力データ = 馬名|騎手名,馬名|騎手名,馬名|騎手名・・・
in_meta = args.meta  # メタデータ = 場所|馬場|天気|距離|日付YYYYMMDD
num_gates = args.num
max_position = args.position
max_horses = args.horses
use_algorizm = args.algorizm
use_history = False

if use_algorizm=='xgb':
	N_ansemble = 10
elif use_algorizm=='lgbm':
	N_ansemble = 10
elif use_algorizm=='ensemble':
	N_ansemble = 20
else:
	print('invalid algorizm.')
	exit()

df_history = pd.DataFrame()
if len(history_file) > 0 and os.path.isfile(history_file):
	X = pd.read_csv(history_file, index_col=0, header=None)
	use_history = True
	pca = PCA(n_components=10)
	pca_X = pd.DataFrame(pca.fit_transform(X), index=X.index, columns=['pca%d'%i for i in range(10)])
	tsvd = TruncatedSVD(n_components=10)
	tsvd_X = pd.DataFrame(tsvd.fit_transform(X), index=X.index, columns=['tsvd%d'%i for i in range(10)])
	grp = GaussianRandomProjection(n_components=10, eps=0.1)
	grp_X = pd.DataFrame(grp.fit_transform(X), index=X.index, columns=['grp%d'%i for i in range(10)])
	srp = SparseRandomProjection(n_components=10, dense_output=True)
	srp_X = pd.DataFrame(srp.fit_transform(X), index=X.index, columns=['srp%d'%i for i in range(10)])
	df_history = X.join([pca_X, tsvd_X, grp_X, srp_X])
	del pca, tsvd, grp, srp, pca_X, tsvd_X, grp_X, srp_X

def norm_racedata(data, query):
	cur_pos = 0
	for q in query:
		data[cur_pos:cur_pos+q] = data[cur_pos:cur_pos+q] - np.min(data[cur_pos:cur_pos+q])
		data[cur_pos:cur_pos+q] = data[cur_pos:cur_pos+q] / np.sum(data[cur_pos:cur_pos+q])
		cur_pos += q
	return data

def get_horsemeta(name, date):
	if not use_history:
		return []
	else:
		if len(date) == 10 and date.replace("/","").isdigit() and date[4] == '/' and date[7] == '/':  # YYYY/MM/DD
			sdate_y = date[2:4]
			sdate_m = date[5:7]
		elif date.isdigit() and len(date) == 8:  # YYYYMMDD
			sdate_y = date[2:4]
			sdate_m = date[4:6]
		elif date.isdigit() and len(date) == 6:  # YYMMDD
			sdate_y = date[0:2]
			sdate_m = date[2:4]
		else:
			print('date format error: %s'%date)
			return []
		colname = name + sdate_y + sdate_m
		if colname in df_history.index:
			return df_history.loc[colname].values.tolist()
		else:
			date_y = int(sdate_y)
			date_m = int(sdate_m)
			for i in reversed(range(1,date_m)):
				colname = name + sdate_y + '%02d'%i
				if colname in df_history.index:
					return df_history.loc[colname].values.tolist()
			for i in reversed(range(16,date_y)):
				for j in reversed(range(1,13)):
					colname = name + '%02d%02d'%(i,j)
					if colname in df_history.index:
						return df_history.loc[colname].values.tolist()
			return np.zeros((len(df_history.columns),)).tolist()

all_horse_name = []
all_jockey_name = []
all_where_str = []
all_baba_str = []
all_tenki_str = []
all_races_train = []
all_races_test = []

def read_file(file, all_races):
	with open(file, 'r') as csvfile:
		csvreader = csv.reader(csvfile)
		for race in csvreader:
			all_races.append(race)
			race_meta = race[0].split('|')
			if len(race_meta) > 5:
				all_where_str.append(race_meta[1])
				all_baba_str.append(race_meta[2])
				all_tenki_str.append(race_meta[4])
			for e in range(1, len(race)):
				entry = race[e]
				result = entry.split('|')
				if len(result) >= 3:
					all_horse_name.append(result[1])
					all_jockey_name.append(result[2])
read_file(train_src, all_races_train)

horse_names = all_horse_name
horse_name_count = Counter(horse_names)
jockey_name_count = Counter(all_jockey_name)
if max_horses > 0:
	name, _ = zip(*horse_name_count.most_common(max_horses-2))
	horse_names = list(name)

if len(test_src) > 0:
	read_file(test_src, all_races_test)

horse_names.append('その他')
horse_names.append('未出走')
jockey_names = list(set(all_jockey_name))
jockey_names.append('その他')

del all_horse_name, all_jockey_name

horses_i = LabelEncoder().fit(horse_names)
jockeys_i = LabelEncoder().fit(jockey_names)
where_i = LabelEncoder().fit(all_where_str)
baba_i = LabelEncoder().fit(all_baba_str)
tenki_i = LabelEncoder().fit(all_tenki_str)

def get_jockey_i(name):
	if name in jockeys_i.classes_:
		return jockeys_i.transform([name])[0]
	else:
		return jockeys_i.transform(['その他'])[0]
def get_horse_i(name):
	if name in horses_i.classes_:
		return horses_i.transform([name])[0]
	else:
		return horses_i.transform(['その他'])[0]

def get_race_odds(all_races):
	race_odds = []
	for race in all_races:
		race_meta = race[0].split('|')
		if len(race_meta) > 6:
			odds = race_meta[5].split(':')
			race_odds.append([
				int(odds[0]), # 単勝
				int(odds[1].split('_')[0]), # 複勝
				int(odds[1].split('_')[1]), # 複勝
				int(odds[1].split('_')[2]), # 複勝
				int(odds[2]), # 枠連
				int(odds[3]), # 馬連
				int(odds[4].split('_')[0]), # ワイド
				int(odds[4].split('_')[1]), # ワイド
				int(odds[4].split('_')[2]), # ワイド
				int(odds[5]), # 馬単
				int(odds[6]), # 三連複
				int(odds[7]) # 三連単
			])
	return race_odds


def get_race_gets(all_races, all_races_rank, all_races_query, all_races_target):
	for race in all_races:
		race_meta = race[0].split('|')
		if len(race_meta) > 6:
			where_num = where_i.transform([race_meta[1]])[0]
			baba_num = baba_i.transform([race_meta[2]])[0]
			tenki_num = tenki_i.transform([race_meta[4]])[0]
			len_num = int(race_meta[3])
			date_s = race_meta[6]
			target = []
			for e in range(1, len(race)):
				entry = race[e]
				result = entry.split('|')
				if len(result) >= 3:
					horse_num = get_horse_i(result[1])
					jockey_num = get_jockey_i(result[2])
					target.append(([horse_num, jockey_num, where_num, baba_num, tenki_num, len_num] + get_horsemeta(result[1], date_s),min(e, max_position)))
			random.shuffle(target)
			all_races_query.append(len(target))
			for tgt in target:
				all_races_rank.append(tgt[0])
				all_races_target.append(tgt[1])

if len(test_src) > 0:
	all_races_rank_test = []
	all_races_query_test = []
	all_races_target_test = []
	get_race_gets(all_races_test, all_races_rank_test, all_races_query_test, all_races_target_test)
	all_races_rank_test = np.array(all_races_rank_test)
	all_races_query_test = np.array(all_races_query_test)
	all_races_target_test = np.array(all_races_target_test)
	test_validation_regression = np.zeros((len(all_races_target_test),N_ansemble))

if len(in_data)!=0 and len(in_meta)!=0:
	predict_races_target = []
	where_num = where_i.transform([in_meta.split('|')[0]])[0]
	baba_num = baba_i.transform([in_meta.split('|')[1]])[0]
	tenki_num = tenki_i.transform([in_meta.split('|')[2]])[0]
	len_num = int(in_meta.split('|')[3])
	date_s = in_meta.split('|')[4]
	for g in in_data.split(','):
		gs = g.split('|')
		horse_num = get_horse_i(gs[0])
		jockey_num = get_jockey_i(gs[1])
		predict_races_target.append([horse_num, jockey_num, where_num, baba_num, tenki_num, len_num] + get_horsemeta(g.split('|')[0], date_s))
	predict_races_target = np.array(predict_races_target)
	predict_validation_regression = np.zeros((len(predict_races_target),N_ansemble))

all_races_train = np.array(all_races_train)

def main_xgb(fold_offset):
	all_races_rank_regression = []
	all_races_query_regression = []
	all_races_target_regression = []
	get_race_gets(all_races_train, all_races_rank_regression, all_races_query_regression, all_races_target_regression)
	all_races_rank_regression = np.array(all_races_rank_regression)
	all_races_query_regression = np.array(all_races_query_regression)
	all_races_target_regression = np.array(all_races_target_regression)
	
	if use_history:
		categorical_feature = [0,1,2,3,4]+list(range(6,21))
	else:
		categorical_feature = [0,1,2,3,4]
	categorical_dim = [int(np.max(all_races_rank_regression[:,c])) for c in categorical_feature]
	
	del all_races_rank_regression, all_races_query_regression, all_races_target_regression
	
	def get_matrix(mat):
		shape = list(mat.shape)
		shape[1] += int(np.sum(categorical_dim)) - len(categorical_feature)
		matrix = np.zeros(tuple(shape))
		cur_dim = 0
		cur_ind = 0
		while cur_dim < shape[1]:
			if cur_ind in categorical_feature:
				dim = categorical_dim[categorical_feature.index(cur_ind)]
				for z in range(shape[0]):
					matrix[z,cur_dim+int(mat[z,cur_ind])] = 1
				cur_dim += dim
			else:
				matrix[:,cur_dim] = mat[:,cur_ind]
				cur_dim += 1
			cur_ind += 1
		return matrix
	
	if len(test_src) > 0:
		all_races_rank_test_x = get_matrix(all_races_rank_test)
	if len(in_data)!=0 and len(in_meta)!=0:
		predict_races_target_x = get_matrix(predict_races_target)
	
	for fold_id, (train_index, test_index) in enumerate(KFold(n_splits=10).split(all_races_train)):
		all_races_train_train = all_races_train[train_index]
		all_races_train_valid = all_races_train[test_index]
		all_races_rank_train_train = []
		all_races_query_train_train = []
		all_races_target_train_train = []
		all_races_rank_train_valid = []
		all_races_query_train_valid = []
		all_races_target_train_valid = []
		get_race_gets(all_races_train_train, all_races_rank_train_train, all_races_query_train_train, all_races_target_train_train)
		get_race_gets(all_races_train_valid, all_races_rank_train_valid, all_races_query_train_valid, all_races_target_train_valid)
		all_races_rank_train_train = get_matrix(np.array(all_races_rank_train_train))
		all_races_query_train_train = np.array(all_races_query_train_train)
		all_races_target_train_train = np.array(all_races_target_train_train)
		all_races_rank_train_valid = get_matrix(np.array(all_races_rank_train_valid))
		all_races_query_train_valid = np.array(all_races_query_train_valid)
		all_races_target_train_valid = np.array(all_races_target_train_valid)
		
		xgb_params =  {
			'objective': 'rank:pairwise',
			'eta': 0.1,
			'gamma': 0.0001,
			'min_child_weight': 0.1,
			'max_depth': 6
		}
		xgtrain = DMatrix(all_races_rank_train_train, all_races_target_train_train)
		xgtrain.set_group(all_races_query_train_train)
		xgvalid = DMatrix(all_races_rank_train_valid, all_races_target_train_valid)
		xgvalid.set_group(all_races_query_train_valid)

		del all_races_train_train, all_races_train_valid, all_races_rank_train_train, all_races_target_train_train, all_races_query_train_train, all_races_rank_train_valid, all_races_target_train_valid, all_races_query_train_valid

		xgb_clf = xgb.train(
			xgb_params,
			xgtrain,
			num_boost_round=10,
			evals=[(xgvalid, 'validation')]
		)
		del xgtrain, xgvalid
		
		if len(test_src) > 0:
			dst = norm_racedata(xgb_clf.predict(DMatrix(all_races_rank_test_x)), all_races_query_test)
			for dst_ind in range(len(dst)):
				test_validation_regression[dst_ind][fold_offset+fold_id] = dst[dst_ind]
			cur_pos = 0
		if len(in_data)!=0 and len(in_meta)!=0:
			dst = norm_racedata(xgb_clf.predict(DMatrix(predict_races_target_x)), [len(predict_races_target_x)])
			for dst_ind in range(len(dst)):
				predict_validation_regression[dst_ind][fold_offset+fold_id] = dst[dst_ind]

def main_lgbm(fold_offset):
	
	for fold_id, (train_index, test_index) in enumerate(KFold(n_splits=10).split(all_races_train)):
		all_races_train_train = all_races_train[train_index]
		all_races_train_valid = all_races_train[test_index]
		all_races_rank_train_train = []
		all_races_query_train_train = []
		all_races_target_train_train = []
		all_races_rank_train_valid = []
		all_races_query_train_valid = []
		all_races_target_train_valid = []
		get_race_gets(all_races_train_train, all_races_rank_train_train, all_races_query_train_train, all_races_target_train_train)
		get_race_gets(all_races_train_valid, all_races_rank_train_valid, all_races_query_train_valid, all_races_target_train_valid)
		all_races_rank_train_train = np.array(all_races_rank_train_train)
		all_races_query_train_train = np.array(all_races_query_train_train)
		all_races_target_train_train = np.array(all_races_target_train_train)
		all_races_rank_train_valid = np.array(all_races_rank_train_valid)
		all_races_query_train_valid = np.array(all_races_query_train_valid)
		all_races_target_train_valid = np.array(all_races_target_train_valid)
		
		lgbm_params =  {
			'task': 'train',
			'boosting_type': 'gbdt',
			'objective': 'lambdarank',
			'metric': 'ndcg',   # for lambdarank
			'ndcg_eval_at': [1,2,3],  # for lambdarank
			'max_position': max_position,  # for lambdarank
			'learning_rate': 1e-8,
			'min_data': 1,
			'min_data_in_bin': 1,
		}
		lgtrain = lgb.Dataset(all_races_rank_train_train, all_races_target_train_train, categorical_feature=[0,1,2,3,4,7]+list(range(8,23)), group=all_races_query_train_train)
		lgvalid = lgb.Dataset(all_races_rank_train_valid, all_races_target_train_valid, categorical_feature=[0,1,2,3,4,7]+list(range(8,23)), group=all_races_query_train_valid)
		lgb_clf = lgb.train(
			lgbm_params,
			lgtrain,
			categorical_feature=[0,1,2,3,4]+list(range(6,21)),
			num_boost_round=10,
			valid_sets=[lgtrain, lgvalid],
			valid_names=['train','valid'],
			early_stopping_rounds=2,
			verbose_eval=1
		)
		
		if len(test_src) > 0:
			dst = norm_racedata(lgb_clf.predict(all_races_rank_test), all_races_query_test)
			for dst_ind in range(len(dst)):
				test_validation_regression[dst_ind][fold_offset+fold_id] = dst[dst_ind]
			cur_pos = 0
		if len(in_data)!=0 and len(in_meta)!=0:
			dst = norm_racedata(lgb_clf.predict(predict_races_target), [len(predict_races_target)])
			for dst_ind in range(len(dst)):
				predict_validation_regression[dst_ind][fold_offset+fold_id] = dst[dst_ind]

def main_emsemble():	
	df_outfile = sys.stdout
	
	if len(test_src) > 0:
		race_odds = get_race_odds(all_races_test)
		ret_score = [0,0,0,0,0,0,0,0,0]
		ret_hitnum = [0,0,0,0,0,0,0,0,0]
		num_retrace = 0
		cur_pos = 0
		
		test_validation_result = test_validation_regression.mean(axis=1)
		for i, o in zip(all_races_query_test, race_odds):
			order = np.argsort(test_validation_result[cur_pos:cur_pos+i])
			order_t = np.argsort(all_races_target_test[cur_pos:cur_pos+i])
			if order[0] == order_t[0]:  # 単勝あたり
				ret_score[0] += o[0]
				ret_hitnum[0] += 1
			if order[0] == order_t[0] or order[1] == order_t[0] or order[2] == order_t[0]:  # 複勝あたり
				ret_score[1] += o[1]
				ret_hitnum[1] += 1
			if order[0] == order_t[1] or order[1] == order_t[1] or order[2] == order_t[1]:  # 複勝あたり
				ret_score[1] += o[2]
				ret_hitnum[1] += 1
			if order[0] == order_t[2] or order[1] == order_t[2] or order[2] == order_t[2]:  # 複勝あたり
				ret_score[1] += o[3]
				ret_hitnum[1] += 1
			if order[0] == order_t[0]:  # 複勝あたり
				ret_score[2] += o[1]
				ret_hitnum[2] += 1
			elif order[0] == order_t[1]:  # 複勝あたり
				ret_score[2] += o[2]
				ret_hitnum[2] += 1
			elif order[0] == order_t[2]:  # 複勝あたり
				ret_score[2] += o[3]
				ret_hitnum[2] += 1
			if (order[0] == order_t[0] and order[1] == order_t[1]) or (order[0] == order_t[1] and order[1] == order_t[0]):  # 馬連あたり
				ret_score[3] += o[5]
				ret_hitnum[3] += 1
			if (order[0] == order_t[0] or order[1] == order_t[0] or order[2] == order_t[0]) and (order[0] == order_t[1] or order[1] == order_t[1] or order[2] == order_t[1]):  # ワイドあたり
				ret_score[4] += o[6]
				ret_hitnum[4] += 1
			if (order[0] == order_t[0] or order[1] == order_t[0] or order[2] == order_t[0]) and (order[0] == order_t[2] or order[1] == order_t[2] or order[2] == order_t[2]):  # ワイドあたり
				ret_score[4] += o[7]
				ret_hitnum[4] += 1
			if (order[0] == order_t[1] or order[1] == order_t[1] or order[2] == order_t[1]) and (order[0] == order_t[2] or order[1] == order_t[2] or order[2] == order_t[2]):  # ワイドあたり
				ret_score[4] += o[8]
				ret_hitnum[4] += 1
			if (order[0] == order_t[0] and order[1] == order_t[1]) or (order[0] == order_t[1] and order[1] == order_t[0]):  # ワイドあたり
				ret_score[5] += o[6]
				ret_hitnum[5] += 1
			elif (order[0] == order_t[0] and order[1] == order_t[2]) or (order[0] == order_t[2] and order[1] == order_t[0]):  # ワイドあたり
				ret_score[5] += o[7]
				ret_hitnum[5] += 1
			elif (order[0] == order_t[1] and order[1] == order_t[2]) or (order[0] == order_t[2] and order[1] == order_t[1]):  # ワイドあたり
				ret_score[5] += o[8]
				ret_hitnum[5] += 1
			if order[0] == order_t[0] and order[1] == order_t[1]:  # 馬単あたり
				ret_score[6] += o[9]
				ret_hitnum[6] += 1
			if (order[0] == order_t[0] or order[0] == order_t[1] or order[0] == order_t[2]) and (order[1] == order_t[0] or order[1] == order_t[1] or order[1] == order_t[2]) and (order[2] == order_t[0] or order[2] == order_t[1] or order[2] == order_t[2]):  # 三連複あたり
				ret_score[7] += o[10]
				ret_hitnum[7] += 1
			if order[0] == order_t[0] and order[1] == order_t[1] and order[2] == order_t[2]:  # 三連単あたり
				ret_score[8] += o[11]
				ret_hitnum[8] += 1
			num_retrace = num_retrace+1
			cur_pos = cur_pos+i
		
		ret_score_r = [ret_score[r] / num_retrace for r in range(9)]
		ret_score_r[1] = ret_score[1] / (num_retrace*3)
		ret_score_r[4] = ret_score[4] / (num_retrace*3)
		
		df_outfile.write('払い戻し予想： [レース数%d]\n'%len(all_races_query_test))
		df_outfile.write('\t単勝\t複勝\t複勝(1枚)\t馬連\tワイド\tワイド(1枚)\t馬単\t三連複\t三連単\n')
		df_outfile.write('オッズ：'+'\t'.join(list(map(str,ret_score_r)))+'\n')
		df_outfile.write('当り数：'+'\t'.join(list(map(str,ret_hitnum)))+'\n')
		
	if len(in_data)!=0 and len(in_meta)!=0:
		predict_validation_result = predict_validation_regression.mean(axis=1)
		order = np.argsort(predict_validation_result)
		horse = [g.split('|')[0] for g in in_data.split(',')]
		date_d = datetime.datetime.today()
		
		df_outfile.write('競馬予想 in %s\n'%date_d.strftime('%Y/%m/%d %H:%M:%S'))
		for j,i in zip(range(len(order)),order):
			df_outfile.write('%d着予想：%s\t%s\n'%(j+1,horse[i],str(predict_validation_result[i])))

if __name__ == '__main__':
	if use_algorizm=='xgb':
		 main_xgb(0)
	elif use_algorizm=='lgbm':
		 main_lgbm(0)
	elif use_algorizm=='ensemble':
		 main_xgb(0)
		 main_lgbm(10)
	main_emsemble()
