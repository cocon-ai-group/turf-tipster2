# -*- coding: utf-8 -*-
import codecs
import re
import urllib.parse
import urllib.request
from urllib.error import HTTPError
import os
import socket
import sys
import math
import calendar
import datetime
import numpy as np
from bs4 import BeautifulSoup
import csv
import random
import time
from multiprocessing import Pool
from multiprocessing import Process

scrap_year = [2016,2017,2018]
num_process = 4
url_base_netkeiba = 'http://db.netkeiba.com'
url_base_keibalab = 'https://www.keibalab.jp'

with open('horse_id.txt') as idfile:
	horse_ids = idfile.readlines()
	horse_urls = [u.strip() for u in horse_ids]
	horse_urls = list(set(horse_urls))


def get_meta_info(meta):
	s = eval('b"{}"'.format(str(meta))).decode('EUC_JP')
	wp = 5
	if s.find('小倉') >= 0:
		wp = 0
	elif s.find('阪神') >= 0:
		wp = 1
	elif s.find('京都') >= 0:
		wp = 2
	elif s.find('中京') >= 0:
		wp = 3
	elif s.find('中山') >= 0:
		wp = 4
	elif s.find('東京') >= 0:
		wp = 5
	elif s.find('新潟') >= 0:
		wp = 6
	elif s.find('福島') >= 0:
		wp = 7
	elif s.find('函館') >= 0:
		wp = 8
	elif s.find('札幌') >= 0:
		wp = 9
	elif s.find('大井') >= 0:
		wp = 10
	elif s.find('園田') >= 0:
		wp = 11
	elif s.find('門別') >= 0:
		wp = 12
	elif s.find('水沢') >= 0:
		wp = 13
	elif s.find('盛岡') >= 0:
		wp = 14
	elif s.find('浦和') >= 0:
		wp = 15
	bp = 1
	if s.find('芝') >= 0:
		bp = 0
	elif s.find('障') >= 0:
		bp = 2
	ep = 0
	if s.find('晴') >= 0:
		ep = 0
	elif s.find('曇') >= 0:
		ep = 1
	elif s.find('小雨') >= 0:
		ep = 4
	elif s.find('小雪') >= 0:
		ep = 5
	elif s.find('雨') >= 0:
		ep = 2
	elif s.find('雪') >= 0:
		ep = 3
	lp = 0
	met = re.findall(\
		r'([0-9][0-9][0-9][0-9])',\
		s, re.DOTALL)
	if len(met) > 0:
		lep = int(met[0])
		if lep < 1400:
			lp = 1
		elif lep < 1800:
			lp = 2
		elif lep < 2200:
			lp = 3
		elif lep < 2600:
			lp = 4
		else:
			lp = 5
	bb = 0
	if s.find('良') >= 0:
		bb = 0
	elif s.find('重') >= 0:
		bb = 1
	elif s.find('不') >= 0:
		bb = 2
	elif s.find('稍') >= 0:
		bb = 3
	return wp, bp, ep, lp, bb

def getBloodCol(colname):
	if colname == 'blood_11300497':
		return 1
	if colname == 'blood_11300917':
		return 2
	if colname == 'blood_11300980':
		return 3
	if colname == 'blood_11300493':
		return 4
	if colname == 'blood_11300494':
		return 5
	if colname == 'blood_11301018':
		return 6
	if colname == 'blood_11300912':
		return 7
	if colname == 'blood_11301098':
		return 8
	return 0
	
def isDigit(x):
	try:
		float(x)
		return True
	except TypeError:
		return False
	except ValueError:
		return False

def scrap_one_horse(url):
	time.sleep(0.5 + random.random())
	result_tbl = []
	horse_meta = []
	horse_name = ''
	if not url:
		return None
	try:
		with urllib.request.urlopen(url_base_keibalab+'/db/horse/'+url+'/') as response:
			# URLから読み込む
			html = str(response.read())
			trainer = re.findall(\
				r'<a href=\"/db/trainer/([0-9][0-9][0-9][0-9][0-9])/\"',\
				html, re.DOTALL)
			if len(trainer) > 0:
				horse_meta.append(int(trainer[0]))
			else:
				horse_meta.append(0)
			bl1 = html.find('<!-- \\xe8\\xa1\\x80\\xe7\\xb5\\xb1\\xe8\\xa1\\xa8 -->')
			bl2 = html.rfind('<!-- \\xe8\\xa1\\x80\\xe7\\xb5\\xb1\\xe8\\xa1\\xa8 -->')
			if bl1 > 0 and bl2 > 0:
				bl = re.findall(\
					r'<td rowspan="4" class="std2 bg([A-Z])  (blood_[0-9]*)"',\
					html[bl1:bl2], re.DOTALL)
				for b, bd in bl:
					horse_meta.append(getBloodCol(bd))
				bl = re.findall(\
					r'<td rowspan="2" class="std2 bg([A-Z])  (blood_[0-9]*)"',\
					html[bl1:bl2], re.DOTALL)
				for b, bd in bl:
					horse_meta.append(getBloodCol(bd))
				bl = re.findall(\
					r'<td class="std2 bg([A-Z])  (blood_[0-9]*)"',\
					html[bl1:bl2], re.DOTALL)
				for b, bd in bl:
					horse_meta.append(getBloodCol(bd))
			soup = BeautifulSoup(html, 'html.parser')
			meta = soup.find("meta", attrs={"name":"keywords"})
			if meta:
				keywords = eval('b"{}"'.format(meta['content'])).decode('UTF8')
				skey = keywords.split(',')
				if len(skey) > 0 and len(skey[0]) > 0:
					horse_name = skey[0]
		
		with urllib.request.urlopen(url_base_netkeiba+'/horse/'+url+'/') as response:
			# URLから読み込む
			html = str(response.read())
			soup = BeautifulSoup(html, 'html.parser')
			tbl = soup.find_all('div', 'cate_bar')
			if len(tbl) == 1:
				ttl = tbl[0].findNextSibling()
				for yy in scrap_year:
					for mm in range(1,13):
						yymm = '%04d/%02d/01'%((yy),mm)
						m_rank = []
						m_rank_h = []
						m_rank_h_d = []
						m_rank_h_s = []
						m_rank_s = []
						m_fav = []
						m_odds = []
						m_weight = []
						m_kin = []
						m_delta = []
						m_pacea = []
						m_paceb = []
						m_passa = []
						m_passb = []
						m_threef = []
						m_award = []
						r_where = np.zeros( (16, 4) ).tolist()
						r_baba = np.zeros( (3, 4) ).tolist()
						r_weather = np.zeros( (6, 4) ).tolist()
						r_len = np.zeros( (6, 4) ).tolist()
						r_omosa = np.zeros( (4, 4) ).tolist()
						p_where = p_baba = p_weather = p_len = p_omosa = 0
						to_add = False
						for tr in ttl.find_all("tr"):
							td_s = [i for i in tr.find_all("td")]
							if len(td_s) == 28:
								for i, td in zip(range(28), td_s):
									if i % 28 == 0:
										if td.text < yymm:
											to_add = True
									if i % 28 == 1:
										if to_add:
											meta_s = td.text + td_s[2].text + td_s[14].text + td_s[15].text
											p_where, p_baba, p_weather, p_len, p_omosa = get_meta_info(meta_s)
									if i % 28 == 11:
										if to_add and td.text.isdigit():
											rnk = int(td.text)
											m_rank.append(rnk)
											if p_baba != 2:
												m_rank_h.append(rnk)
												if p_baba == 0:
													m_rank_h_s.append(rnk)
												else:
													m_rank_h_d.append(rnk)
											else:
												m_rank_s.append(rnk)
											if rnk <= 0:
												rnk = 1
											if rnk > 4:
												rnk = 4
											r_where[p_where][rnk-1] += 1
											r_baba[p_baba][rnk-1] += 1
											r_weather[p_weather][rnk-1] += 1
											r_len[p_len][rnk-1] += 1
											r_omosa[p_omosa][rnk-1] += 1
									if i % 28 == 10:
										if to_add and td.text.isdigit():
											m_fav.append(int(td.text))
									if i % 28 == 9:
										if to_add and isDigit(td.text):
											m_odds.append(float(td.text))
									if i % 28 == 23:
										if to_add and td.text[:3].isdigit():
											m_weight.append(int(td.text[:3]))
									if i % 28 == 18:
										if to_add and isDigit(td.text):
											m_delta.append(float(td.text))
									if i % 28 == 22:
										if to_add and isDigit(td.text):
											m_threef.append(float(td.text))
									if i % 28 == 21:
										if to_add:
											pace = td.text.split('-')
											if len(pace) == 2 and isDigit(pace[0]) and isDigit(pace[1]):
												m_pacea.append(float(pace[0]))
												m_paceb.append(float(pace[1]))
									if i % 28 == 20:
										if to_add:
											pss = td.text.split('-')
											if len(pss) >= 2 and pss[0].isdigit() and pss[1].isdigit():
												m_passa.append(int(pss[0]))
												m_passb.append(int(pss[1]))
									if i % 28 == 13:
										if to_add and td.text.isdigit():
											m_kin.append(int(td.text))
									if i % 28 == 27:
										if to_add:
											if isDigit(td):
												m_award.append(float(td.text))
											else:
												m_award.append(0.0)
							
						hist_meta = []
						hist_meta.append(len(m_rank))
						hist_meta.append(np.mean(m_rank) if len(m_rank) > 0 else 0)
						hist_meta.append(np.mean(m_rank_h) if len(m_rank_h) > 0 else 0)
						hist_meta.append(np.mean(m_rank_h_d) if len(m_rank_h_d) > 0 else 0)
						hist_meta.append(np.mean(m_rank_h_s) if len(m_rank_h_s) > 0 else 0)
						hist_meta.append(np.mean(m_rank_s) if len(m_rank_s) > 0 else 0)
						hist_meta.append(np.mean(m_fav) if len(m_fav) > 0 else 0)
						hist_meta.append(np.mean(m_odds) if len(m_odds) > 0 else 0)
						hist_meta.append(np.mean(m_weight) if len(m_weight) > 0 else 0)
						hist_meta.append(np.mean(m_kin) if len(m_kin) > 0 else 0)
						hist_meta.append(np.mean(m_delta) if len(m_delta) > 0 else 0)
						hist_meta.append(np.mean(m_pacea) if len(m_pacea) > 0 else 0)
						hist_meta.append(np.mean(m_paceb) if len(m_paceb) > 0 else 0)
						hist_meta.append(np.mean(m_passa) if len(m_passa) > 0 else 0)
						hist_meta.append(np.mean(m_passb) if len(m_passb) > 0 else 0)
						hist_meta.append(np.mean(m_threef) if len(m_threef) > 0 else 0)
						hist_meta.append(np.mean(m_award) if len(m_award) > 0 else 0)
						hist_meta.append(np.mean(m_rank[:3]) if len(m_rank) >= 3 else 0)
						hist_meta.append(np.mean(m_rank_h[:3]) if len(m_rank_h) >= 3 else 0)
						hist_meta.append(np.mean(m_rank_h_d[:3]) if len(m_rank_h_d) >= 3 else 0)
						hist_meta.append(np.mean(m_rank_h_s[:3]) if len(m_rank_h_s) >= 3 else 0)
						hist_meta.append(np.mean(m_rank_s[:3]) if len(m_rank_s) >= 3 else 0)
						hist_meta.append(np.mean(m_fav[:3]) if len(m_fav) >= 3 else 0)
						hist_meta.append(np.mean(m_odds[:3]) if len(m_odds) >= 3 else 0)
						hist_meta.append(np.mean(m_weight[:3]) if len(m_weight) >= 3 else 0)
						hist_meta.append(np.mean(m_kin[:3]) if len(m_kin) >= 3 else 0)
						hist_meta.append(np.mean(m_delta[:3]) if len(m_delta) >= 3 else 0)
						hist_meta.append(np.mean(m_pacea[:3]) if len(m_pacea) >= 3 else 0)
						hist_meta.append(np.mean(m_paceb[:3]) if len(m_paceb) >= 3 else 0)
						hist_meta.append(np.mean(m_passa[:3]) if len(m_passa) >= 3 else 0)
						hist_meta.append(np.mean(m_passb[:3]) if len(m_passb) >= 3 else 0)
						hist_meta.append(np.mean(m_threef[:3]) if len(m_threef) >= 3 else 0)
						hist_meta.append(np.mean(m_award[:3]) if len(m_award) >= 3 else 0)
						
						each_rank = sum(r_where, []) + sum(r_baba, []) + sum(r_weather, []) + sum(r_len, []) + sum(r_omosa, [])
						
						rank_t = [0,0,0,0]
						for i in range(len(m_rank)):
							rnk = m_rank[i]
							if rnk > 0:
								if rnk > 4:
									rnk = 4
								rank_t[rnk-1] += 1
						rank_h = [0,0,0,0]
						for i in range(len(m_rank_h)):
							rnk = m_rank_h[i]
							if rnk > 0:
								if rnk > 4:
									rnk = 4
								rank_h[rnk-1] += 1
						rank_h_s = [0,0,0,0]
						for i in range(len(m_rank_h_s)):
							rnk = m_rank_h_s[i]
							if rnk > 0:
								if rnk > 4:
									rnk = 4
								rank_h_s[rnk-1] += 1
						rank_h_d = [0,0,0,0]
						for i in range(len(m_rank_h_d)):
							rnk = m_rank_h_d[i]
							if rnk > 0:
								if rnk > 4:
									rnk = 4
								rank_h_d[rnk-1] += 1
						rank_s = [0,0,0,0]
						for i in range(len(m_rank_s)):
							rnk = m_rank_s[i]
							if rnk > 0:
								if rnk > 4:
									rnk = 4
								rank_s[rnk-1] += 1
						rank_val = rank_t + rank_h + rank_h_s + rank_h_d + rank_s
						
						if len(m_rank) > 0:
							rank_val.append(rank_t.count(1) / len(m_rank))
							rank_val.append((rank_t.count(1)+rank_t.count(2)+rank_t.count(3)) / len(m_rank))
							rank_val.append(rank_h.count(1) / len(m_rank))
							rank_val.append((rank_h.count(1)+rank_h.count(2)+rank_h.count(3)) / len(m_rank))
							rank_val.append(m_rank_h_s.count(1) / len(m_rank))
							rank_val.append((m_rank_h_s.count(1)+m_rank_h_s.count(2)+m_rank_h_s.count(3)) / len(m_rank))
							rank_val.append(rank_h_d.count(1) / len(m_rank))
							rank_val.append((rank_h_d.count(1)+rank_h_d.count(2)+rank_h_d.count(3)) / len(m_rank))
							rank_val.append(rank_s.count(1) / len(m_rank))
							rank_val.append((rank_s.count(1)+rank_s.count(2)+rank_s.count(3)) / len(m_rank))
						else:
							rank_val.extend([0,0,0,0,0,0,0,0,0,0])
						
						meta = horse_meta + hist_meta + each_rank + rank_val
						# データを追加
						if horse_name!= '' and len(meta) == 218:
							ikey = '%s%02d%02d'%(horse_name,(yy-2000),mm)
							result_tbl.append( (ikey, meta) )
						else:
							print('Err:'+horse_name+' '+url)
							print(len(horse_meta),len(hist_meta),len(each_rank),len(rank_val),len(meta))
	except HTTPError as e:
		print(url)
		print(e)
		pass
	except SyntaxError as e:
		print(url)
		print(e)
		pass
	return result_tbl

proc_pool = Pool(num_process)
result = proc_pool.map(scrap_one_horse, horse_urls)
with open('horse_history.csv', 'w') as wf:
	for tbl in result:
		if tbl:
			for ikey, meta in tbl:
				wf.write('%s,%s\n'%(ikey,','.join(list(map(str,meta)))))



