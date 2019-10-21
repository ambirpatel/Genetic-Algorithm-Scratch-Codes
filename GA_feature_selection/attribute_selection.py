import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,make_scorer
#%%
data = pd.read_csv("data.csv")
train_data = data.drop(["id","diagnosis","Unnamed: 32"],axis=1)
target = data.diagnosis
#%% Population Generation
popln = 10 ;Pc = 0.7 ; Pm = 1/popln ;bits =30

pop_binary = []
for i in range(popln):
   binr = np.random.choice([0, 1], size=bits)
   pop_binary.append(binr)

new_data = pd.DataFrame(pop_binary)
#%%
score = make_scorer(accuracy_score)
model1 = RandomForestClassifier()
column_name = train_data.columns.values
maximum_acc = []
features = []

for i in range(5):
	#============Fitness function===========#
	fitness_score = []
	for i in range(popln):
		popl_feature = []
		for j in range(bits):
			if new_data.iloc[i][j]==1:
				popl_feature.append(column_name[j])
		new_feaure_space = train_data[popl_feature]
		cv_acc = cross_val_score(estimator=model1,X=new_feaure_space,y=target,cv=5,scoring=score)
		mean_cv_acc = np.mean(cv_acc)
		c = [0.2,0.4,0.6,0.8,1]
		fittest = []
		for k in c:
			fitness = mean_cv_acc - (k*len(popl_feature))/bits
			fittest.append(fitness)
		fitness_score.append(np.max(fittest))
	max_acc = np.max(fitness_score)
	max_acc_feature = new_data.loc[np.argmax(fitness_score)]
	maximum_acc.append(max_acc)
	features.append(max_acc_feature)
	if max_acc > 0.95:
		break
	#================ tournament selection selection==========================#
	fitt = pd.DataFrame(fitness_score)
	selection_data = pd.concat([new_data,fitt],axis=1)
	selection_data = selection_data.rename({0:"fitness"},axis=1)
	select_rand = np.random.randint(0,10,size=(10,2))
	tourn_select = []
	for l in range(len(select_rand)):
		select1 = select_rand[l][0]
		select2 = select_rand[l][1]
		sel1_fit1 = selection_data.iloc[select1,30]
		sel1_fit2 = selection_data.iloc[select2,30]
		if sel1_fit1 > sel1_fit2:
			tourn_select.append(selection_data.iloc[select1])
		else:
			tourn_select.append(selection_data.iloc[select2])
	tourn_df = pd.DataFrame(np.array(tourn_select))
	tourn_df = tourn_df.copy()
	tourn_df1 = tourn_df.drop(30,axis=1).astype(int)
	#=================cross-over==============================================#
	#=====Generate random number=======#
	cross_rand = np.random.randint(0,9,size=(5,2))
	rnd_no = np.random.uniform(0,1)
	P = Pc #Crossover prob
	cross_df = []
	for m in range(int(popln/2)):
		cross1 = cross_rand[m][0]
		cross2 = cross_rand[m][1]
		cross_fit1 = tourn_df1.iloc[cross1]
		cross_fit2 = tourn_df1.iloc[cross2]
		for i in range(tourn_df1.shape[1]):
			rnd_no = np.random.uniform(0,1)
			if rnd_no < P:
				break
		cross_ex1 = cross_fit1.iloc[0:i,].append(cross_fit2.iloc[i:,])
		cross_ex2 = cross_fit2.iloc[0:i,].append(cross_fit1.iloc[i:,])
		cross_df.append(cross_ex1)
		cross_df.append(cross_ex2)
	cross_feature = pd.DataFrame(np.array(cross_df))
	cross_feature1 = cross_feature.copy()
	#================== Mutation Process=========================================#
	P = Pm #Mutation prob
	for i in range(popln):
		mute_x = cross_feature.loc[i]
		for j in range(bits):
			rnd_no = np.random.uniform(0,1)
			if rnd_no < P:
				if mute_x[j] == 0:
					mute_x[j] = 1
					break
				else:
					mute_x[j] = 0
					break
	new_data = cross_feature.copy()
feature = pd.DataFrame(features).reset_index()
#%%
accuracy = pd.DataFrame(maximum_acc).reset_index()
final_report = pd.concat([feature,accuracy],axis=1)
final_report.drop("index",axis=1,inplace=True)
all_column_name = np.append(column_name,"fitness")
final_report.columns = all_column_name
final_report.to_csv("feature_s_GA.csv",sep='\t', encoding='utf-8')
