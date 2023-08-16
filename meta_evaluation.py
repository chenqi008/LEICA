from tqdm import tqdm
import numpy as np

# ================================
# calculate accuracy of each score
# ================================
'''
path_correct = "/home/qichen/Desktop/Evaluation/MyEvaluation/run_scripts/image_gen/toy_data/coco_karpathy_split/ablation_study/origin/scores_fin_woindicator.txt"
path_incorrect = "/home/qichen/Desktop/Evaluation/MyEvaluation/run_scripts/image_gen/toy_data/coco_karpathy_split/ablation_study/mismatch/scores_fin_woindicator.txt"

scores = {}

# scores for correct images
with open(path_correct, "r") as f:
	scores["correct"] = {}
	temp = f.readlines()
	for item in temp:
		index, p, lp, lp_h = item.replace("\n", "").split("\t")
		index = int(index)
		scores["correct"][index] = {}
		scores["correct"][index]["p"] = float(p)
		scores["correct"][index]["lp"] = float(lp)
		scores["correct"][index]["lp_h"] = float(lp_h)

# scores for incorrect images
with open(path_incorrect, "r") as f:
	scores["incorrect"] = {}
	temp = f.readlines()
	for item in temp:
		index, p, lp, lp_h = item.replace("\n", "").split("\t")
		index = int(index)
		scores["incorrect"][index] = {}
		scores["incorrect"][index]["p"] = float(p)
		scores["incorrect"][index]["lp"] = float(lp)
		scores["incorrect"][index]["lp_h"] = float(lp_h)

# calculate accuracy for each score
correct_p = 0
correct_lp = 0
correct_lp_h = 0
# num_samples = len(scores["correct"])
num_samples = 30
failure_cases = [[], [], []]
for index in tqdm(range(1, num_samples+1)):
	if scores["correct"][index]["p"] > scores["incorrect"][index]["p"]:
		correct_p += 1
	else:
		failure_cases[0].append(index)
	if scores["correct"][index]["lp"] > scores["incorrect"][index]["lp"]:
		correct_lp += 1
	else:
		failure_cases[1].append(index)
	if scores["correct"][index]["lp_h"] > scores["incorrect"][index]["lp_h"]:
		correct_lp_h += 1
	else:
		failure_cases[2].append(index)

print("acc(p): {}| acc(lp): {}| acc(lp_h): {}".format(
	correct_p/num_samples, correct_lp/num_samples, correct_lp_h/num_samples))

# print(failure_cases[0])
# # print(failure_cases[1])
# print(failure_cases[2])

# with open("failure_cases.txt", "a") as f:
# 	# f.write("average p")
# 	for idx in failure_cases[0]:
# 		f.write("{}\n".format(idx))
# 	# f.write("average p_h")
# 	for idx in failure_cases[2]:
# 		f.write("{}\n".format(idx))
'''


# ===================================
# calculate Kendall Tau of each score
# ===================================

# path_better = "/home/qichen/Desktop/Evaluation/MyEvaluation/run_scripts/image_gen/toy_data/coco_karpathy_split/ablation_study/origin/scores_fin_wocredit.txt"
# path_worse = "/home/qichen/Desktop/Evaluation/MyEvaluation/run_scripts/image_gen/toy_data/coco_karpathy_split/ablation_study/gaussian_blur_8/scores_fin_wocredit.txt"

path_better = "/home/qichen/Desktop/Evaluation/MyEvaluation/run_scripts/image_gen/toy_data/coco_karpathy_split/ablation_study/origin/scores_fin_wof_woimage.txt"
path_worse = "/home/qichen/Desktop/Evaluation/MyEvaluation/run_scripts/image_gen/toy_data/coco_karpathy_split/Curve/GB1_C/scores_fin_woimagecredit.txt"

scores = {}

# scores for better images
with open(path_better, "r") as f:
	scores["better"] = {}
	temp = f.readlines()
	for item in temp:
		temp = item.replace("\n", "").split("\t")
		if len(temp)==4:
			index, p, _, _ = temp
		else:
			index, p = temp
		index = int(index)
		scores["better"][index] = {}
		scores["better"][index]["p"] = float(p)
		# scores["better"][index]["lp"] = float(lp)
		# scores["better"][index]["lp_h"] = float(lp_h)

# scores for worse images
with open(path_worse, "r") as f:
	scores["worse"] = {}
	temp = f.readlines()
	for item in temp:
		temp = item.replace("\n", "").split("\t")
		if len(temp)==4:
			index, p, _, _ = temp
		else:
			index, p = temp
		index = int(index)
		scores["worse"][index] = {}
		scores["worse"][index]["p"] = float(p)
		# scores["worse"][index]["lp"] = float(lp)
		# scores["worse"][index]["lp_h"] = float(lp_h)

# calculate kendall tau for each score
conc_p, disc_p = 0, 0
# conc_lp, disc_lp = 0, 0
# conc_lp_h, disc_lp_h = 0, 0

num_samples = len(scores["worse"])
# num_samples = 20
for index in range(1, num_samples+1):
	if scores["better"][index]["p"] > scores["worse"][index]["p"]:
		conc_p += 1
	else:
		disc_p += 1
	# if scores["better"][index]["lp"] > scores["worse"][index]["lp"]:
	# 	conc_lp += 1
	# else:
	# 	disc_lp += 1
	# if scores["better"][index]["lp_h"] > scores["worse"][index]["lp_h"]:
	# 	conc_lp_h += 1
	# else:
	# 	disc_lp_h += 1

# print("ktau(p): {}| ktau(lp): {}| ktau(lp_h): {}".format(
# 	(conc_p-disc_p)/num_samples, (conc_lp-disc_lp)/num_samples, (conc_lp_h-disc_lp_h)/num_samples))
print("ktau(p): {}".format(
	(conc_p-disc_p)/num_samples))



# ============================================
# calculate Spearman correlation of each score
# ============================================


rankpath_human = "/home/qichen/Desktop/Evaluation/human_study/rank_human_flower.txt"
# rankpath_pred = "/home/qichen/Desktop/Evaluation/FID/clip-fid/rank_CLIP_FID_cub.txt"
# rankpath_pred = "/home/qichen/Desktop/Evaluation/IS/rank_IS.txt"
# rankpath_pred = "/home/qichen/Desktop/Evaluation/SOA/rank_SOA_I.txt"
rankpath_pred = "/home/qichen/Desktop/Evaluation/GenerativeResultsFlower100/rank_ourscore.txt"

# rank (human)
with open(rankpath_human, "r") as f:
	temp = f.readlines()
	rank_human = temp[0].split("\t")[:11]
	# rank_human = temp[0].split("\t")[:5]

# rank (pred)
with open(rankpath_pred, "r") as f:
	temp = f.readlines()
	rank_pred = temp[0].split("\t")	[:11]
	# rank_pred = temp[0].replace("\n", "").split("\t")	[:5]

# # rewrite rank_human and rank_pred
# rank_human = [4, 5, 2, 1, 3]
# rank_pred = [4, 5, 3, 2, 1]

for i in range(len(rank_human)):
	rank_human[i] = str(rank_human[i])
	rank_pred[i] = str(rank_pred[i])

print(rank_human)
print(rank_pred)

# calculate spearman correlation
num_rank = len(rank_human)
diff = 0.0
for i in range(num_rank):
	# diff += (int(rank_human[i]) - int(rank_pred[i]))**2
	diff += (rank_human.index('{}'.format(i+1)) - rank_pred.index('{}'.format(i+1)))**2
spear = 1 - (6*diff)/(num_rank*(num_rank**2-1))
print("Spearman correlation: ", spear)


# ======================================
# calculate Pearson and KTau correlation 
# of each score (based on rank)
# ======================================

# ====================================================
# COCO

# a1 = [0.9741, 5.4296, 7.2370, 4.8407, 4.2629, 6.0851, 5.8111, 2.7592, 3, 7.9296, 6.6703]  # human
# a1 = [3, 2.5, 3, 2, 2.5]  # human_single_image

# a1 = [1.32, 4.37, 6.08, 3.98, 3.79, 4.71, 4.64, 2.07, 2.26, 7.34, 6.32]  #human score (absolute score)

# a2 = [2.5721, 15.5859, 24.2530, 12.8300, 15.1055, 16.8398, 24.5600, 9.5648, 8.7310, 26.8730, 21.9536]	# ours
# a2 = [15.9763, 77.6882, 111.9723, 64.6733, 68.2254, 84.8881, 117.2855, 50.5783, 45.8134, 123.3857, 103.4911]  # ours (without patch-wise)
# a2 = [14.6403, 90.1502, 126.6252, 57.6403, 74.1246, 82.7093, 137.2757, 55.6180, 43.9798, 136.4796, 110.2941]  # ours (without patch-wise, 30 samples)
# a2 = [0.0013, 0.0093, 0.0116, 0.0061, 0.0092, 0.0103, 0.0136, 0.0043, 0.0051, 0.0114, 0.0121]  # ours (without indicator)
# a2 = [0.1774, 0.2345, 0.2599, 0.2298, 0.2420, 0.2397, 0.2451, 0.2160, 0.2161, 0.2609, 0.2562]  # ours (without image-wise & f)
# a2 = [0.1697, 0.2303, 0.2532, 0.2244, 0.2393, 0.2359, 0.2411, 0.2153, 0.2116, 0.2634, 0.2486]  # ours (without image-wise & f, 30 samples)
# a2 = [0.1690, 0.2303, 0.2532, 0.2240, 0.2393, 0.2353, 0.2411, 0.2147, 0.2106, 0.2614, 0.2386]  # ours (without image-wise & f)

# a2 = [97.27, 38.0735, 56.2331, 52.2095, 50.6834, 45.2642, 56.4722, 49.3247, 49.0368, 27.4043, 52.3333]	# FID
# a2 = [0.0368, 0.0112, 0.0154, 0.0176, 0.0212, 0.0143, 0.0253, 0.0005, 0.0222, 0.0072, 0.0224]	# KID
# a2 = [11.9614, 21.9238, 24.6718, 22.4698, 26.9268, 18.9741, 21.6151, 22.8696, 18.8433, 34.3436, 22.5687] # IS
# a2 = [0.0109, 0.3044, 0.408, 0.2522, 0.2738, 0.2279, 0.2313, 0.2073, 0.1747, 0.4256, 0.2584] # SOA-C
# a2 = [0.0597, 0.3390, 0.4146, 0.2849, 0.2041, 0.2958, 0.3084, 0.2537, 0.2393, 0.4540, 0.2804] # SOA-I

# a2 = [44.9171, 16.6546, 16.8674, 16.3108, 18.2699, 17.9985, 21.9243, 24.9345, 18.9532, 14.2830, 21.1876]   # CLIP-FID
# a2 = [3.3606, 21.7823, 29.1341, 15.6197, 19.2801, 20.9263, 28.6608, 11.4587, 11.2684, 32.3540, 26.9476]  # ours-base
# a2 = [3.3606, 21.7823, 29.1341, 15.6197, 19.2801, 20.9263, 28.6608, 15.4587, 11.2684, 32.3540, 22.9476]  # ours-base-new

# a2 = [35.4565, 34.3756, 30.5915, 38.1478, 31.7543]  # ours_single_image
# a2 = [3, 3, 3, 3, 3]   # another human single image


# # ====================================================
# CUB
# a1 = [2.31, 2.36, 1.99, 4.97, 3.37] # human

# # a2 = [35.7255, 37.4008, 39.6864, 43.3752, 39.8252]  # ours
# # a2 = [43.7391, 57.8065, 91.8937, 30.8861, 57.2698]  # FID
# # a2 = [0.02711, 0.03453, 0.07441, 0.01191, 0.04217]  # KID
# a2= [4.5531, 2.8777, 2.4393, 3.5921, 3.8937]  # IS
# a2= [13.2552, 22.3915, 27.3548, 13.6101, 21.8422]  # CLIP-FID

# # ====================================================

# ====================================================
# Flower
a1 = [3.63, 3.59, 2.31, 2.87, 2.6] # human score
# a1 = [2.37, 2.41, 3.69, 3.13, 3.4] # human rank

a2 = [30.4031, 29.1366, 29.1316, 24.4925, 24.5621]  # ours
# a2 = [52.3837, 49.5453, 51.2440, 42.0608, 42.6935]  # ours (1/8192/100)
# a2 = [28.0765, 25.9954, 28.7493, 24.6587, 23.6573]  # ours 30
# a2 = [75.5599, 79.0139, 79.094, 81.4431, 78.4558]  # FID
# a2 = [2.7051, 2.4987, 2.1957, 2.8542, 2.7443]  # IS
# a2 = [0.0508, 0.0574, 0.0508, 0.0333, 0.0496]  # KID
# a2= [15.5964, 19.4723, 21.6562, 17.2339, 19.3545]  # CLIP-FID

# ====================================================


a1_np = np.array(a1)
a2_np = np.array(a2)

# if a2 is lower is better, use it
# a2_np =  100 - a2_np # FID
# a2_np =  1 - a2_np # KID

# Pearson
correlation = np.corrcoef(a1_np, a2_np)
# print(correlation)
print("pearson: ", correlation[0, 1])

# K-Tau, A(2,11)
pairs_a1 = []
pairs_a2 = []
for i in range(10):
	for j in range(i+1, 11):
# for i in range(4):
# 	for j in range(i+1, 5):
		pairs_a1.append((a1_np[i], a1_np[j]))
		pairs_a2.append((a2_np[i], a2_np[j]))

conc = 0.0
disc = 0.0
for p in range(len(pairs_a1)):
	if (pairs_a1[p][0]>=pairs_a1[p][1]) and (pairs_a2[p][0]>=pairs_a2[p][1]):
		conc += 1
	elif (pairs_a1[p][0]<=pairs_a1[p][1]) and (pairs_a2[p][0]<=pairs_a2[p][1]):
		conc += 1
	else:
		disc += 1

print("conc:", conc, "disc:", disc, "K-Tau: ", (conc-disc)/(conc+disc))
