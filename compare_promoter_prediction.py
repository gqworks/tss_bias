import pandas as pd
import re, os
import numpy as np
from Bio import SeqIO
import matplotlib.pyplot as plt
import pyBigWig
import seaborn as sns
import statsmodels.stats.multitest as multi

def get_gc_null(tss_dat_np,fasta_sequences):
    null_gc = []
    for chrom in list(set(tss_dat_np[:,0])):
        subset = tss_dat_np[np.in1d(tss_dat_np[:,0],chrom)]
        random_tss = randomize(subset[:,6])
        strand_perc = sum(subset[:,4]=='+')/len(subset)
        genome_len = len(fasta_sequences[chrom].seq)
        for i in random_tss:
            if i <= 501 or i >= genome_len-500:
                continue
            left_perc = 100*len(re.findall("G|C",str(fasta_sequences[chrom].seq[i-500:i])))/500
            right_perc = 100*len(re.findall("G|C",str(fasta_sequences[chrom].seq[i:i+500])))/500
            if np.random.binomial(1,strand_perc)==1: 
                null_gc.append(right_perc-left_perc)
            else:
                null_gc.append(left_perc-right_perc)
    return(np.mean(null_gc))

def randomize(intervals):
    intervals = intervals.astype(int)
    point_diffs = intervals[1:] - intervals[:-1]
    max_len = intervals[-1]
    new_points = np.concatenate((np.array(point_diffs).cumsum(),[max_len]))
    new_points = (new_points + np.random.randint(0,max_len))%max_len
    return new_points

def scale_mat(d):
    return (d-d.mean(axis=1)[:,None]) / d.std(axis=1)[:,None]

def calc_bias(x):
    left_sum = np.sum(x[:500])
    right_sum = np.sum(x[500:])
    return(right_sum-left_sum)

def get_fixed_null(depth,tss_dat_np):
    genome_dict = depth.chroms()
    null_mat = []
    for chrom in list(set(tss_dat_np[:,0])):
        subset = tss_dat_np[np.in1d(tss_dat_np[:,0],chrom)]
        random_tss = randomize(subset[:,6])
        strand_perc = sum(subset[:,4]=='+')/len(subset)
        for i in random_tss:
            if i <= 500 or i >= genome_dict[chrom]-500:
                continue
            window = depth.values(chrom,i-501,i+500)
            if np.random.binomial(1, strand_perc) == 0: window.reverse()
            null_mat.append(window)
    return(pd.DataFrame(null_mat))

def get_tpm(file):
    dat = pd.read_table(file)
    dat = dat.drop_duplicates(subset='target_id')
    dat = dat[['target_id','tpm']]
    dat.index = dat['target_id']
    dat = dat.drop(columns = ['target_id'])
    return dat


###Initialise Data
with open('Franzosa_8_samples') as f:
    Franzosa_8 = f.read().splitlines()
Franzosa_8 = {x.split('\t')[0]:x.split('\t')[1] for x in Franzosa_8}

# Test of F. prausnitzii GCF_902388275.1_UHGG_MGYG-HGUT-02545
asm = 'GCF_902388275.1_UHGG_MGYG-HGUT-02545'

### Prepare predicted promoters ###
tss_dat = pd.read_csv('src/promotech_genome_predictions.csv', sep = '\t')
tss_dat = tss_dat[tss_dat.score>=0.8]
#tss_dat = pd.read_csv('src/ProPr.Promoters.gff', sep = '\t')
#tss_dat = tss_dat.drop(['db', 'type','name'], axis=1)
tss_dat['tss'] = [x['end'] if x['strand'] == '+' else x['start'] for i, x in tss_dat.iterrows()]
tss_dat_np = tss_dat.to_numpy()
#Remove overlaping TSS -TSS doesnt overlap with TSS
non_overlap = []
#First case
if float(tss_dat_np[0,6])+500 < float(tss_dat_np[1,6]):
    non_overlap.append(tss_dat_np[0])
#Mid case
for i in range(1,len(tss_dat_np)-1):
    if float(tss_dat_np[i,6])-500 > float(tss_dat_np[i-1,6]) and float(tss_dat_np[i,6])+500 < float(tss_dat_np[i+1,6]):
        non_overlap.append(tss_dat_np[i])
#Last case
if float(tss_dat_np[len(tss_dat_np)-1,6])-500 > float(tss_dat_np[len(tss_dat_np)-2,6]):
    non_overlap.append(tss_dat_np[len(tss_dat_np)-1])
non_overlap = pd.DataFrame(non_overlap, columns = ["Genome","left","right","score",'Strand',"desc","Start"])
#non_overlap = pd.DataFrame(non_overlap, columns = ["Genome","left","right","Strand",'score',"desc","Start"])
tss_dat_np = non_overlap.to_numpy()

### GC content Null distribution ###
fna = f'strains/{asm}/{asm}.fna'
fasta_sequences = SeqIO.to_dict(SeqIO.parse(fna, "fasta"))
gc_bias = []
for i in tss_dat_np:
    genome_len = len(fasta_sequences[i[0]].seq)
    start = i[6]
    strand = i[4]
    if start <= 501 or start >= genome_len-500:
        continue
    left_perc = 100*len(re.findall("G|C",str(fasta_sequences[i[0]].seq[start-500:start])))/500
    right_perc = 100*len(re.findall("G|C",str(fasta_sequences[i[0]].seq[start:start+500])))/500
    if strand == '+': 
        gc_bias.append(right_perc-left_perc)
    else:
        gc_bias.append(left_perc-right_perc)
real_bias = np.mean(gc_bias)
null_bias = []
for i in range(1000):
    null_bias.append(get_gc_null(tss_dat_np,fasta_sequences))
pval = (sum(null_bias > real_bias)+1)/len(null_bias)
plt.suptitle('GC null distribution')
plt.hist(null_bias, bins=50)
plt.axvline(x=real_bias, color='r')
plt.text(1, 10, f'p={pval}', size=12, color='black',weight='bold')
plt.savefig("plots/promoter_prediction/gc_distribution.pdf")
plt.clf()


###################################
####### REAL READ COVERAGE ########
###################################
all_pvals = []
for sample in Franzosa_8.keys():
    # Calculate matrix
    bp_depth = pyBigWig.open(f'beds/{asm}_{sample}.bw')
    genome_dict = bp_depth.chroms()

    mat = []
    for row in tss_dat_np:
        chrom = row[0]
        if (row[6] <= 550) or (row[6] + 550 >= genome_dict[chrom]):
            window = [0]*1001
        else:
            window = bp_depth.values(chrom,int(row[6])-501,int(row[6])+500)
            if row[4]=="-": window.reverse()
        mat.append(window)
    mat = pd.DataFrame(mat)

    #Calculate mean sample TSS bias
    scaled_mat = scale_mat(mat)
    scaled_mat = scaled_mat.fillna(0)
    real_bias = list(map(calc_bias, np.array(scaled_mat)))

    #Calculate null TSS bias distribution
    null_bias = []
    for i in range(500):
        null_mat = get_fixed_null(bp_depth,tss_dat_np)
        scaled_null = scale_mat(null_mat)
        scaled_null = scaled_null.fillna(0)
        null_bias.append(np.mean(list(map(calc_bias, np.array(scaled_null)))))
    pval = (sum(null_bias > np.mean(real_bias))+1)/len(null_bias)
    plt.suptitle(f'{sample} {asm}')
    plt.hist(null_bias, bins=50)
    plt.axvline(x=np.mean(real_bias), color='r')
    plt.text(-5, 3, f'p={pval}', size=12, color='white',weight='bold')
    plt.savefig(f"plots/promoter_prediction/{sample}_null_distribution.pdf")
    plt.clf()

    all_pvals.append([sample,pval])
        
all_pvals = pd.DataFrame(all_pvals)
all_pvals = all_pvals.rename(columns={0: "sample", 1: "pval"})
all_pvals['adj_pval'] = multi.multipletests(all_pvals['pval'], method = 'fdr_bh', alpha = 0.05)[1]
all_pvals['neg_log'] = -np.log(all_pvals['adj_pval'])
all_pvals.to_csv("plots/promoter_prediction/sample_pvals.csv")


# Compare difference of predicted promoters to prodigal TSSs
prodigal_tss = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
differences = []
for pos in tss_dat_np[:,6]:
    closest = min(prodigal_tss.Start, key=lambda x:abs(x-pos))
    diff = abs(closest-pos)
    differences.append(diff)
plt.hist(differences, bins=50, range = (0,1000))
plt.xlabel("Distance to closest TSS (bp)")
plt.ylabel("No. promoters")
plt.xticks(np.arange(0, 1000, 100))
plt.savefig(f"plots/promoter_prediction/distance_from_prodigal_positions.pdf")
plt.clf()