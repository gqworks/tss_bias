import re, os
import pandas as pd
import seaborn as sns
import numpy as np
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
import itertools
import pyBigWig

def get_cor_mat(asm):
    hmp2_samples_bias = np.array([get_bias_from_mat(asm,samp) for samp in hmp2_samples])
    Franzosa_samples_bias = np.array([get_bias_from_mat(asm,samp) for samp in Franzosa_samples])
    liu2016_bias = np.array([get_bias_from_mat(asm,samp) for samp in liu2016_samples])
    hmp3_stool_bias = np.array([get_bias_from_mat(asm,samp) for samp in hmp3_stool_samples])
    
    hmp2_samples_bias = hmp2_samples_bias[[False if len(x)==1 else True for x in hmp2_samples_bias]]
    Franzosa_samples_bias = Franzosa_samples_bias[[False if len(x)==1 else True for x in Franzosa_samples_bias]]
    liu2016_bias = liu2016_bias[[False if len(x)==1 else True for x in liu2016_bias]]
    hmp3_stool_bias = hmp3_stool_bias[[False if len(x)==1 else True for x in hmp3_stool_bias]]
    all_bias_groups = [hmp2_samples_bias,Franzosa_samples_bias,liu2016_bias,hmp3_stool_bias]
    cor_vals = []
    p_vals = []
    for x in all_bias_groups:
        for y in all_bias_groups:
            avg_r, avg_p = get_avg_r_p(x, y)
            cor_vals.append(avg_r)
            p_vals.append(avg_p)
    cor_mat = np.array(cor_vals).reshape(4,4)
    p_mat = np.array(p_vals).reshape(4,4)
    return cor_mat, p_mat

def get_avg_r_p(a,b):
    r_vals = []
    p_vals = []
    for r in itertools.product(a, b):
        nan = np.isnan(r[0]) | np.isnan(r[1])
        clean_x = r[0][~nan]
        clean_y = r[1][~nan]
        if len(clean_x) < 2 or np.array_equal(clean_x, clean_y):
            continue
        corr_r, corr_p = stats.pearsonr(clean_x, clean_y)
        r_vals.append(corr_r)
        p_vals.append(corr_p)
    return np.mean(r_vals), np.mean(p_vals)

def get_bias_from_mat(asm,sample):
    if not os.path.exists(f'matrices/{asm}_{sample}.csv'):
        return(np.array([0]))
    mat = pd.read_table(f'matrices/{asm}_{sample}.csv', sep = ',', header=None, index_col=0)
    bias = np.array(list(map(calc_bias, np.array(scale_mat(mat)))))
    #bias[np.isnan(bias)] = 0
    return bias

def scale_mat(d):
    return (d-d.mean(axis=1)[:,None]) / d.std(axis=1)[:,None]

def calc_bias(x):
    left_sum = np.sum(x[:500])
    right_sum = np.sum(x[500:])
    return(right_sum-left_sum)

def randomize(intervals):
    intervals = intervals.astype(int)
    point_diffs = intervals[1:] - intervals[:-1]
    max_len = intervals[-1]
    new_points = np.concatenate((np.array(point_diffs).cumsum(),[max_len]))
    new_points = (new_points + np.random.randint(0,max_len))%max_len
    return new_points

def get_null_bias_from_mat(asm,samp,annot):
    if not os.path.exists(f'beds/{asm}_{samp}.bw'):
        return(np.array([0]))
    depth = pyBigWig.open(f'beds/{asm}_{samp}.bw')
    genome_dict = depth.chroms()
    null_mat = []
    for chrom in list(set(annot.Genome)):
        if genome_dict[chrom] < 1000:
            continue
        random_tss = randomize(np.array(annot[annot.Genome==chrom]['Start']))
        strand_perc = sum((annot.Genome==chrom)&(annot.Strand=='+'))/sum(annot.Genome==chrom)
        for i in random_tss:
            if i <= 500:
                i = 501
            if i >= genome_dict[chrom]-500:
                i = genome_dict[chrom]-501
            window = depth.values(chrom,i-501,i+500)
            if np.random.binomial(1, strand_perc) == 0: window.reverse()
            null_mat.append(window)
    null_mat = np.array(null_mat)
    bias = np.array(list(map(calc_bias, np.array(scale_mat(null_mat)))))
    return bias

def get_null_mat(asm):
    tss_dat = pd.read_csv(f'refseq_genomes/{asm}/{asm}_filtered_tss.csv')
    hmp2_samples_bias = np.array([get_null_bias_from_mat(asm,samp,tss_dat) for samp in hmp2_samples])
    Franzosa_samples_bias = np.array([get_null_bias_from_mat(asm,samp,tss_dat) for samp in Franzosa_samples])
    liu2016_bias = np.array([get_null_bias_from_mat(asm,samp,tss_dat) for samp in liu2016_samples])
    hmp3_stool_bias = np.array([get_null_bias_from_mat(asm,samp,tss_dat) for samp in hmp3_stool_samples])
    
    hmp2_samples_bias = hmp2_samples_bias[[False if len(x)==1 else True for x in hmp2_samples_bias]]
    Franzosa_samples_bias = Franzosa_samples_bias[[False if len(x)==1 else True for x in Franzosa_samples_bias]]
    liu2016_bias = liu2016_bias[[False if len(x)==1 else True for x in liu2016_bias]]
    hmp3_stool_bias = hmp3_stool_bias[[False if len(x)==1 else True for x in hmp3_stool_bias]]
    null_bias_groups = [hmp2_samples_bias,Franzosa_samples_bias,liu2016_bias,hmp3_stool_bias]
    cor_vals = []
    p_vals = []
    for x in null_bias_groups:
        for y in null_bias_groups:
            avg_r, avg_p = get_avg_r_p(x, y)
            cor_vals.append(avg_r)
            p_vals.append(avg_p)
    cor_mat = np.array(cor_vals).reshape(4,4)
    p_mat = np.array(p_vals).reshape(4,4)
    return cor_mat, p_mat



###Initialise Data
with open('sample_lists/hmp2_samples') as f:
    hmp2_samples = f.read().splitlines()
hmp2_samples = {x.split('\t')[0]:x.split('\t')[1] for x in hmp2_samples}

with open('sample_lists/Franzosa_8_samples') as f:
    Franzosa_samples = f.read().splitlines()
#Franzosa_samples = [x.rsplit('\t',1)[0] for x in Franzosa_samples]
Franzosa_samples = [x.rsplit('/',1)[1] for x in Franzosa_samples]

# Liu et al. 2016. sample_id values are SRA run accessions.
liu2016_samples = pd.read_csv('data/manifests/liu2016.tsv', sep='\t')['sample_id'].tolist()

# Public HMP Phase III stool specimens. sample_id values are SRA run accessions.
hmp3_stool_samples = pd.read_csv('data/manifests/hmp3_stool.tsv', sep='\t')['sample_id'].tolist()

#Read species abundances
mpa_res = pd.read_table('mpa4_res/merged_mpa.txt', comment= '#')
mpa_res = mpa_res[(mpa_res['clade_name'].str.contains("s__"))&(~mpa_res['clade_name'].str.contains("t__"))]
mpa_res.index = [re.match('.*\|([^\|]*)',x).group(1) for x in mpa_res['clade_name']]
mpa_res = mpa_res.drop(columns=['clade_name','NCBI_tax_id'])
# Filter species with atleast 0.5% abundance in atleast 5 samples in each dataset
mpa_res = mpa_res.loc[((mpa_res[Franzosa_samples] > 0.5).sum(axis=1) >= 5) & 
                      ((mpa_res[hmp2_samples] > 0.5).sum(axis=1) >= 30) &
                      ((mpa_res[liu2016_samples] > 0.5).sum(axis=1) >= 5) & 
                      ((mpa_res[hmp3_stool_samples] > 0.5).sum(axis=1) >= 5),:]
mpa_res = mpa_res.where(mpa_res > 0.5, np.nan)

all_genomes = {'s__Phocaeicola_vulgatus': 'GCF_000012825.1_ASM1282v1',
 's__Agathobacter_rectalis': 'GCF_022453685.1_ASM2245368v1',
 's__Alistipes_putredinis': 'GCF_000154465.1_ASM15446v1',
 's__Faecalibacterium_prausnitzii': 'GCF_000154385.1_ASM15438v1',
 's__Bacteroides_uniformis': 'GCF_044361425.1_ASM4436142v1'}


all_spe_mat = np.zeros((4, 4))
all_p_mat = np.zeros((4, 4))
axis_labels = ['HMP2','Franzosa','Liu et al.','HMP3']
diag = []
triangle = []
for species in  all_genomes.keys():
    asm = all_genomes[species]
    cor_mat, p_mat = get_cor_mat(asm)
    p_mat = (p_mat < 0.05).astype(int)
    all_spe_mat += cor_mat
    all_p_mat += p_mat
    diag.append(cor_mat.diagonal())
    triangle.append(cor_mat[np.triu(cor_mat, k =1)!=0])
    g = sns.heatmap(cor_mat, xticklabels=axis_labels, yticklabels=axis_labels,vmin=0)
    g.set_title(species)
    g.figure.savefig(f"plots/correlation_heatmaps/illumina_dataset_comparison_{species}_78.png")
    plt.clf()
all_spe_mat /= len(all_genomes)
all_p_mat = all_p_mat.astype(int) 
fig, ax = plt.subplots(figsize=(8, 6))
g = sns.heatmap(all_spe_mat, xticklabels=axis_labels, yticklabels=axis_labels,vmax = 0.8, vmin=0, cmap="Oranges",
               annot = all_p_mat, fmt='d', annot_kws={'size':12}, ax=ax)
g.set_title("5 species average")
g.figure.savefig(f"plots/correlation_heatmaps/illumina_dataset_comparison_hmp2_78.pdf")
plt.clf() 

diag = all_spe_mat.diagonal()
diag.mean() # 0.36843026334046935
diag.std() # 0.0677569273553093

triangle = all_spe_mat[np.triu(all_spe_mat, k =1)!=0]
triangle.mean() # 0.26496918025217003
triangle.std() # 0.0414803198608098
stats.ttest_ind(diag, triangle) # Ttest_indResult(statistic=2.6765768715298925, pvalue=0.028074557008736664)

### Get Null ###
all_spe_mat = np.zeros((4, 4))
all_p_mat = np.zeros((4, 4))
axis_labels = ['HMP2','Franzosa','Liu et al.','HMP3']
for species in  all_genomes.keys():
    asm = all_genomes[species]
    cor_mat, p_mat = get_null_mat(asm)
    p_mat = (p_mat < 0.05).astype(int)
    all_spe_mat += cor_mat
    all_p_mat += p_mat
all_spe_mat /= len(all_genomes)
all_spe_mat = abs(all_spe_mat)
all_p_mat = all_p_mat.astype(int) 
fig, ax = plt.subplots(figsize=(8, 6))
g = sns.heatmap(all_spe_mat, xticklabels=axis_labels, yticklabels=axis_labels,vmax = 0.8, vmin=0, cmap="Oranges", annot = all_p_mat, fmt='d', annot_kws={'size':12}, ax=ax)
g.set_title("Null species average")
g.figure.savefig(f"plots/correlation_heatmaps/illumina_dataset_comparison_hmp2_78_null.pdf")
plt.clf() 

all_spe_mat.mean() # 0.0008986888560413867
all_spe_mat.std() # 0.0005564765835261918
 
null_diag = all_spe_mat.diagonal()
null_diag.mean() # 0.0011232751153674616
null_diag.std() # 0.0009434070684256246

null_triangle = all_spe_mat[np.triu(all_spe_mat, k =1)!=0]
null_triangle.mean() # 0.0012638908500645655
null_triangle.std() # 0.0008115640543393425
stats.ttest_ind(null_diag, null_triangle) # Ttest_indResult(statistic=-0.22480712575119202, pvalue=0.8277651370569676)

stats.ttest_ind(diag, null_diag) # Ttest_indResult(statistic=9.388452704113645, pvalue=8.290906093578645e-05)
stats.ttest_ind(triangle, null_triangle) # Ttest_indResult(statistic=14.212767132507427, pvalue=5.86284876915737e-08)
