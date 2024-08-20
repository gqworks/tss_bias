import re, os
import pandas as pd
import seaborn as sns
import numpy as np
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
import itertools
import pyBigWig

def taxid2asm(taxid,n=1):
    strain = refseq[(refseq['species_taxid']==int(taxid)) & (refseq['assembly_level']=='Complete Genome')]
    if len(strain) == 0:
        strain = refseq[(refseq['species_taxid']==int(taxid))]
    ftp = strain['ftp_path'][:n].tolist()
    asm = [x.rsplit('/',1)[1] for x in ftp]
    return(asm)

def get_cor_mat(asm):
    Franzosa_samples_bias = np.array([get_bias_from_mat(asm,samp) for samp in Franzosa_samples])
    arg_samples_bias = np.array([get_bias_from_mat(asm,samp) for samp in arg_samples])
    hyp_samples_bias = np.array([get_bias_from_mat(asm,samp) for samp in hyp_samples])
    Franzosa_samples_bias = Franzosa_samples_bias[[False if len(x)==1 else True for x in Franzosa_samples_bias]]
    arg_samples_bias = arg_samples_bias[[False if len(x)==1 else True for x in arg_samples_bias]]
    hyp_samples_bias = hyp_samples_bias[[False if len(x)==1 else True for x in hyp_samples_bias]]
    all_bias_groups = [Franzosa_samples_bias,arg_samples_bias,hyp_samples_bias]
    heatmap_vals = []
    for x in all_bias_groups:
        for y in all_bias_groups:
            heatmap_vals.append(get_avg_r(x,y))
    cor_mat = np.array(heatmap_vals).reshape(3,3)
    return(cor_mat)

def get_avg_r(a,b):
    r_vals = []
    for r in itertools.product(a, b):
        nan = np.isnan(r[0]) | np.isnan(r[1])
        if np.array_equal(r[0][~nan], r[1][~nan]):
            continue
        r_vals.append(np.corrcoef(r[0][~nan],r[1][~nan])[0][1])
    return np.mean(r_vals)

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
    tss_dat = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
    Franzosa_samples_bias = np.array([get_null_bias_from_mat(asm,samp,tss_dat) for samp in Franzosa_samples])
    arg_samples_bias = np.array([get_null_bias_from_mat(asm,samp,tss_dat) for samp in arg_samples])
    hyp_samples_bias = np.array([get_null_bias_from_mat(asm,samp,tss_dat) for samp in hyp_samples])
    Franzosa_samples_bias = Franzosa_samples_bias[[False if len(x)==1 else True for x in Franzosa_samples_bias]]
    arg_samples_bias = arg_samples_bias[[False if len(x)==1 else True for x in arg_samples_bias]]
    hyp_samples_bias = hyp_samples_bias[[False if len(x)==1 else True for x in hyp_samples_bias]]
    null_bias_groups = [Franzosa_samples_bias,arg_samples_bias,hyp_samples_bias]
    heatmap_vals = []
    for x in null_bias_groups:
        for y in null_bias_groups:
            heatmap_vals.append(get_avg_r(x,y))
    cor_mat = np.array(heatmap_vals).reshape(3,3)
    return(cor_mat)



###Initialise Data
with open('Franzosa_8_samples') as f:
    Franzosa_samples = f.read().splitlines()
Franzosa_samples = [x.rsplit('\t',1)[0] for x in Franzosa_samples]

with open('arg_samples') as f:
    arg_samples = f.read().splitlines()
arg_samples = [x.rsplit('\t',1)[0] for x in arg_samples]

with open('hyp_samples') as f:
    hyp_samples = f.read().splitlines()
hyp_samples = [x.rsplit('\t',1)[0] for x in hyp_samples]

#Read species abundances
mpa_res = pd.read_table('metaphlan3_res/merged_mpa.txt', comment= '#')
mpa_res = mpa_res[mpa_res['clade_name'].str.contains("s__")]
mpa_res.index = [re.match('.*\|([^\|]*)',x).group(1) for x in mpa_res['NCBI_tax_id']]
mpa_res = mpa_res.drop(columns=['clade_name', 'NCBI_tax_id'])
# Filter species with atleast 1% abundance in atleast 5 samples in each dataset
mpa_res = mpa_res.loc[((mpa_res[Franzosa_samples] > 1).sum(axis=1) >= 5) & ((mpa_res[arg_samples] > 1).sum(axis=1) >= 5) & ((mpa_res[hyp_samples] > 1).sum(axis=1) >= 5),:]
mpa_res[mpa_res < 1] = np.nan
mpa_res = mpa_res.drop('39491') #Remove Eubacterium Rectale because no reference genome in refseq

refseq = pd.read_table('src/assembly_summary_refseq.txt', header = 1)

all_spe_mat = np.array([[0]*3]*3)
axis_labels = ['matched','ARG','HYP']
diag = []
triangle = []
for taxid in tqdm(mpa_res.index):
    asms = taxid2asm(taxid,3)
    cor_mat = np.array([[0]*3]*3)
    for asm in asms:
        cor_mat = cor_mat + get_cor_mat(asm)
    cor_mat = cor_mat/len(asms)
    all_spe_mat = all_spe_mat + cor_mat
    diag.append(cor_mat.diagonal())
    triangle.append(cor_mat[np.triu(cor_mat, k =1)!=0])
    g = sns.heatmap(cor_mat, xticklabels=axis_labels, yticklabels=axis_labels,vmin=0)
    g.set_title(taxid)
    g.figure.savefig(f"plots/correlation_heatmaps/illumina_data_comparison_{taxid}.png")
    plt.clf()
all_spe_mat = all_spe_mat/len(mpa_res)
g = sns.heatmap(all_spe_mat, xticklabels=axis_labels, yticklabels=axis_labels,vmax = 0.8, vmin=0, cmap="Oranges")
g.set_title("4 species average")
g.figure.savefig(f"plots/correlation_heatmaps/illuminua_dataset_comparison_all.pdf")
plt.clf() 

diag = all_spe_mat.diagonal()
diag.mean() #0.345664292477652
diag.std() #0.030740803371698006

triangle = all_spe_mat[np.triu(all_spe_mat, k =1)!=0]
triangle.mean() # 0.2987134763291012
triangle.std() # 0.023134904046383655
stats.ttest_ind(diag, triangle) #Ttest_indResult(statistic=1.7258176090272515, pvalue=0.1594608070751777)

### Get Null ###
all_spe_mat = np.array([[0]*3]*3)
axis_labels = ['matched','ARG','HYP']
for taxid in tqdm(mpa_res.index):
    asms = taxid2asm(taxid,3)
    cor_mat = np.array([[0]*3]*3)
    for asm in asms:
        cor_mat = cor_mat + get_null_mat(asm)
    cor_mat = cor_mat/len(asms)
    all_spe_mat = all_spe_mat + cor_mat
all_spe_mat = all_spe_mat/len(mpa_res)
all_spe_mat = abs(all_spe_mat)
g = sns.heatmap(all_spe_mat, xticklabels=axis_labels, yticklabels=axis_labels,vmax = 0.8, vmin=0, cmap="Oranges")
g.set_title("Null species average")
g.figure.savefig(f"plots/correlation_heatmaps/illumina_dataset_comparison_all_null.pdf")
plt.clf() 

all_spe_mat.mean() # 0.00027055032813890146
all_spe_mat.std() # 0.0008381293685912436
 