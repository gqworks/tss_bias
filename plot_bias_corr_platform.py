import re, os
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import itertools
import pyBigWig


# --- Gadi/portable bootstrap (added in refactor) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from tss_bias.runtime import bootstrap as _bootstrap, mpa_path as _mpa_path
_CFG = _bootstrap(chdir_work=False)
_MPA = _mpa_path(_CFG)
_HMP2_FQ = _CFG["cohorts"]["hmp2"]["fastq_dir"]
# --- end bootstrap ---
def scale_mat(d):
    return (d-d.mean(axis=1)[:,None]) / d.std(axis=1)[:,None]

def calc_bias(x):
    left_sum = np.sum(x[:500])
    right_sum = np.sum(x[500:])
    return(left_sum-right_sum)

def randomize(intervals):
    intervals = intervals.astype(int)
    point_diffs = intervals[1:] - intervals[:-1]
    max_len = intervals[-1]
    new_points = np.concatenate((np.array(point_diffs).cumsum(),[max_len]))
    new_points = (new_points + np.random.randint(0,max_len))%max_len
    return new_points
    
def get_bias_from_mat_filtshort(asm,sample):
    if not os.path.exists(f'matrices/{asm}_{sample}.csv'):
        return(np.array([0]))
    tss_dat = pd.read_csv(f'refseq_genomes/{asm}/{asm}_filtered_tss.csv')
    tss_dat.index = np.array([re.match(r'ID=([^;]*)',x).group(1) for x in tss_dat['Desc']])
    tss_dat['len'] = tss_dat['geneRight']-tss_dat['geneLeft']
    tss_dat = tss_dat[tss_dat['len'] > 1000]
    mat = pd.read_table(f'matrices/{asm}_{sample}.csv', sep = ',', header=None, index_col=0)
    mat = mat.loc[tss_dat.index,:]
    bias = np.array(list(map(calc_bias, np.array(scale_mat(mat)))))
    #bias[np.isnan(bias)] = 0
    return bias
    
def get_bias_from_mat(asm,sample):
    if not os.path.exists(f'matrices/{asm}_{sample}.csv'):
        return(np.array([0]))
    mat = pd.read_table(f'matrices/{asm}_{sample}.csv', sep = ',', header=None, index_col=0)
    bias = np.array(list(map(calc_bias, np.array(scale_mat(mat)))))
    #bias[np.isnan(bias)] = 0
    return bias

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

def get_cor_mat(asm):
    illumina_samples_bias = np.array([get_bias_from_mat_filtshort(asm,samp) for samp in hmp2_samples])
    pacbio_samples_bias = np.array([get_bias_from_mat_filtshort(asm,samp) for samp in pacbio_samples])
    nanopore_samples_bias = np.array([get_bias_from_mat_filtshort(asm,samp) for samp in nanopore_samples])
    
    illumina_samples_bias = illumina_samples_bias[[False if len(x)==1 else True for x in illumina_samples_bias]]
    pacbio_samples_bias = pacbio_samples_bias[[False if len(x)==1 else True for x in pacbio_samples_bias]]
    nanopore_samples_bias = nanopore_samples_bias[[False if len(x)==1 else True for x in nanopore_samples_bias]]
    all_bias_groups = [illumina_samples_bias,pacbio_samples_bias,nanopore_samples_bias]
    cor_vals = []
    p_vals = []
    for x in all_bias_groups:
        for y in all_bias_groups:
            avg_r, avg_p = get_avg_r_p(x, y)
            cor_vals.append(avg_r)
            p_vals.append(avg_p)
    cor_mat = np.array(cor_vals).reshape(3,3)
    p_mat = np.array(p_vals).reshape(3,3)
    return cor_mat, p_mat

def get_null_mat(asm):
    tss_dat = pd.read_csv(f'refseq_genomes/{asm}/{asm}_filtered_tss.csv')
    illumina_samples_bias = np.array([get_null_bias_from_mat(asm,samp,tss_dat) for samp in hmp2_samples])
    pacbio_samples_bias = np.array([get_null_bias_from_mat(asm,samp,tss_dat) for samp in pacbio_samples])
    nanopore_samples_bias = np.array([get_null_bias_from_mat(asm,samp,tss_dat) for samp in nanopore_samples])
    illumina_samples_bias = illumina_samples_bias[[False if len(x)==1 else True for x in illumina_samples_bias]]
    pacbio_samples_bias = pacbio_samples_bias[[False if len(x)==1 else True for x in pacbio_samples_bias]]
    nanopore_samples_bias = nanopore_samples_bias[[False if len(x)==1 else True for x in nanopore_samples_bias]]
    null_bias_groups = [illumina_samples_bias,pacbio_samples_bias,nanopore_samples_bias]
    cor_vals = []
    p_vals = []
    for x in null_bias_groups:
        for y in null_bias_groups:
            avg_r, avg_p = get_avg_r_p(x, y)
            cor_vals.append(avg_r)
            p_vals.append(avg_p)
    cor_mat = np.array(cor_vals).reshape(3,3)
    p_mat = np.array(p_vals).reshape(3,3)
    return cor_mat, p_mat


###Initialise Data
with open('sample_lists/hmp2_samples') as f:
    hmp2_samples = [ln for ln in f.read().splitlines() if ln.strip() and not ln.startswith('#')]
hmp2_samples = {x.split('\t')[0]:x.split('\t')[1] for x in hmp2_samples}

with open('sample_lists/Franzosa_8_samples') as f:
    Franzosa_samples = [ln for ln in f.read().splitlines() if ln.strip() and not ln.startswith('#')]
Franzosa_samples = [x.rsplit('\t',1)[0] for x in Franzosa_samples]

with open('sample_lists/pacbio_idlist') as f:
    pacbio_samples = [ln for ln in f.read().splitlines() if ln.strip() and not ln.startswith('#')]
pacbio_samples = [x.rsplit('\t',1)[0] for x in pacbio_samples]

with open('sample_lists/nanopore_idlist') as f:
    nanopore_samples = [ln for ln in f.read().splitlines() if ln.strip() and not ln.startswith('#')]
nanopore_samples = [x.rsplit('\t',1)[0] for x in nanopore_samples]

#Read species abundances
illumina = pd.read_table(_MPA, comment= '#')
illumina = illumina[(illumina['clade_name'].str.contains("s__"))&(illumina['clade_name'].str.contains("k__Bacteria"))&(~illumina['clade_name'].str.contains("t__"))]
illumina.index = [re.match('.*\|([^\|]*)',x).group(1) for x in illumina['clade_name']]
_hmp2_ids = list(hmp2_samples.keys()) if isinstance(hmp2_samples, dict) else list(hmp2_samples)
illumina = illumina[_hmp2_ids]
hmp2_samples = _hmp2_ids
illumina = illumina[illumina.sum(axis=1) > 0]
illumina = illumina.div(illumina.sum(axis=0), axis=1) * 100

pacbio = pd.read_csv(_CFG['paths']['reference_dir'] + '/pacbio_merged_mpa.tsv',header = 0, index_col = 0, sep = '\t')
pacbio = pacbio[(pacbio.index.str.contains("s__"))&(pacbio.index.str.contains("k__Bacillati|k__Pseudomonadati"))&(~pacbio.index.str.contains("t__"))]
pacbio.index = [re.match('.*\|([^\|]*)',x).group(1) for x in pacbio.index]
pacbio = pacbio[pacbio.sum(axis=1) > 100]
pacbio = pacbio.div(pacbio.sum(axis=0), axis=1) * 100

nanopore = pd.read_csv(_CFG['paths']['reference_dir'] + '/nanopore_merged_mpa.tsv',header = 0, index_col = 0, sep = '\t')
nanopore = nanopore[(nanopore.index.str.contains("s__"))&(nanopore.index.str.contains("k__Bacillati|k__Pseudomonadati"))&(~nanopore.index.str.contains("t__"))]
nanopore.index = [re.match('.*\|([^\|]*)',x).group(1) for x in nanopore.index]
nanopore = nanopore[nanopore.sum(axis=1) > 100]
nanopore = nanopore.div(nanopore.sum(axis=0), axis=1) * 100

merge = pd.concat([illumina,pacbio,nanopore], axis=1)
merge = merge.loc[((merge[hmp2_samples] > 0.5).sum(axis=1) >= 30) & ((merge[pacbio_samples] > 0.5).sum(axis=1) >= 4) & ((merge[nanopore_samples] > 0.5).sum(axis=1) >= 4),:].copy() # Filter species with atleast 0.5% abundance in atleast 4 samples in each dataset
merge[merge < 0.5] = np.nan

all_genomes = {'s__Phocaeicola_vulgatus': 'GCF_000012825.1_ASM1282v1',
 's__Bacteroides_uniformis': 'GCF_044361425.1_ASM4436142v1',
 's__Alistipes_putredinis': 'GCF_000154465.1_ASM15446v1',
 's__Faecalibacterium_prausnitzii': 'GCF_000154385.1_ASM15438v1',
 's__Agathobacter_rectalis': 'GCF_022453685.1_ASM2245368v1'}

#Generate heatmaps
axis_labels = ['illumina','pacbio','nanopore']
all_spe_mat = np.zeros((3, 3))
all_p_mat = np.zeros((3, 3))
pacbio_vals = []
nanopore_vals = []
pac_vs_nano_vals = []
ill_vs_pac_vals = []
ill_vs_nano_vals = []
for species in all_genomes.keys():
    asm = all_genomes[species]
    cor_mat, p_mat = get_cor_mat(asm)
    p_mat = (p_mat < 0.05).astype(int)
    
    pacbio_vals.append(cor_mat[1][1])
    nanopore_vals.append(cor_mat[2][2])
    pac_vs_nano_vals.append(cor_mat[1][2])
    ill_vs_pac_vals.append(cor_mat[0][1])
    ill_vs_nano_vals.append(cor_mat[0][2])
    
    all_spe_mat += cor_mat
    all_p_mat += p_mat
    #g = sns.heatmap(cor_mat, xticklabels=axis_labels, yticklabels=axis_labels,vmin=0)
    #g.set_title(species)
    #g.figure.savefig(f"plots/correlation_heatmaps/platform_comparison_{species}_78.png")
    #plt.clf()
all_spe_mat /= len(all_genomes)
all_p_mat = all_p_mat.astype(int) 
fig, ax = plt.subplots(figsize=(8, 6))
g = sns.heatmap(all_spe_mat, xticklabels=axis_labels, yticklabels=axis_labels,vmax = 0.8, vmin=0, cmap="Oranges",
               annot = all_p_mat, fmt='d', annot_kws={'size':12}, ax=ax)
g.set_title("5 species average")
g.figure.savefig(f"plots/correlation_heatmaps/platform_comparison_hmp2_78.pdf")
plt.clf() 

np.mean(pacbio_vals) #0.6164892602091754
np.std(pacbio_vals) #0.051651736481564806
np.mean(nanopore_vals) #0.7175412499699104
np.std(nanopore_vals) #0.04408875550926504
np.mean(pac_vs_nano_vals) #0.5454766525180952
np.std(pac_vs_nano_vals) #0.04032750308824896
np.mean(ill_vs_pac_vals) #0.1459892762028021
np.std(ill_vs_pac_vals) #0.09894320252505984
np.mean(ill_vs_nano_vals) #0.14544262574544037
np.std(ill_vs_nano_vals) #0.11182717681031078

### Get Null ###
all_spe_mat = np.zeros((3, 3))
all_p_mat = np.zeros((3, 3))
null_ill_vs_pac_vals = []
null_ill_vs_nano_vals = []
for species in all_genomes.keys():
    asm = all_genomes[species]
    cor_mat, p_mat = get_null_mat(asm)
    p_mat = (p_mat < 0.05).astype(int)
    null_ill_vs_pac_vals.append(cor_mat[0][1])
    null_ill_vs_nano_vals.append(cor_mat[0][2])
    all_spe_mat += cor_mat
    all_p_mat += p_mat
all_spe_mat /= len(all_genomes)
all_spe_mat = abs(all_spe_mat)
all_p_mat = all_p_mat.astype(int) 
fig, ax = plt.subplots(figsize=(8, 6))
g = sns.heatmap(all_spe_mat, xticklabels=axis_labels, yticklabels=axis_labels,vmax = 0.8, vmin=0, cmap="Oranges", annot = all_p_mat, fmt='d', annot_kws={'size':12}, ax=ax)
g.set_title("Null species average")
g.figure.savefig(f"plots/correlation_heatmaps/platform_comparison_hmp2_null.pdf")
plt.clf()

np.mean(null_ill_vs_pac_vals) #-0.0009519519843921348
np.std(null_ill_vs_pac_vals) #0.0019484040213831578
np.mean(np.abs(null_ill_vs_nano_vals)) #0.00043809267677187845
np.std(null_ill_vs_nano_vals) #0.00043017503975316764

stats.ttest_ind(ill_vs_pac_vals, null_ill_vs_pac_vals) #Ttest_indResult(statistic=2.969637978972378, pvalue=0.017881026661356134)
stats.ttest_ind(ill_vs_nano_vals, null_ill_vs_nano_vals) #Ttest_indResult(statistic=2.605171993589753, pvalue=0.03136470815361509)
