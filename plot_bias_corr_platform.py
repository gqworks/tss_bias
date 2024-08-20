import re, os
import pandas as pd
import seaborn as sns
import numpy as np
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
    tss_dat = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
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

def get_avg_r(a,b):
    r_vals = []
    for r in itertools.product(a, b):
        nan = np.isnan(r[0]) | np.isnan(r[1])
        if np.array_equal(r[0][~nan], r[1][~nan]):
            continue
        r_vals.append(np.corrcoef(r[0][~nan],r[1][~nan])[0][1])
    return np.mean(r_vals)

def get_cor_mat(asm):
    illumina_samples_bias = np.array([get_bias_from_mat_filtshort(asm,samp) for samp in Franzosa_samples])
    pacbio_samples_bias = np.array([get_bias_from_mat_filtshort(asm,samp) for samp in pacbio_samples])
    nanopore_samples_bias = np.array([get_bias_from_mat_filtshort(asm,samp) for samp in nanopore_samples])
    illumina_samples_bias = illumina_samples_bias[[False if len(x)==1 else True for x in illumina_samples_bias]]
    pacbio_samples_bias = pacbio_samples_bias[[False if len(x)==1 else True for x in pacbio_samples_bias]]
    nanopore_samples_bias = nanopore_samples_bias[[False if len(x)==1 else True for x in nanopore_samples_bias]]
    all_bias_groups = [illumina_samples_bias,pacbio_samples_bias,nanopore_samples_bias]
    heatmap_vals = []
    for x in all_bias_groups:
        for y in all_bias_groups:
            heatmap_vals.append(get_avg_r(x,y))
    cor_mat = np.array(heatmap_vals).reshape(3,3)
    return(cor_mat)

def get_null_mat(asm):
    tss_dat = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
    illumina_samples_bias = np.array([get_null_bias_from_mat(asm,samp,tss_dat) for samp in Franzosa_samples])
    pacbio_samples_bias = np.array([get_null_bias_from_mat(asm,samp,tss_dat) for samp in pacbio_samples])
    nanopore_samples_bias = np.array([get_null_bias_from_mat(asm,samp,tss_dat) for samp in nanopore_samples])
    illumina_samples_bias = illumina_samples_bias[[False if len(x)==1 else True for x in illumina_samples_bias]]
    pacbio_samples_bias = pacbio_samples_bias[[False if len(x)==1 else True for x in pacbio_samples_bias]]
    nanopore_samples_bias = nanopore_samples_bias[[False if len(x)==1 else True for x in nanopore_samples_bias]]
    null_bias_groups = [illumina_samples_bias,pacbio_samples_bias,nanopore_samples_bias]
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

with open('pacbio_idlist') as f:
    pacbio_samples = f.read().splitlines()
pacbio_samples = [x.rsplit('\t',1)[0] for x in pacbio_samples]

with open('nanopore_idlist') as f:
    nanopore_samples = f.read().splitlines()
nanopore_samples = [x.rsplit('\t',1)[0] for x in nanopore_samples]

#Read species abundances
illumina = pd.read_table('metaphlan3_res/merged_mpa.txt', comment= '#')
illumina = illumina[illumina['clade_name'].str.contains("s__")]
illumina.index = [int(re.match('.*\|([^\|]*)',x).group(1)) for x in illumina['NCBI_tax_id']]
illumina = illumina[Franzosa_samples]
pacbio = pd.read_csv('src/pacbio_merged_k2_raw.csv',header = 0, index_col = 0)
pacbio = 100*pacbio/pacbio.sum(axis=0)
pacbio.index = pacbio.index.values
nanopore = pd.read_csv('src/nanopore_merged_k2_raw.csv',header = 0, index_col = 0)
nanopore = 100*nanopore/nanopore.sum(axis=0)

merge = pd.concat([illumina,pacbio,nanopore], axis=1)
merge = merge.loc[((merge[Franzosa_samples] > 0.5).sum(axis=1) >= 4) & ((merge[pacbio_samples] > 0.5).sum(axis=1) >= 4) & ((merge[nanopore_samples] > 0.5).sum(axis=1) >= 4),:] # Filter species with atleast 0.5% abundance in atleast 4 samples in each dataset
merge[merge < 0.5] = np.nan
merge = merge.drop(39491) #Remove Eubacterium Rectale because no reference genome in refseq

refseq = pd.read_table('src/assembly_summary_refseq.txt', header = 1)

#Generate heatmaps
axis_labels = ['illumina','pacbio','nanopore']
all_spe_mat = np.array([[0]*3]*3)
pacbio_vals = []
nanopore_vals = []
pac_vs_nano_vals = []
ill_vs_pac_vals = []
ill_vs_nano_vals = []
for taxid in merge.index:
    asms = taxid2asm(taxid,3)
    cor_mat = np.array([[0]*3]*3)
    for asm in asms:
        cor_mat = cor_mat + get_cor_mat(asm)
    cor_mat = cor_mat/len(asms)
    pacbio_vals.append(cor_mat[1][1])
    nanopore_vals.append(cor_mat[2][2])
    pac_vs_nano_vals.append(cor_mat[1][2])
    ill_vs_pac_vals.append(cor_mat[0][1])
    ill_vs_nano_vals.append(cor_mat[0][2])
    all_spe_mat = all_spe_mat + cor_mat
    g = sns.heatmap(cor_mat, xticklabels=axis_labels, yticklabels=axis_labels,vmin=0)
    g.set_title(taxid)
    g.figure.savefig(f"plots/correlation_heatmaps/platform_comparison_{taxid}.png")
    plt.clf()
all_spe_mat = all_spe_mat/len(merge.index)
g = sns.heatmap(all_spe_mat, xticklabels=axis_labels, yticklabels=axis_labels,vmin=0,vmax=0.8,cmap='Oranges')
g.set_title("Species average")
g.figure.savefig(f"plots/correlation_heatmaps/platform_comparison_all.pdf")
plt.clf() 

np.mean(pacbio_vals) #0.6537183081745702
np.std(pacbio_vals) #0.05270351454758188
np.mean(nanopore_vals) #0.7781289304655508
np.std(nanopore_vals) #0.021207757622492003
np.mean(pac_vs_nano_vals) #0.5758093478960193
np.std(pac_vs_nano_vals) #0.03389603249458659
np.mean(ill_vs_pac_vals) #0.21704527373906526
np.std(ill_vs_pac_vals) #0.09512392621586485
np.mean(ill_vs_nano_vals) #0.21850245470620638
np.std(ill_vs_nano_vals) #0.11898358083836053

### Get Null ###
all_spe_mat = np.array([[0]*3]*3)
null_ill_vs_pac_vals = []
null_ill_vs_nano_vals = []
for taxid in merge.index:
    asms = taxid2asm(taxid,3)
    cor_mat = np.array([[0]*3]*3)
    for asm in asms:
        cor_mat = cor_mat + get_null_mat(asm)
    cor_mat = cor_mat/len(asms)
    null_ill_vs_pac_vals.append(cor_mat[0][1])
    null_ill_vs_nano_vals.append(cor_mat[0][2])
    all_spe_mat = all_spe_mat + cor_mat
all_spe_mat = all_spe_mat/len(merge.index)
all_spe_mat = abs(all_spe_mat)
g = sns.heatmap(all_spe_mat, xticklabels=axis_labels, yticklabels=axis_labels, vmin =0, vmax = 0.8, cmap = 'Oranges')
g.set_title("Null species average")
g.figure.savefig(f"plots/correlation_heatmaps/platform_comparison_all_null.pdf")
plt.clf()

np.mean(null_ill_vs_pac_vals) #0.0011569491286293658
np.std(null_ill_vs_pac_vals) #0.0023390248272495383
np.mean(np.abs(null_ill_vs_nano_vals)) #0.0026153140555543928
np.std(null_ill_vs_nano_vals) #0.003398552002540536

stats.ttest_ind(ill_vs_pac_vals, null_ill_vs_pac_vals) #Ttest_indResult(statistic=3.929784739436785, pvalue=0.007714653398191839)
stats.ttest_ind(ill_vs_nano_vals, null_ill_vs_nano_vals) #Ttest_indResult(statistic=3.1863041212551475, pvalue=0.018924476266285115)
