import pandas as pd
import re
import numpy as np
import pyBigWig
from pysam import FastaFile
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel

def scale_mat(d):
    return (d-d.mean(axis=1)[:,None]) / d.std(axis=1)[:,None]

def calc_bias(x):
    left_sum = np.sum(x[:500])
    right_sum = np.sum(x[500:])
    return(right_sum-left_sum)

def filter_overlap_genes(gff,ref_len_dict):
    filt_genes = []
    for contig in list(ref_len_dict.keys()):
        genes = gff[gff[:,0] == contig, :]
        #genes = np.array(gff[gff[0]==contig])
        tot_len = ref_len_dict[contig]
        for i in range(1,len(genes)-1): # Ignore first and last gene to avoid writing edge cases, also they most likely fall into the 500bp long read edge anyway
            left_pos = genes[i,2]
            right_pos = genes[i,3]
            strand = genes[i,5]
            start = left_pos if strand == '+' else right_pos
            if start-500 < 500 or start+500 > tot_len: #remove 500bp upstream/downstream of edges
                continue
            previous_gene_start = genes[i-1,2] if genes[i-1,5] == '+' else genes[i-1,3]
            next_gene_start = genes[i+1,2] if genes[i+1,5] == '+' else genes[i+1,3]
            if start-500 > previous_gene_start and start+500 < next_gene_start:
                filt_genes.append(genes[i])
    return(np.array(filt_genes))

def add_p_val(ax,lft,rgt,y,h,p):
    ax.plot([lft, lft, rgt, rgt], [y, y+h, y+h, y], lw=1.5, c='k')
    ax.text((lft + rgt) * .5, y+h, ('n.s.' if p > 0.15 else 'p < %.2g' if p > 0.001 else 'p < %.1g') % max(p+1e-20, 1e-20), ha='center', va='bottom', color='k')


###Initialise sample data
with open('matched_dataset_idlist') as f:
    samples = f.read().splitlines()
samples = [x.split('\t') for x in samples]

mpa_res = pd.read_table('metaphlan3_res/merged_mpa.txt', comment= '#')
mpa_res = mpa_res[mpa_res['clade_name'].str.contains("s__")]
mpa_res.index = [re.match('.*\|([^\|]*)',x).group(1) for x in mpa_res['clade_name']]
mpa_res = mpa_res.drop(columns=['clade_name','NCBI_tax_id'])
mpa_res = mpa_res.loc[(mpa_res > 0.5).sum(axis=1) > 5,:] # Filter species with atleast 0.5% abundance in more than 5 samples
mpa_res = mpa_res.where(mpa_res > 0.5, np.nan)
mpa_res = mpa_res.rename(index={'s__Bacteroides_vulgatus': 's__Phocaeicola_vulgatus', 's__Eubacterium_hallii':'s__Anaerobutyricum_hallii'})

selected_genomes = {'s__Bacteroides_uniformis': 'GCF_044361425.1_ASM4436142v1',
 's__Phocaeicola_vulgatus': 'GCF_000012825.1_ASM1282v1',
 's__Alistipes_putredinis': 'GCF_000154465.1_ASM15446v1',
 's__Anaerobutyricum_hallii': 'GCF_018140635.1_ASM1814063v1',
 's__Anaerostipes_hadrus': 'GCF_030864025.1_ASM3086402v1',
 's__Dorea_longicatena': 'GCF_025150085.1_ASM2515008v1',
 's__Fusicatenibacter_saccharivorans': 'GCF_027667425.1_ASM2766742v1',
 's__Roseburia_faecis': 'GCF_015550045.1_ASM1555004v1',
 's__Faecalibacterium_prausnitzii': 'GCF_000154385.1_ASM15438v1'}

bias_means = []
for species in mpa_res.index:
    asm = selected_genomes[species]
    for samp in samples:
        if np.isnan(mpa_res.loc[species,samp[0]]):
            continue
        samp_id = samp[0]

        #Reference genome
        ref_bias_vec = []
        mat = pd.read_csv(f'matrices/{asm}_{samp_id}.csv', header = None, index_col = 0)
        scaled_mat = scale_mat(mat)
        scaled_mat = scaled_mat.dropna()
        ref_mean = np.mean(list(map(calc_bias, np.array(scaled_mat))))

        #Long read reference
        chrom_names = FastaFile(f'strains/{asm}/{asm}.fna').references
        bw_longRead = pyBigWig.open(f'beds/longread_{samp_id}.bw')
        bw_longRead_simu = pyBigWig.open(f'beds/{samp_id}_simu.bw')
        longRead_dict = bw_longRead.chroms()
        minimap_longRead_annot = pd.read_table(f'minimap_res/{samp_id}_minimap2.paf', header=None, usecols=range(12))
        minimap_longRead_annot['readName'] = [x.rsplit('_',1)[0] for x in minimap_longRead_annot[0]]
        minimap_longRead_annot['chromName'] = [x.rsplit('_',1)[0] for x in minimap_longRead_annot[5]]
        minimap_longRead_annot = minimap_longRead_annot[minimap_longRead_annot['chromName'].isin(chrom_names)]

        #Filter longreads with matching genes with genome
        gff_longRead = pd.read_table(f'primary_seq/{samp[2]}_100K_prodigal.gff', comment = '#', header=None)
        gff_longRead = gff_longRead[gff_longRead[0].isin(minimap_longRead_annot['readName'])]
        gff_longRead['len'] = gff_longRead[0].map(longRead_dict)
        gff_longRead = gff_longRead[gff_longRead['len']>10000]
        read_set = list(set(gff_longRead[0]))
        gff_longRead[3] = gff_longRead[3] - 1 # change for 0-indexed
        gff_longRead[4] = gff_longRead[4] - 1 # change for 0-indexed

        filt_genes = []
        for longRead in read_set:
            read_len = longRead_dict[longRead]
            genes = np.array(gff_longRead[gff_longRead[0]==longRead])
            filt_genes = filt_genes + filter_overlap_genes(genes,longRead_dict).tolist()
        filt_genes = np.array(filt_genes)

        #Gene Coverage at TSS
        mat = []
        simu_mat = []
        for gene in filt_genes:
            strand = gene[5]
            start = int(gene[2]) if strand == '+' else int(gene[3])
            read_cv = np.array(bw_longRead.values(gene[0], start-500, start+500))
            read_cv = np.flip(read_cv) if strand == '-' else read_cv
            mat.append(read_cv)
            simu_read_cv = np.array(bw_longRead_simu.values(gene[0], start-500, start+500))
            simu_read_cv = np.flip(simu_read_cv) if strand == '-' else simu_read_cv
            simu_mat.append(simu_read_cv)
        mat = pd.DataFrame(mat)
        longread_scaled_mat = scale_mat(mat)
        longread_scaled_mat = longread_scaled_mat.dropna()
        longread_mean = np.mean(list(map(calc_bias, np.array(longread_scaled_mat))))

        simu_mat = pd.DataFrame(simu_mat)
        longread_scaled_mat_simu = scale_mat(simu_mat)
        longread_scaled_mat_simu = longread_scaled_mat_simu.dropna()
        simu_mean = np.mean(list(map(calc_bias, np.array(longread_scaled_mat_simu))))

        bias_means.append([species,samp[0],ref_mean,longread_mean,simu_mean])
bias_means = pd.DataFrame(bias_means)
bias_means.to_csv("matched_dataset_all_bias_values_thresh0.05.csv")


df = pd.read_csv("matched_dataset_all_bias_values_thresh0.05.csv", index_col=0)
df = df.rename(columns={'0': 'species', '1': 'sample', '2': 'genome', '3': 'long read', '4': 'simulated'})
df_melt = df.groupby(['sample','species']).mean().reset_index()
sns.boxplot(data=df_melt, showfliers=False, color = 'tab:blue')
plt.ylim(-20, 140)
add_p_val(plt,0,1,125,3,ttest_rel(df_melt['genome'],df_melt['long read']).pvalue) # Ttest_relResult(statistic=5.352728716095362, pvalue=1.4342280717297276e-06)
add_p_val(plt,1,2,80,3,ttest_rel(df_melt['simulated'],df_melt['long read']).pvalue) # Ttest_relResult(statistic=-18.815192521191126, pvalue=7.551605816172399e-27)
plt.savefig(f"plots/matched_dataest_bias_reference_vs_longread_boxplot_thresh0.05.pdf")
plt.clf()

print(df_melt['genome'].mean()) # 53.994549116870736
print(df_melt['long read'].mean()) # 29.461491026061914
print(df_melt['simulated'].mean()) # -1.866182757025613
