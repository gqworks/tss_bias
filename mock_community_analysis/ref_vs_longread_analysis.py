import pandas as pd
import re
import numpy as np
import pyBigWig
from pysam import FastaFile
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel


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

def asm_2_species(asm):
    name = refseq[refseq['ftp_path'].str.contains(asm)]['organism_name'].item()
    return re.match(r'(^[^ ]+ [^ ]+).*',name).group(1)

###Initialise sample data
with open('matched_dataset_idlist') as f:
    samples = f.read().splitlines()
samples = [x.split('\t') for x in samples]

mpa_res = pd.read_table('metaphlan3_res/merged_mpa.txt', comment= '#')
mpa_res = mpa_res[mpa_res['clade_name'].str.contains("s__")]
mpa_res.index = [int(re.match('.*\|([^\|]*)',x).group(1)) for x in mpa_res['NCBI_tax_id']]
mpa_res = mpa_res.drop(columns=['clade_name', 'NCBI_tax_id'])
mpa_res = mpa_res.loc[(mpa_res > 1).sum(axis=1) > 5,:] # Filter species with atleast 1% abundance in more than 5 samples
mpa_res[mpa_res < 1] = np.nan
refseq = pd.read_table('../src/assembly_summary_refseq.txt', header = 1)

bias_means = []
for taxid in mpa_res.index:
    asms = taxid2asm(taxid,3)
    for asm in asms:
        for samp in samples:
            if np.isnan(mpa_res.loc[taxid,samp[0]]):
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
            
            bias_means.append([asm,samp[0],ref_mean,longread_mean,simu_mean])
bias_means = pd.DataFrame(bias_means)
bias_means.to_csv("matched_dataset_all_bias_values.csv")


df = pd.read_csv("matched_dataset_all_bias_values.csv", index_col=0)
df = df.rename(columns={'0': 'asm', '1': 'sample', '2': 'genome', '3': 'long read', '4': 'simulated'})
asm_2_species_dict = {x:asm_2_species(x) for x in set(df.asm)}
df['species'] = df.asm.map(asm_2_species_dict)
df_melt = df.groupby(['sample','species']).mean().reset_index()
sns.boxplot(data=df_melt, showfliers=False, color = 'tab:blue')
plt.ylim(-35, 120)
add_p_val(plt,0,1,105,3,ttest_rel(df_melt['genome'],df_melt['long read']).pvalue) # Ttest_relResult(statistic=4.0750025977880835, pvalue=0.0003262267921443576)
add_p_val(plt,1,2,80,3,ttest_rel(df_melt['simulated'],df_melt['long read']).pvalue) # Ttest_relResult(statistic=-18.701907962016264, pvalue=1.0037119994019263e-17)
plt.savefig(f"plots/matched_dataest_bias_reference_vs_longread_boxplot.pdf")
plt.clf()

print(df_melt['genome'].mean()) # 48.50438649198689
print(df_melt['long read'].mean()) # 28.996167152517973
print(df_melt['simulated'].mean()) # -1.55602686983081
