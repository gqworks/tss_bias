import pandas as pd
import re, os
import numpy as np
import subprocess
from pysam import FastaFile
from Bio import SeqIO
import seaborn as sns
from matplotlib import pyplot as plt
import pysam, functools
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
from scipy.stats.mstats import kruskal


# --- Gadi/portable bootstrap (added in refactor) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from tss_bias.runtime import bootstrap as _bootstrap, mpa_path as _mpa_path
_CFG = _bootstrap(chdir_work=False)
_MPA = _mpa_path(_CFG)
_HMP2_FQ = _CFG["cohorts"]["hmp2"]["fastq_dir"]
# --- end bootstrap ---
def get_tpm(file):
    dat = pd.read_table(file)
    dat = dat.drop_duplicates(subset='target_id')
    dat = dat[['target_id','tpm']]
    dat.index = dat['target_id']
    dat = dat.drop(columns = ['target_id'])
    return dat

def scale_mat(d):
    mean_vals = d.mean(axis=1).to_numpy()
    std_vals = d.std(axis=1).to_numpy()
    return (d - mean_vals[:, None]) / std_vals[:, None]

def calc_bias(x):
    left_sum = np.sum(x[:500])
    right_sum = np.sum(x[500:])
    return(right_sum-left_sum)

def get_upstream_dist(tss_dat,ref):
    dist_2_gene = []
    for genome in list(set(tss_dat[0])):
        dat = tss_dat[tss_dat[0]==genome]
        genome_len = len(ref[genome].seq)
        for i, row in dat.iterrows():
            if row[6] == '-':
                if i == dat.index[-1]:
                    dist_2_gene.append(genome_len - row[4])
                else:
                    dist_2_gene.append(dat.loc[i+1,3] - row[4])
            else:
                if i == dat.index[0]:
                    dist_2_gene.append(row[3])
                else:
                    dist_2_gene.append(row[3] - dat.loc[i-1,4])
    return(dist_2_gene)

def add_p_val(lft,rgt,y,h,p):
    plt.plot([lft, lft, rgt, rgt], [y, y+h, y+h, y], lw=1.5, c='k')
    plt.text((lft + rgt) * .5, y+h, ('n.s.' if p > 0.15 else 'p < %.2g' if p > 0.001 else 'p < %.1g') % max(p+1e-20, 1e-20), ha='center', va='bottom', color='k')

def plot_line_groups(mat,tpm_g,png_name,fig_title, hue_order):
    tot = mat.shape[1]
    xlab = [''] * tot
    xlab[0] = f'-{tot//2}'
    xlab[tot-1] =  f'+{tot//2}'
    xlab[tot//2] = 'TSS'
    pdFrame = pd.DataFrame(mat)
    pdFrame = pdFrame.assign(group=tpm_g)
    pdFrame_melt = pd.melt(pdFrame, id_vars = 'group',var_name='pos', value_name='read_cv')
    lines = sns.lineplot(data=pdFrame_melt, x="pos", y="read_cv", hue='group', hue_order=hue_order)
    lines.set_xticks(range(len(set(pdFrame_melt.pos))))
    lines.set_xticklabels(xlab)
    lines.tick_params(bottom=False)
    lines.axvline(x=tot//2, color = 'red')
    lines.set(title=fig_title)
    lines.get_figure().savefig(png_name)
    plt.clf()

with open('sample_lists/hmp2_samples') as f:
    hmp2_samples = [ln for ln in f.read().splitlines() if ln.strip() and not ln.startswith('#')]
hmp2_RNA = {x.split('\t')[0]:x.split('\t')[1] for x in hmp2_samples}

seq_path = _HMP2_FQ
#Read species abundances
mpa_res = pd.read_table(_MPA, comment= '#')
mpa_res = mpa_res[(mpa_res['clade_name'].str.contains("s__"))&(~mpa_res['clade_name'].str.contains("t__"))]
mpa_res.index = [re.match('.*\|([^\|]*)',x).group(1) for x in mpa_res['clade_name']]
mpa_res = mpa_res.drop(columns=['clade_name'])

mpa_res = mpa_res.loc[:,hmp2_RNA.keys()]
mpa_res = mpa_res.loc[((mpa_res > 0.5).sum(axis=1) >= 30),:] # Filter species with atleast 0.5% abundance in atleast 30 samples
mpa_res = mpa_res.where(mpa_res > 0.5, np.nan)
selected_genomes = {'s__Phocaeicola_vulgatus': 'GCF_000012825.1_ASM1282v1',
 's__Agathobacter_rectalis': 'GCF_022453685.1_ASM2245368v1',
 's__Alistipes_putredinis': 'GCF_000154465.1_ASM15446v1',
 's__Faecalibacterium_prausnitzii': 'GCF_000154385.1_ASM15438v1',
 's__Bacteroides_uniformis': 'GCF_044361425.1_ASM4436142v1',
 's__Parabacteroides_distasonis': 'GCF_018279895.1_ASM1827989v1',
 's__Bacteroides_caccae': 'GCF_018292205.1_ASM1829220v1'}

xlab = [''] * 1001
xlab[0] = f'-{1001//2}'
xlab[1001-1] =  f'+{1001//2}'
xlab[1001//2] = 'TSS'

all_df = pd.DataFrame()

for sample in mpa_res.columns:
    RNA_path = seq_path + '/' + hmp2_RNA[sample]
    for species in mpa_res.index:
        if np.isnan(mpa_res.loc[species,sample]):
            continue
        asm = selected_genomes[species]
        
        #Quantify RNA
        if not os.path.exists(f'kallisto_res/{asm}_{sample}'):
            cds_fna = f'refseq_genomes/{asm}/{asm}_prodigal.fna'
            subprocess.call(f'kallisto index {cds_fna} -i refseq_genomes/{asm}/{asm}_kallisto.db', shell = True)
            subprocess.call(f'kallisto quant -t 4 -i refseq_genomes/{asm}/{asm}_kallisto.db -o kallisto_res/{asm}_{sample} {RNA_path}_1.fastq.gz {RNA_path}_2.fastq.gz', shell = True)

        #Read TPM sample data
        tpm_file = get_tpm(f'kallisto_res/{asm}_{sample}/abundance.tsv')
        chrom_map = pd.read_table(f'refseq_genomes/{asm}/{asm}_filtered_tss.csv', sep = ',', header=0, index_col=0)
        chrom_map = chrom_map[~chrom_map.index.duplicated(keep='first')]
        chrom_map = {chrom:re.match(r'ID=(\d*)_',chrom_map.loc[chrom,'Desc']).group(1) for chrom in chrom_map.index}
        tpm_file.index = [x.replace(x.rsplit('_',1)[0],chrom_map.get(x.rsplit('_',1)[0],'na')) for x in tpm_file.index]
        tpm_dict = tpm_file.to_dict()

        mat = pd.read_table(f'matrices/{asm}_{sample}.csv', sep = ',', header=None, index_col=0)
        mat = mat.loc[mat.sum(axis=1)!=0,:]
        tpm_val = [tpm_dict['tpm'].get(x,np.nan) for x in mat.index]
        nan = pd.Series(tpm_val).notnull()
        mat = mat.loc[nan.tolist(),:]
        tpm_val = pd.Series(tpm_val)[nan]
        tpm_val = tpm_val.reset_index(drop=True)
        tpm_color = pd.Series(np.where(tpm_val == 0, "zero", np.where(tpm_val < 100, 'low', np.where(tpm_val < 1000, 'mid', 'high'))))

        ###Get Intergenic distances 
        ### Wrong need to use originall GFF instead of filtered tss to accurately calculate upstream distance.
        tss_dat = pd.read_csv(f'refseq_genomes/{asm}/{asm}_prodigal.gff', comment = '#', sep = '\t', header = None)
        reference_genome = SeqIO.to_dict(SeqIO.parse(f'refseq_genomes/{asm}/{asm}.fna', "fasta"))
        tss_dat['upstream_dist'] = get_upstream_dist(tss_dat,reference_genome)
        tss_dat['operon_state'] = ['short' if x <=50 else 'long' for x in tss_dat['upstream_dist']]
        tss_dat.index = np.array([re.match(r'ID=([^;]*)',x).group(1) for x in tss_dat[8]])

        ### Box/Line plot, group by TPM expression ###
        scaled_mat = scale_mat(mat)
        bias_vec = list(map(calc_bias, np.array(scaled_mat)))
        df = pd.DataFrame({'tpm':tpm_val.tolist(),'bias':bias_vec, 'samp':sample, 'asm':asm,
                           'operon_state':tss_dat.loc[scaled_mat.index,'operon_state'], 'upstream_dist':tss_dat.loc[scaled_mat.index,'upstream_dist']}, index = scaled_mat.index)
        plot_line_groups(scaled_mat,tpm_color.tolist(),f'plots/gene_expression/{asm}_{sample}.pdf',"Read Coverage Mean", ['zero','low','mid','high'])

        #Calculate GC Bias
        tss_dat = tss_dat.loc[scaled_mat.index,:]
        tss_dat_np = tss_dat.to_numpy()
        gc_bias = []
        for i in tss_dat_np:
            genome_len = len(reference_genome[i[0]].seq)
            strand = i[6]
            if strand == '-':
                start = i[4]
            else:
                start = i[3]
            if start <= 501 or start >= genome_len-500:
                continue
            left_perc = 100*len(re.findall("G|C",str(reference_genome[i[0]].seq[start-500:start])))/500
            right_perc = 100*len(re.findall("G|C",str(reference_genome[i[0]].seq[start:start+500])))/500
            if strand == '+': 
                gc_bias.append(right_perc-left_perc)
            else:
                gc_bias.append(left_perc-right_perc)
        df['gc_bias'] = gc_bias

        #Estimate species abundance (total metagenomic reads aligned to genome)
        df['mapped_reads'] = np.sum([int(x.split('\t')[2]) for x in pysam.idxstats(f'bams/{asm}_{sample}.sorted.bam').split('\n')[:-1]])

        all_df = pd.concat([all_df, df])
all_df.to_csv('plots/hmp2_tpm_vs_bias.csv')

all_df = pd.read_csv('plots/hmp2_tpm_vs_bias.csv', index_col = 0)
all_df['species'] =  [list(selected_genomes.keys())[list(selected_genomes.values()).index(x)] for x in all_df['asm']]
#Normalise by abundance (per million metagenomic reads)
all_df['norm_tpm'] = 1000000*all_df['tpm']/all_df['mapped_reads']
all_df['log_norm_tpm'] = np.log(all_df['norm_tpm'] + 0.001)
all_df['tpm_group'] = np.where(all_df['norm_tpm'] == 0, "zero", np.where(all_df['norm_tpm'] < 100, 'low', np.where(all_df['norm_tpm'] < 1000, 'mid', 'high')))
grouped_df = all_df.groupby(['species','samp','tpm_group']).mean().reset_index()

#Fill missing tpm groups
pairs = grouped_df[['samp', 'species']].drop_duplicates()
tpm_groups = grouped_df['tpm_group'].drop_duplicates()
full_index = (
    pairs
    .merge(tpm_groups, how='cross')          # adds tpm_group column
    .set_index(['samp', 'species', 'tpm_group'])
    .index
)
# Reindex and fill missing with 0
grouped_df_completed = (
    grouped_df.set_index(['samp', 'species', 'tpm_group'])
      .reindex(full_index, fill_value=0)
      .reset_index()
)


g = sns.boxplot(x="tpm_group", y="bias", data=grouped_df_completed, order = ['zero','low','mid','high'])
plt.ylim(-250,800)
add_p_val(0,3,700,10,ttest_rel(grouped_df_completed.loc[grouped_df_completed['tpm_group']=='high','bias'],grouped_df_completed.loc[grouped_df_completed['tpm_group']=='zero','bias']).pvalue)
add_p_val(2,3,650,10,ttest_rel(grouped_df_completed.loc[grouped_df_completed['tpm_group']=='mid','bias'],grouped_df_completed.loc[grouped_df_completed['tpm_group']=='high','bias']).pvalue)
add_p_val(1,2,600,10,ttest_rel(grouped_df_completed.loc[grouped_df_completed['tpm_group']=='low','bias'],grouped_df_completed.loc[grouped_df_completed['tpm_group']=='mid','bias']).pvalue)
add_p_val(0,1,550,10,ttest_rel(grouped_df_completed.loc[grouped_df_completed['tpm_group']=='zero','bias'],grouped_df_completed.loc[grouped_df_completed['tpm_group']=='low','bias']).pvalue) 
g.figure.savefig(f"plots/gene_expression/hmp2_bias_vs_tpm_boxplot.pdf")
plt.clf()
kruskal(grouped_df.loc[grouped_df['tpm_group']=='high','bias'].tolist(),grouped_df.loc[grouped_df['tpm_group']=='mid','bias'].tolist(),
      grouped_df.loc[grouped_df['tpm_group']=='low','bias'].tolist(),grouped_df.loc[grouped_df['tpm_group']=='zero','bias'].tolist())
# KruskalResult(statistic=344.12013050951873, pvalue=2.7977879319054566e-74)

#GC content
g = sns.boxplot(x="tpm_group", y="gc_bias", data=grouped_df_completed, order = ['zero','low','mid','high'], showfliers=False)
plt.ylim(-0.5,6.5)
add_p_val(0,3,6,0.05,ttest_rel(grouped_df_completed.loc[grouped_df_completed['tpm_group']=='high','gc_bias'],grouped_df_completed.loc[grouped_df_completed['tpm_group']=='zero','gc_bias']).pvalue)
add_p_val(2,3,5.5,0.05,ttest_rel(grouped_df_completed.loc[grouped_df_completed['tpm_group']=='mid','gc_bias'],grouped_df_completed.loc[grouped_df_completed['tpm_group']=='high','gc_bias']).pvalue)
add_p_val(1,2,5,0.05,ttest_rel(grouped_df_completed.loc[grouped_df_completed['tpm_group']=='low','gc_bias'],grouped_df_completed.loc[grouped_df_completed['tpm_group']=='mid','gc_bias']).pvalue)
add_p_val(0,1,4.5,0.05,ttest_rel(grouped_df_completed.loc[grouped_df_completed['tpm_group']=='zero','gc_bias'],grouped_df_completed.loc[grouped_df_completed['tpm_group']=='low','gc_bias']).pvalue) 
g.figure.savefig(f"plots/gene_expression/hmp2_gc_vs_tpm_boxplot.pdf")
plt.clf()

###Species specific
for spec in list(set(grouped_df_completed.species)):
    df = grouped_df_completed[grouped_df_completed['species']==spec]
    g = sns.boxplot(x="tpm_group", y="bias", data=df, order = ['zero','low','mid','high'])
    plt.ylim(-200,540)
    add_p_val(0,3,450,10,ttest_rel(df.loc[df['tpm_group']=='high','bias'],df.loc[df['tpm_group']=='zero','bias']).pvalue)
    add_p_val(2,3,400,10,ttest_rel(df.loc[df['tpm_group']=='mid','bias'],df.loc[df['tpm_group']=='high','bias']).pvalue)
    add_p_val(1,2,350,10,ttest_rel(df.loc[df['tpm_group']=='low','bias'],df.loc[df['tpm_group']=='mid','bias']).pvalue)
    add_p_val(0,1,300,10,ttest_rel(df.loc[df['tpm_group']=='zero','bias'],df.loc[df['tpm_group']=='low','bias']).pvalue) 
    g.figure.savefig(f"plots/gene_expression/hmp2_{spec}_bias_vs_tpm_boxplot.pdf")
    plt.clf()

    #g = sns.boxplot(x="tpm_group", y="gc_bias", data=df, order = ['zero','low','mid','high'])
    #plt.ylim(-1,5.5)
    #add_p_val(0,3,4,0.05,ttest_rel(df.loc[df['tpm_group']=='high','gc_bias'],df.loc[df['tpm_group']=='zero','gc_bias']).pvalue)
    #add_p_val(2,3,3.6,0.05,ttest_rel(df.loc[df['tpm_group']=='mid','gc_bias'],df.loc[df['tpm_group']=='high','gc_bias']).pvalue)
    #add_p_val(1,2,3.3,0.05,ttest_rel(df.loc[df['tpm_group']=='low','gc_bias'],df.loc[df['tpm_group']=='mid','gc_bias']).pvalue)
    #add_p_val(0,1,3,0.05,ttest_rel(df.loc[df['tpm_group']=='zero','gc_bias'],df.loc[df['tpm_group']=='low','gc_bias']).pvalue) 
    #g.figure.savefig(f"plots/gene_expression/{spec}_gc_vs_tpm_boxplot.pdf")
    #plt.clf()

### TPM Line plots
all_samp_pos_df = pd.DataFrame()
for samp in mpa_res.columns:
    for species in mpa_res.index:
        if np.isnan(mpa_res.loc[species,samp]):
            continue
        asm = selected_genomes[species]
        # Get bp read coverge data
        mat = pd.read_table(f'matrices/{asm}_{samp}.csv', sep = ',', header=None, index_col=0)
        mat = mat.loc[mat.sum(axis=1)!=0,:]
        scaled_mat = scale_mat(mat)
        scaled_mat['tpm_group'] = all_df[(all_df['asm']==asm)&(all_df['samp']==samp)]['tpm_group']
        pdFrame_melt = pd.melt(scaled_mat, id_vars = 'tpm_group',var_name='pos', value_name='read_cv')
        pdFrame_melt['species'] = species
        pdFrame_melt = pdFrame_melt.groupby(['species','tpm_group','pos']).mean().reset_index()
        pdFrame_melt['sample'] = samp
        all_samp_pos_df = pd.concat([all_samp_pos_df,pdFrame_melt])           
all_samp_pos_df.to_csv('plots/hmp2_tpm_vs_bias_lineplot.csv')

lines = sns.lineplot(data=all_samp_pos_df, x="pos", y="read_cv", hue='tpm_group', hue_order=['zero','low','mid','high'])
lines.set_xticks(range(len(set(all_samp_pos_df.pos))))
lines.set_xticklabels(xlab)
lines.tick_params(bottom=False)
lines.axvline(x=1001//2, color = 'red')
lines.set(title="Species Average")
lines.get_figure().savefig(f'plots/gene_expression/hmp2_species_lineplot.pdf')
plt.clf()

#Species-specific
for spe in list(set(all_samp_pos_df.species)):
    df = all_samp_pos_df[all_samp_pos_df['species']==spe]
    lines = sns.lineplot(data=df, x="pos", y="read_cv", hue='tpm_group', hue_order=['zero','low','mid','high'])
    lines.set_xticks(range(len(set(df.pos))))
    lines.set_xticklabels(xlab)
    lines.tick_params(bottom=False)
    lines.axvline(x=1001//2, color = 'red')
    lines.set(title=spe)
    lines.get_figure().savefig(f'plots/gene_expression/hmp2_{spe}_lineplot.pdf')
    plt.clf()
'''