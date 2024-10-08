import pandas as pd
import re
import numpy as np
from pysam import FastaFile
import pysam
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from scipy.stats import kruskal

import pyBigWig
import os, sys
from glob import glob
import numpy as np
from tqdm import tqdm
from scipy import stats
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind



SMALL_BUFFER = 5

def asm_2_species(asm):
    name = refseq[refseq['ftp_path'].str.contains(asm)]['organism_name'].item()
    name = re.match(r'(^[^ ]+ [^ ]+).*',name).group(1)
    if name == '[Eubacterium] eligens':
        name = 'Lachnospira eligens'
    return name

def taxid2asm(taxid,n=1):
    strain = refseq[(refseq['species_taxid']==int(taxid)) & (refseq['assembly_level']=='Complete Genome')]
    if len(strain) == 0:
        strain = refseq[(refseq['species_taxid']==int(taxid))]
    ftp = strain['ftp_path'][:n].tolist()
    asm = [x.rsplit('/',1)[1] for x in ftp]
    return(asm)

def adj_ratio(a,b):
    return (a+1)/(b+1)

def calc_gene_bias(annot,aln_file, read_len = 151, window=50, offset=0):
    inward_reads = []
    outward_reads = []
    for gene in annot:
        inward = 0
        outward = 0
        if gene[3] == '+':
            start = int(gene[5]) + offset - SMALL_BUFFER
            stop = int(gene[5]) + window + offset
            for read in aln_file.fetch(gene[0],start,stop):
                if read.mate_is_unmapped or read.is_unmapped or read.reference_start < start or read.reference_start > stop:
                    continue
                if read.is_reverse:
                    outward += 1
                else:
                    inward += 1
        else:
            stop = int(gene[5]) - read_len - offset + SMALL_BUFFER
            start = int(gene[5]) - read_len - window - offset
            for read in aln_file.fetch(gene[0],start,stop):
                if read.mate_is_unmapped or read.is_unmapped or read.reference_start < start or read.reference_start > stop:
                    continue
                if read.is_reverse:
                    inward += 1
                else:
                    outward += 1
        inward_reads.append(inward)
        outward_reads.append(outward)
    return inward_reads, outward_reads

def calc_gene_bias_single(annot,aln_file, read_len = 151, window=50, offset=0):
    inward_reads = []
    outward_reads = []
    for gene in annot:
        inward = 0
        outward = 0
        if gene[3] == '+':
            start = int(gene[5]) + offset - SMALL_BUFFER
            stop = int(gene[5]) + window + offset
            for read in aln_file.fetch(gene[0],start,stop):
                if read.is_unmapped or read.reference_start < start or read.reference_start > stop:
                    continue
                if read.is_reverse:
                    outward += 1
                else:
                    inward += 1
        else:
            stop = int(gene[5]) - read_len - offset + SMALL_BUFFER
            start = int(gene[5]) - read_len - window - offset
            for read in aln_file.fetch(gene[0],start,stop):
                if read.is_unmapped or read.reference_start < start or read.reference_start > stop:
                    continue
                if read.is_reverse :
                    inward += 1
                else:
                    outward += 1
        inward_reads.append(inward)
        outward_reads.append(outward)
    return inward_reads, outward_reads

def get_offset_diff(sample,asm,gff):
    # Load alignments
    aln_file = pysam.AlignmentFile(f'bams/{asm}_{sample}.sorted.bam', "rb")
    #Get read length
    for read in aln_file.fetch():
        read_len = len(read.get_forward_sequence())
        break
    #calc inward vs outward
    inward_offset_0, outward_offset_0 = calc_gene_bias(np.array(gff),aln_file,read_len=read_len,window=50,offset=0)
    inward_offset_100, outward_offset_100 = calc_gene_bias(np.array(gff),aln_file,read_len=read_len,window=50,offset=100)
    inward_offset_200, outward_offset_200 = calc_gene_bias(np.array(gff),aln_file,read_len=read_len,window=50,offset=200)
    inward_offset_300, outward_offset_300 = calc_gene_bias(np.array(gff),aln_file,read_len=read_len,window=50,offset=300)
    inward_offset_400, outward_offset_400 = calc_gene_bias(np.array(gff),aln_file,read_len=read_len,window=50,offset=400)
    inward_offset_0_s, outward_offset_0_s = calc_gene_bias_single(np.array(gff),aln_file,read_len=read_len,window=50,offset=0)
    inward_offset_100_s, outward_offset_100_s = calc_gene_bias_single(np.array(gff),aln_file,read_len=read_len,window=50,offset=100)
    inward_offset_200_s, outward_offset_200_s = calc_gene_bias_single(np.array(gff),aln_file,read_len=read_len,window=50,offset=200)
    inward_offset_300_s, outward_offset_300_s = calc_gene_bias_single(np.array(gff),aln_file,read_len=read_len,window=50,offset=300)
    inward_offset_400_s, outward_offset_400_s = calc_gene_bias_single(np.array(gff),aln_file,read_len=read_len,window=50,offset=400)
    
    ###The ratios
    diff_offest_0 = [adj_ratio(a,b) for a, b in zip(inward_offset_0, outward_offset_0)]
    diff_offest_100 = [adj_ratio(a,b) for a, b in zip(inward_offset_100, outward_offset_100)]
    diff_offest_200 = [adj_ratio(a,b) for a, b in zip(inward_offset_200, outward_offset_200)]
    diff_offest_300 = [adj_ratio(a,b) for a, b in zip(inward_offset_300, outward_offset_300)]
    diff_offest_400 = [adj_ratio(a,b) for a, b in zip(inward_offset_400, outward_offset_400)]
    diff_offest_0_s = [adj_ratio(a,b) for a, b in zip(inward_offset_0, outward_offset_0_s)]
    diff_offest_100_s = [adj_ratio(a,b) for a, b in zip(inward_offset_100_s, outward_offset_100_s)]
    diff_offest_200_s = [adj_ratio(a,b) for a, b in zip(inward_offset_200_s, outward_offset_200_s)]
    diff_offest_300_s = [adj_ratio(a,b) for a, b in zip(inward_offset_300_s, outward_offset_300_s)]
    diff_offest_400_s = [adj_ratio(a,b) for a, b in zip(inward_offset_400_s, outward_offset_400_s)]

    dat_diff = pd.DataFrame({
        "diff_offest_0": diff_offest_0, "diff_offest_0_s": diff_offest_0_s,
        "diff_offest_100": diff_offest_100, "diff_offest_100_s": diff_offest_100_s,
        "diff_offest_200": diff_offest_200, "diff_offest_200_s": diff_offest_200_s,
        "diff_offest_300": diff_offest_300, "diff_offest_300_s": diff_offest_300_s,
        "diff_offest_400": diff_offest_400, "diff_offest_400_s": diff_offest_400_s
    })
    dat_diff['sample'] = sample
    dat_diff['gene'] = gff['gene'].tolist()
    return(dat_diff)

def get_tpm(file):
    dat = pd.read_table(file)
    dat = dat.drop_duplicates(subset='target_id')
    dat = dat[['target_id','tpm']]
    dat.index = dat['target_id']
    dat = dat.drop(columns = ['target_id'])
    return dat

def exp_group(val):
    if val == 0:
        return 'zero'
    if val >= 0 and val < 100:
        return 'low'
    if val >= 100 and val < 1000:
        return 'mid'
    if val >= 1000:
        return 'high'
    return np.nan

def get_tpm_diff(sample,asm,gff):
    # Load alignments
    aln_file = pysam.AlignmentFile(f'bams/{asm}_{sample}.sorted.bam', "rb")
    #Get read length
    for read in aln_file.fetch():
        read_len = len(read.get_forward_sequence())
        break
    # Load gene expression
    tpm_file = get_tpm(f'kallisto_res/{asm}_{sample}/abundance.tsv')
    chrom_map = pd.read_table(f'strains/{asm}/{asm}_filtered_tss.csv', sep = ',', header=0, index_col=0)
    chrom_map = gff[~gff.Genome.duplicated(keep='first')]
    chrom_map = {chrom:re.match(r'ID=(\d*)_',chrom_map.loc[chrom_map.Genome==chrom,'Desc'].item()).group(1) for chrom in chrom_map.Genome}
    tpm_file.index = [x.replace(x.rsplit('_',1)[0],chrom_map.get(x.rsplit('_',1)[0],'na')) for x in tpm_file.index]
    tpm_dict = tpm_file.to_dict()
    total_mapped_reads = np.sum([int(x.split('\t')[2]) for x in pysam.idxstats(f'bams/{asm}_{sample}.sorted.bam').split('\n')[:-1]])
    gff['tpm_val'] = [tpm_dict['tpm'].get(x,np.nan) for x in gff.gene]
    gff['norm_tpm'] = 1000000*gff['tpm_val']/total_mapped_reads
    gff['tpm_group'] = [exp_group(x) for x in gff.norm_tpm]

    gff['inward'],gff['outward'] = calc_gene_bias(np.array(gff),aln_file,read_len=read_len,window=50,offset=0)
    gff['inward_single'],gff['outward_single'] = calc_gene_bias_single(np.array(gff),aln_file,read_len=read_len,window=50,offset=0)
    gff['paired'] = [adj_ratio(a,b) for a, b in zip(gff['inward'], gff['outward'])]
    gff['singleton'] = [adj_ratio(a,b) for a, b in zip(gff['inward_single'], gff['outward_single'])]
    gff['sample'] = sample
    #return(gff[['tpm_group','sample','paired','singleton']].groupby(['tpm_group','sample']).mean().reset_index())
    return(gff[['gene','tpm_val','norm_tpm','tpm_group','sample','paired','singleton']])

def add_p_val(ax,lft,rgt,y,h,p):
    ax.plot([lft, lft, rgt, rgt], [y, y+h, y+h, y], lw=1.5, c='k')
    ax.text((lft + rgt) * .5, y+h, ('n.s.' if p > 0.15 else 'p < %.2g' if p > 0.001 else 'p < %.1g') % max(p+1e-20, 1e-20), ha='center', va='bottom', color='k')
    
def get_upstream_dist(tss_dat,ref):
    dist_2_gene = []
    for genome in list(set(tss_dat.Genome)):
        dat = tss_dat[tss_dat.Genome==genome]
        genome_len = reference_genome.get_reference_length(genome)
        for i, row in dat.iterrows():
            if row['Strand'] == '-':
                if i == dat.index[-1]:
                    dist_2_gene.append(genome_len - row['Start'])
                    continue
                dist_2_gene.append(dat.loc[i+1,'geneLeft'] - row['Start'])
            else:
                if i == dat.index[0]:
                    dist_2_gene.append(row['Start'])
                    continue
                dist_2_gene.append(row['Start'] - dat.loc[i-1,'geneRight'])
    return(dist_2_gene)

with open('Franzosa_8_samples') as f:
    samples = f.read().splitlines()
samples = [x.split('\t',1)[0] for x in samples]

#Read species abundances
mpa_res = pd.read_table('metaphlan3_res/merged_mpa.txt', comment= '#')
mpa_res = mpa_res[mpa_res['clade_name'].str.contains("s__")]
mpa_res.index = [re.match('.*\|([^\|]*)',x).group(1) for x in mpa_res['NCBI_tax_id']]
mpa_res = mpa_res.drop(columns=['clade_name', 'NCBI_tax_id'])
mpa_res = mpa_res.loc[:,samples]
mpa_res = mpa_res.loc[((mpa_res > 1).sum(axis=1) >= 5),:] # Filter species with atleast 1% abundance in atleast 5 samples
mpa_res[mpa_res < 1] = np.nan
mpa_res = mpa_res.drop('39491') #Eubacterium Rectale no reference in database.

refseq = pd.read_table('src/assembly_summary_refseq.txt', header = 1)

all_strains = []
for taxid in mpa_res.index:
    asms = taxid2asm(taxid,3)
    all_strains = all_strains + asms
asm_2_species_dict = {x:asm_2_species(x) for x in all_strains}

all_offset = pd.DataFrame()
all_exp = pd.DataFrame()
for taxid in mpa_res.index:
    strains = taxid2asm(taxid,3)
    for asm in strains:
        # Load gene regions
        tss_dat = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
        tss_dat['gene'] = np.array([re.match(r'ID=([^;]*)',x).group(1) for x in tss_dat['Desc']])
        # Filter short genes and edge genes
        tss_dat['len'] = tss_dat['geneRight']-tss_dat['geneLeft']
        tss_dat = tss_dat[tss_dat['len'] > 500]
        tss_dat = tss_dat[(tss_dat.Start > 750)] #Incase of fetching beyond start of genome

        samples = mpa_res.columns[mpa_res.loc[taxid,].notna()].to_list()
        offset_res = pd.concat([get_offset_diff(samp,asm,tss_dat) for samp in samples])
        exp_res = pd.concat([get_tpm_diff(samp,asm,tss_dat) for samp in samples])

        offset_melt = offset_res.melt(id_vars=['sample','gene'])
        offset_melt['mapping_type'] = ['singleton' if '_s' in x else 'pair' for x in offset_melt.variable]
        offset_melt['offset'] = [re.match('.*_(\d+).*',x).group(1) for x in offset_melt.variable]
        offset_melt['asm'] = asm
        all_offset = pd.concat([all_offset,offset_melt])

        exp_melt = exp_res.melt(id_vars=['tpm_group', 'sample'], ignore_index=False, var_name = 'mapping_type')
        exp_res['asm'] = asm
        all_exp = pd.concat([all_exp,exp_res])
    
all_offset.to_csv('plots/all_offset_genelevel.csv')
all_exp.to_csv('plots/all_exp_genelevel.csv')

######################################################################################################################
all_offset = pd.read_csv('plots/all_offset_genelevel.csv', index_col = 0)
all_offset['species'] = all_offset['asm'].map(asm_2_species_dict)
offset_melt = all_offset.groupby(['sample','species','mapping_type','offset']).mean().reset_index()

offset_melt_pair = offset_melt[offset_melt.mapping_type=='pair']
upper_lim = 1.65
sns.boxplot(data = offset_melt_pair, x = 'offset', y= 'value', showfliers=False, color = 'tab:blue')
add_p_val(plt,0,1,upper_lim,0.01,ttest_rel(offset_melt_pair.loc[(offset_melt_pair.offset==0),'value'],offset_melt_pair.loc[(offset_melt_pair.offset==100),'value']).pvalue) # Ttest_relResult(statistic=15.92425141695283, pvalue=2.434158426018579e-20)
plt.savefig(f"plots/paired_end_analysis/all_species_sample_offset_pair.pdf")
plt.clf()
offset_melt_pair[offset_melt_pair.offset==0]['value'].mean() # 1.4016040708594473
offset_melt_pair[offset_melt_pair.offset==0]['value'].std() # 0.09036779330817589
offset_melt_pair[offset_melt_pair.offset==100]['value'].mean() # 1.1555676233829497
offset_melt_pair[offset_melt_pair.offset==100]['value'].std() # 0.058435576080828534

offset_melt_single = offset_melt[offset_melt.mapping_type=='singleton']
upper_lim = 1.55
sns.boxplot(data = offset_melt_single, x = 'offset', y= 'value', showfliers=False, color = 'tab:orange')
add_p_val(plt,0,1,upper_lim,0.01,ttest_rel(offset_melt_single.loc[(offset_melt_single.offset==0),'value'],offset_melt_single.loc[(offset_melt_single.offset==100),'value']).pvalue) # Ttest_relResult(statistic=9.06398711323789, pvalue=8.419065951920693e-12)
plt.savefig(f"plots/paired_end_analysis/all_species_sample_offset_single.pdf")
plt.clf()
offset_melt_single[offset_melt_single.offset==0]['value'].mean() # 1.2839737005467138
offset_melt_single[offset_melt_single.offset==0]['value'].std() # 0.09015220703741732
offset_melt_single[offset_melt_single.offset==100]['value'].mean() # 1.144519656478592
offset_melt_single[offset_melt_single.offset==100]['value'].std() # 0.058684472045713906


all_exp = pd.read_csv('plots/all_exp_genelevel.csv', index_col = 0)
all_exp['species'] = all_exp['asm'].map(asm_2_species_dict)
all_exp = all_exp[['sample','species','tpm_group','paired','singleton']].melt(id_vars = ['sample','species','tpm_group'], var_name = 'mapping_type')
exp_melt = all_exp.groupby(['sample','species','tpm_group','mapping_type']).mean().reset_index()

exp_melt_pair = exp_melt[exp_melt.mapping_type=='paired']
upper_lim = 2.4
sns.boxplot(data = exp_melt_pair, x = 'tpm_group', y= 'value', showfliers=False, color = 'tab:blue', order = ['high','mid','low','zero'])
add_p_val(plt,0,1,upper_lim,0.01,ttest_rel(exp_melt_pair.loc[(exp_melt_pair.tpm_group=='high'),'value'],exp_melt_pair.loc[(exp_melt_pair.tpm_group=='mid'),'value']).pvalue)
add_p_val(plt,1,2,upper_lim-0.2,0.01,ttest_rel(exp_melt_pair.loc[(exp_melt_pair.tpm_group=='mid'),'value'],exp_melt_pair.loc[(exp_melt_pair.tpm_group=='low'),'value']).pvalue)
add_p_val(plt,2,3,upper_lim-0.4,0.01,ttest_rel(exp_melt_pair.loc[(exp_melt_pair.tpm_group=='low'),'value'],exp_melt_pair.loc[(exp_melt_pair.tpm_group=='zero'),'value']).pvalue)
plt.savefig(f"plots/paired_end_analysis/all_species_sample_exp_pair.pdf")
plt.clf()
kruskal(exp_melt_pair.loc[(exp_melt_pair.tpm_group=='zero'),'value'],exp_melt_pair.loc[(exp_melt_pair.tpm_group=='low'),'value'],
       exp_melt_pair.loc[(exp_melt_pair.tpm_group=='mid'),'value'],exp_melt_pair.loc[(exp_melt_pair.tpm_group=='high'),'value'])
# KruskalResult(statistic=91.20082347108155, pvalue=1.2095344821319948e-19)

upper_lim = 1.7
sns.boxplot(data = offset_melt, x = 'offset', y= 'value', hue = 'mapping_type', showfliers=False)
add_p_val(plt,-0.2,0.2,upper_lim,0.01,ttest_rel(offset_melt.loc[(offset_melt.offset==0)&(offset_melt.mapping_type=='pair'),'value'],offset_melt.loc[(offset_melt.offset==0)&(offset_melt.mapping_type=='singleton'),'value']).pvalue) # Ttest_relResult(statistic=12.806207974487961, pvalue=1.4730674913063753e-17)
add_p_val(plt,0.8,1.2,upper_lim,0.01,ttest_rel(offset_melt.loc[(offset_melt.offset==100)&(offset_melt.mapping_type=='pair'),'value'],offset_melt.loc[(offset_melt.offset==100)&(offset_melt.mapping_type=='singleton'),'value']).pvalue)
add_p_val(plt,1.8,2.2,upper_lim,0.01,ttest_rel(offset_melt.loc[(offset_melt.offset==200)&(offset_melt.mapping_type=='pair'),'value'],offset_melt.loc[(offset_melt.offset==200)&(offset_melt.mapping_type=='singleton'),'value']).pvalue)
add_p_val(plt,2.8,3.2,upper_lim,0.01,ttest_rel(offset_melt.loc[(offset_melt.offset==300)&(offset_melt.mapping_type=='pair'),'value'],offset_melt.loc[(offset_melt.offset==300)&(offset_melt.mapping_type=='singleton'),'value']).pvalue)
add_p_val(plt,3.8,4.2,upper_lim,0.01,ttest_rel(offset_melt.loc[(offset_melt.offset==400)&(offset_melt.mapping_type=='pair'),'value'],offset_melt.loc[(offset_melt.offset==400)&(offset_melt.mapping_type=='singleton'),'value']).pvalue)
plt.savefig(f"plots/paired_end_analysis/all_species_sample_offset.pdf")
plt.clf()

offset_melt[(offset_melt.mapping_type=='singleton')&(offset_melt.offset==0)]['value'].mean() # 1.2839737005467138
offset_melt[(offset_melt.mapping_type=='singleton')&(offset_melt.offset==0)]['value'].std() # 0.09015220703741732
offset_melt[(offset_melt.mapping_type=='singleton')&(offset_melt.offset!=0)]['value'].mean() # 1.1618006785428159
offset_melt[(offset_melt.mapping_type=='singleton')&(offset_melt.offset!=0)]['value'].std() # 0.05450214740160428

'''
upper_lim = 1.6
for spe in list(set(offset_melt.species)):
    dat = offset_melt[offset_melt.species==spe]
    sns.boxplot(data = dat, x = 'offset', y= 'value', hue = 'mapping_type', showfliers=False)
    add_p_val(plt,-0.2,0.2,upper_lim,0.01,ttest_rel(dat.loc[(dat.offset==0)&(dat.mapping_type=='pair'),'value'],dat.loc[(dat.offset==0)&(dat.mapping_type=='singleton'),'value']).pvalue)
    add_p_val(plt,0.8,1.2,upper_lim,0.01,ttest_rel(dat.loc[(dat.offset==100)&(dat.mapping_type=='pair'),'value'],dat.loc[(dat.offset==100)&(dat.mapping_type=='singleton'),'value']).pvalue)
    add_p_val(plt,1.8,2.2,upper_lim,0.01,ttest_rel(dat.loc[(dat.offset==200)&(dat.mapping_type=='pair'),'value'],dat.loc[(dat.offset==200)&(dat.mapping_type=='singleton'),'value']).pvalue)
    add_p_val(plt,2.8,3.2,upper_lim,0.01,ttest_rel(dat.loc[(dat.offset==300)&(dat.mapping_type=='pair'),'value'],dat.loc[(dat.offset==300)&(dat.mapping_type=='singleton'),'value']).pvalue)
    add_p_val(plt,3.8,4.2,upper_lim,0.01,ttest_rel(dat.loc[(dat.offset==400)&(dat.mapping_type=='pair'),'value'],dat.loc[(dat.offset==400)&(dat.mapping_type=='singleton'),'value']).pvalue)
    plt.savefig(f"plots/paired_end_analysis/{spe}_all_sample_offset.pdf")
    plt.clf()
'''
upper_lim = 2.35
sns.boxplot(data = exp_melt, x = 'tpm_group', y= 'value', hue = 'mapping_type', order = ['high','mid','low','zero'], showfliers=False)
add_p_val(plt,-0.2,0.2,upper_lim,0.01,ttest_rel(exp_melt.loc[(exp_melt.tpm_group=='high')&(exp_melt.mapping_type=='paired'),'value'],exp_melt.loc[(exp_melt.tpm_group=='high')&(exp_melt.mapping_type=='singleton'),'value']).pvalue)
add_p_val(plt,0.8,1.2,upper_lim,0.01,ttest_rel(exp_melt.loc[(exp_melt.tpm_group=='mid')&(exp_melt.mapping_type=='paired'),'value'],exp_melt.loc[(exp_melt.tpm_group=='mid')&(exp_melt.mapping_type=='singleton'),'value']).pvalue)
add_p_val(plt,1.8,2.2,upper_lim,0.01,ttest_rel(exp_melt.loc[(exp_melt.tpm_group=='low')&(exp_melt.mapping_type=='paired'),'value'],exp_melt.loc[(exp_melt.tpm_group=='low')&(exp_melt.mapping_type=='singleton'),'value']).pvalue)
add_p_val(plt,2.8,3.2,upper_lim,0.01,ttest_rel(exp_melt.loc[(exp_melt.tpm_group=='zero')&(exp_melt.mapping_type=='paired'),'value'],exp_melt.loc[(exp_melt.tpm_group=='zero')&(exp_melt.mapping_type=='singleton'),'value']).pvalue)
plt.savefig(f"plots/paired_end_analysis/all_species_sample_exp.pdf")
plt.clf()

'''
for spe in list(set(exp_melt.species)):
    dat = exp_melt[exp_melt.species==spe]
    print(len(dat))
    upper_lim = dat.value.max()
    sns.boxplot(data = dat, x = 'tpm_group', y= 'value', hue = 'mapping_type', order = ['high','mid','low','zero'], showfliers=False)
    add_p_val(plt,-0.2,0.2,upper_lim,0.01,ttest_rel(dat.loc[(dat.tpm_group=='high')&(dat.mapping_type=='paired'),'value'],dat.loc[(dat.tpm_group=='high')&(dat.mapping_type=='singleton'),'value']).pvalue)
    add_p_val(plt,0.8,1.2,upper_lim,0.01,ttest_rel(dat.loc[(dat.tpm_group=='mid')&(dat.mapping_type=='paired'),'value'],dat.loc[(dat.tpm_group=='mid')&(dat.mapping_type=='singleton'),'value']).pvalue)
    add_p_val(plt,1.8,2.2,upper_lim,0.01,ttest_rel(dat.loc[(dat.tpm_group=='low')&(dat.mapping_type=='paired'),'value'],dat.loc[(dat.tpm_group=='low')&(dat.mapping_type=='singleton'),'value']).pvalue)
    add_p_val(plt,2.8,3.2,upper_lim,0.01,ttest_rel(dat.loc[(dat.tpm_group=='zero')&(dat.mapping_type=='paired'),'value'],dat.loc[(dat.tpm_group=='zero')&(dat.mapping_type=='singleton'),'value']).pvalue)
    plt.savefig(f"plots/paired_end_analysis/{spe}_all_sample_exp.pdf")
    plt.clf()
'''

exp_melt_single = exp_melt[exp_melt.mapping_type=='singleton']
upper_lim = 2
sns.boxplot(data = exp_melt_single, x = 'tpm_group', y= 'value', showfliers=False, color = 'tab:orange', order = ['high','mid','low','zero'])
add_p_val(plt,0,1,upper_lim,0.01,ttest_rel(exp_melt_single.loc[(exp_melt_single.tpm_group=='high'),'value'],exp_melt_single.loc[(exp_melt_single.tpm_group=='mid'),'value']).pvalue)
add_p_val(plt,1,2,upper_lim-0.1,0.01,ttest_rel(exp_melt_single.loc[(exp_melt_single.tpm_group=='mid'),'value'],exp_melt_single.loc[(exp_melt_single.tpm_group=='low'),'value']).pvalue)
add_p_val(plt,2,3,upper_lim-0.3,0.01,ttest_rel(exp_melt_single.loc[(exp_melt_single.tpm_group=='low'),'value'],exp_melt_single.loc[(exp_melt_single.tpm_group=='zero'),'value']).pvalue)
plt.savefig(f"plots/paired_end_analysis/all_species_sample_exp_single.pdf")
plt.clf()
kruskal(exp_melt_single.loc[(exp_melt_single.tpm_group=='zero'),'value'],exp_melt_single.loc[(exp_melt_single.tpm_group=='low'),'value'],
       exp_melt_single.loc[(exp_melt_single.tpm_group=='mid'),'value'],exp_melt_single.loc[(exp_melt_single.tpm_group=='high'),'value'])
#KruskalResult(statistic=74.74114553019035, pvalue=4.117036198194107e-16)

'''
for spe in list(set(exp_melt_single.species)):
    dat = exp_melt_single[(exp_melt_single.species==spe)&(exp_melt_single.mapping_type=='singleton')]
    upper_lim = 1.8
    sns.boxplot(data = dat, x = 'tpm_group', y= 'value', order = ['high','mid','low','zero'], showfliers=False, color = 'tab:orange')
    add_p_val(plt,0,1,upper_lim,0.01,ttest_rel(dat.loc[dat.tpm_group=='high','value'],dat.loc[dat.tpm_group=='mid','value']).pvalue)
    add_p_val(plt,1,2,upper_lim,0.01,ttest_rel(dat.loc[dat.tpm_group=='mid','value'],dat.loc[dat.tpm_group=='low','value']).pvalue)
    add_p_val(plt,2,3,upper_lim,0.01,ttest_rel(dat.loc[dat.tpm_group=='low','value'],dat.loc[dat.tpm_group=='zero','value']).pvalue)
    plt.savefig(f"plots/paired_end_analysis/{spe}_all_sample_exp_single.pdf")
    plt.clf()

for spe in list(set(exp_melt.species)):
    dat = exp_melt_pair[(exp_melt.species==spe)&(exp_melt.mapping_type=='paired')]
    upper_lim = 1.8
    sns.boxplot(data = dat, x = 'tpm_group', y= 'value', order = ['high','mid','low','zero'], showfliers=False, color = 'tab:blue')
    add_p_val(plt,0,1,upper_lim,0.01,ttest_rel(dat.loc[dat.tpm_group=='high','value'],dat.loc[dat.tpm_group=='mid','value']).pvalue)
    add_p_val(plt,1,2,upper_lim,0.01,ttest_rel(dat.loc[dat.tpm_group=='mid','value'],dat.loc[dat.tpm_group=='low','value']).pvalue)
    add_p_val(plt,2,3,upper_lim,0.01,ttest_rel(dat.loc[dat.tpm_group=='low','value'],dat.loc[dat.tpm_group=='zero','value']).pvalue)
    plt.savefig(f"plots/paired_end_analysis/{spe}_all_sample_exp_pair.pdf")
    plt.clf()
'''


### Difference between Paired and single vs. expression ###
all_exp = pd.read_csv('plots/all_exp_genelevel.csv', index_col = 0)
all_exp['diff'] = all_exp['paired'] - all_exp['singleton']
all_exp['species'] = all_exp['asm'].map(asm_2_species_dict)
exp_melt = all_exp[['sample','species','tpm_group','diff']].groupby(['sample','species','tpm_group']).mean().reset_index()

upper_lim = 0.6
sns.boxplot(data = exp_melt, x = 'tpm_group', y= 'diff', showfliers=False, color = 'tab:red', order = ['high','mid','low','zero'])
add_p_val(plt,0,1,upper_lim,0.01,ttest_rel(exp_melt.loc[(exp_melt.tpm_group=='high'),'diff'],exp_melt.loc[(exp_melt.tpm_group=='mid'),'diff']).pvalue)
add_p_val(plt,1,2,upper_lim-0.15,0.01,ttest_rel(exp_melt.loc[(exp_melt.tpm_group=='mid'),'diff'],exp_melt.loc[(exp_melt.tpm_group=='low'),'diff']).pvalue)
add_p_val(plt,2,3,upper_lim-0.3,0.01,ttest_rel(exp_melt.loc[(exp_melt.tpm_group=='low'),'diff'],exp_melt.loc[(exp_melt.tpm_group=='zero'),'diff']).pvalue)
add_p_val(plt,0,3,upper_lim+0.1,0.01,ttest_rel(exp_melt.loc[(exp_melt.tpm_group=='high'),'diff'],exp_melt.loc[(exp_melt.tpm_group=='zero'),'diff']).pvalue)
plt.savefig(f"plots/paired_end_analysis/all_species_sample_exp_diff.pdf")
plt.clf()
kruskal(exp_melt.loc[(exp_melt.tpm_group=='zero'),'diff'],exp_melt.loc[(exp_melt.tpm_group=='low'),'diff'],
       exp_melt.loc[(exp_melt.tpm_group=='mid'),'diff'],exp_melt.loc[(exp_melt.tpm_group=='high'),'diff'])
#KruskalResult(statistic=17.60902369096118, pvalue=0.0005295380805724416)

'''
for spe in list(set(exp_melt.species)):
    dat = exp_melt[(exp_melt.species==spe)]
    upper_lim = dat['diff'].max()
    sns.boxplot(data = dat, x = 'tpm_group', y= 'diff', color = 'tab:red', order = ['high','mid','low','zero'], showfliers=False)
    add_p_val(plt,0,1,upper_lim,0.01,ttest_rel(dat.loc[(dat.tpm_group=='high'),'diff'],dat.loc[(dat.tpm_group=='mid'),'diff']).pvalue)
    add_p_val(plt,1,2,upper_lim,0.01,ttest_rel(dat.loc[(dat.tpm_group=='mid'),'diff'],dat.loc[(dat.tpm_group=='low'),'diff']).pvalue)
    add_p_val(plt,2,3,upper_lim,0.01,ttest_rel(dat.loc[(dat.tpm_group=='low'),'diff'],dat.loc[(dat.tpm_group=='zero'),'diff']).pvalue)
    plt.savefig(f"plots/paired_end_analysis/{spe}_all_sample_exp_diff.pdf")
    plt.clf()
'''


