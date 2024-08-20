import pandas as pd
import re
import numpy as np
import pysam
from pysam import FastaFile
from align.calign import aligner
from align.matrix import DNAFULL
from Bio.Seq import Seq
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.stats.multitest as multi

SMALL_BUFFER = 5

def taxid2asm(taxid,n=1):
    strain = refseq[(refseq['species_taxid']==int(taxid)) & (refseq['assembly_level']=='Complete Genome')]
    if len(strain) == 0:
        strain = refseq[(refseq['species_taxid']==int(taxid))]
    ftp = strain['ftp_path'][:n].tolist()
    asm = [x.rsplit('/',1)[1] for x in ftp]
    return(asm)

def count_unmapped(gene,aln_file, window=50, read_len = 151):
    count = 0
    if gene[3] == '+':
        start = int(gene[5]) - SMALL_BUFFER
        stop = int(gene[5]) + window
        for read in aln_file.fetch(gene[0],start,stop):
            if read.reference_start < start or read.reference_start > stop:
                continue
            if read.is_reverse and not read.is_unmapped and read.mate_is_unmapped:
                count+=1
    else:
        stop = int(gene[5]) - read_len + SMALL_BUFFER
        start = int(gene[5]) - read_len - window
        for read in aln_file.fetch(gene[0],start,stop):
            if read.reference_start < start or read.reference_start > stop:
                continue
            if not read.is_reverse and not read.is_unmapped and read.mate_is_unmapped:
                count+=1
    return count

def calc_upstream_diversity(gene,aln_file,read_index,window=50, read_len=151):
    align_score = []
    if gene[3] == '+':
        start = int(gene[5]) - SMALL_BUFFER
        stop = int(gene[5]) + window
        genome_seq = reference_genome.fetch(gene[0],int(gene[5])-300,int(gene[5])+200)
        for read in aln_file.fetch(gene[0],start,stop):
            if read.reference_start < start or read.reference_start > stop:
                continue
            if read.is_reverse and not read.is_unmapped:
                if not read.mate_is_unmapped:
                    read_seq = aln_file.mate(read).seq
                else:
                    for pair in read_index.find(read.query_name):
                        if pair.is_unmapped:
                            read_seq = pair.seq #Because mapped read is in reverse direction, unmapped read should be forward, thus get original direction sequence
                align_score.append(aligner(read_seq, genome_seq, method='glocal', max_hits=1, matrix=DNAFULL)[0][-1])
    else:
        stop = int(gene[5]) - read_len + SMALL_BUFFER
        start = int(gene[5]) - read_len - window
        genome_seq = reference_genome.fetch(gene[0],int(gene[5])-200,int(gene[5])+300)
        for read in aln_file.fetch(gene[0],start,stop):
            if read.reference_start < start or read.reference_start > stop:
                continue
            if not read.is_reverse and not read.is_unmapped:
                if not read.mate_is_unmapped:
                    read_seq = aln_file.mate(read).seq
                else:
                    for pair in read_index.find(read.query_name):
                        if pair.is_unmapped:
                            read_seq = str(Seq(pair.seq).reverse_complement()) #Because mapped read is in forward direction, the unmapped read is reverse and needs to be flipped
                align_score.append(aligner(read_seq, genome_seq, method='glocal', max_hits=1, matrix=DNAFULL)[0][-1])
    return np.median(align_score)

def calc_positive_diversity(gene,aln_file,read_index, frag_len=200, read_len=151):
    if gene[3] == '+':
        gene[5] = int(gene[5]) + frag_len
    else:
        gene[5] = int(gene[5]) - frag_len
    return calc_upstream_diversity(gene,aln_file,read_index, read_len=read_len)

def calc_negative_diversity(gene,aln_file,read_set,window=50, read_len=151):
    align_score = []
    num_reads = 0
    if gene[3] == '+':
        start = int(gene[5]) - SMALL_BUFFER
        stop = int(gene[5]) + window
        genome_seq = reference_genome.fetch(gene[0],int(gene[5])-300,int(gene[5])+200)
        for read in aln_file.fetch(gene[0],start,stop):
            if read.reference_start < start or read.reference_start > stop:
                continue
            if read.is_reverse and not read.is_unmapped:
                num_reads += 1
        while num_reads > 1:
            rand_ind = np.random.randint(1,len(read_set))
            read_seq = read_set[rand_ind]
            align_score.append(aligner(read_seq, genome_seq, method='glocal', max_hits=1, matrix=DNAFULL)[0][-1])
            num_reads = num_reads - 1
    else:
        stop = int(gene[5]) - read_len + SMALL_BUFFER
        start = int(gene[5]) - read_len - window
        genome_seq = reference_genome.fetch(gene[0],int(gene[5])-200,int(gene[5])+300)
        for read in aln_file.fetch(gene[0],start,stop):
            if not read.is_reverse and not read.is_unmapped:
                num_reads += 1
        while num_reads > 1:
            rand_ind = np.random.randint(1,len(read_set))
            read_seq = read_set[rand_ind]
            align_score.append(aligner(read_seq, genome_seq, method='glocal', max_hits=1, matrix=DNAFULL)[0][-1])
            num_reads = num_reads - 1
    return np.median(align_score)

def calc_negative_read_count(gene,aln_file,window=50, read_len=151):
    align_score = []
    num_reads = 0
    if gene[3] == '+':
        start = int(gene[5]) - SMALL_BUFFER
        stop = int(gene[5]) + window
        for read in aln_file.fetch(gene[0],start,stop):
            if read.reference_start < start or read.reference_start > stop:
                continue
            if read.is_reverse and not read.is_unmapped:
                num_reads += 1
    else:
        stop = int(gene[5]) - read_len + SMALL_BUFFER
        start = int(gene[5]) - read_len - window
        for read in aln_file.fetch(gene[0],start,stop):
            if read.reference_start < start or read.reference_start > stop:
                continue
            if not read.is_reverse and not read.is_unmapped:
                num_reads += 1
    return num_reads

def add_p_val(lft,rgt,y,h,p):
    plt.plot([lft, lft, rgt, rgt], [y, y+h, y+h, y], lw=1.5, c='k')
    plt.text((lft + rgt) * .5, y+h, ('n.s.' if p > 0.15 else 'p < %.2g' if p > 0.001 else 'p < %.1g') % max(p+1e-20, 1e-20), ha='center', va='bottom', color='k')

def scale_mat(d):
    return (d-d.mean(axis=1)[:,None]) / d.std(axis=1)[:,None]

def calc_bias(x):
    left_sum = np.sum(x[:500])
    right_sum = np.sum(x[500:])
    return(right_sum-left_sum)

def asm_2_species(asm):
    name = refseq[refseq['ftp_path'].str.contains(asm)]['organism_name'].item()
    return re.match(r'(^[^ ]+ [^ ]+).*',name).group(1)


with open('arg_samples') as f:
    samples = f.read().splitlines()
samples = [x.split('\t',1)[0] for x in samples]

mpa_res = pd.read_table('metaphlan3_res/merged_mpa.txt', comment= '#')
mpa_res = mpa_res[mpa_res['clade_name'].str.contains("s__")]
mpa_res.index = [int(re.match('.*\|([^\|]*)',x).group(1)) for x in mpa_res['NCBI_tax_id']]
mpa_res = mpa_res.drop(columns=['clade_name', 'NCBI_tax_id'])
mpa_res = mpa_res.loc[:,samples]
mpa_res = mpa_res.loc[((mpa_res > 1).sum(axis=1) >= 5),:] # Filter species with atleast 1% abundance in atleast 50% samples
mpa_res[mpa_res < 1] = np.nan
mpa_res = mpa_res.drop(39491) #Eubacterium Rectale no reference in database.

refseq = pd.read_table('src/assembly_summary_refseq.txt', header = 1)


all_dat = pd.DataFrame()
for taxid in mpa_res.index:
    asms = taxid2asm(taxid,3)
    for asm in asms:
        # Load gene regions
        gene_annot = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
        #Filter short genes
        gene_annot['len'] = gene_annot['geneRight']-gene_annot['geneLeft']
        gene_annot = gene_annot[gene_annot['len'] > 500]
        gene_annot = gene_annot[(gene_annot.Strand == '-') & (gene_annot.Start > 750)] #Incase of fetching beyond start of genome
        reference_genome = FastaFile(f'strains/{asm}/{asm}.fna')
        
        for sample in samples:
            if np.isnan(mpa_res.loc[taxid,sample]):
                continue
            # Load alignments
            aln_file = pysam.AlignmentFile(f'bams/{asm}_{sample}.sorted.bam', "rb")
            name_indexed = pysam.IndexedReads(aln_file)
            name_indexed.build()
            read_set = np.array([rec.seq for rec in aln_file])
            #Get read length
            for read in aln_file.fetch():
                read_len = len(read.get_forward_sequence())
                break
            #Calculate fragment length distribution
            fragment_len_dist = []
            for read in aln_file.fetch():
                if read.mate_is_unmapped or read.is_unmapped or not read.is_proper_pair:
                    continue
                if not read.is_reverse:
                    fragment_len_dist.append(read.template_length)
            frag_len = np.median(np.abs(fragment_len_dist))

            #Calc upstream read diversity metrics
            gene_annot['upstream_unmapped_count'] = [count_unmapped(x,aln_file,window=50, read_len=read_len) for x in np.array(gene_annot)]
            gene_annot['upstream_diversity'] = [calc_upstream_diversity(x,aln_file,name_indexed,read_len=read_len) for x in np.array(gene_annot)]
            gene_annot['positive_diversity'] = [calc_positive_diversity(x,aln_file,name_indexed, frag_len, read_len) for x in np.array(gene_annot)]
            gene_annot['negative_diversity'] = [calc_negative_diversity(x,aln_file, read_set, read_len=read_len) for x in np.array(gene_annot)]
            gene_annot['negative_read_count'] = [calc_negative_read_count(x,aln_file, read_len=read_len) for x in np.array(gene_annot)]

            sample_summary = gene_annot[['upstream_unmapped_count','upstream_diversity','positive_diversity','negative_diversity','negative_read_count']].mean()
            sample_summary['sample'] = sample
            sample_summary['asm'] = asm
            sample_summary.to_csv(f'plots/read_diversity/{asm}_{sample}_upstream_read_diversity.csv')
            all_dat = all_dat.append(sample_summary, ignore_index=True)
all_dat.to_csv("plots/read_diversity/all_samp_upstream_read_diversity.csv")


all_strains = []
for taxid in mpa_res.index:
    asms = taxid2asm(taxid,3)
    all_strains = all_strains + asms
asm_2_species_dict = {x:asm_2_species(x) for x in all_strains}


all_dat = pd.read_csv("plots/all_samp_upstream_read_diversity.csv", index_col = 0)
all_dat = all_dat.melt(id_vars = ['sample','asm'])
all_dat['species'] = all_dat.asm.map(asm_2_species_dict)

subset = all_dat[all_dat['variable'].isin(['negative_diversity','upstream_diversity','positive_diversity'])]
parsed_dat = subset.groupby(['sample','species','variable']).mean().reset_index()
order = ['negative_diversity','upstream_diversity','positive_diversity']
g = sns.boxplot(x=parsed_dat["variable"],y=parsed_dat["value"], order=order, color = 'tab:blue')
add_p_val(0,1,800,10,stats.ttest_rel(parsed_dat.loc[parsed_dat.variable=='negative_diversity','value'],parsed_dat.loc[parsed_dat.variable=='upstream_diversity','value']).pvalue)
#Ttest_relResult(statistic=-131.38931905950795, pvalue=1.6472879056658431e-74)
add_p_val(1,2,920,10,stats.ttest_rel(parsed_dat.loc[parsed_dat.variable=='upstream_diversity','value'],parsed_dat.loc[parsed_dat.variable=='positive_diversity','value']).pvalue)
#Ttest_relResult(statistic=-11.458839493063426, pvalue=1.2259349831832464e-16)
g.figure.savefig("plots/read_diversity/all_samp_species_paired.pdf")
plt.clf()

order = ['upstream_diversity','positive_diversity']
g = sns.boxplot(x=parsed_dat["variable"],y=parsed_dat["value"], order=order, showfliers=False, color='tab:blue')
add_p_val(0,1,780,5,stats.ttest_rel(parsed_dat.loc[parsed_dat.variable=='upstream_diversity','value'],parsed_dat.loc[parsed_dat.variable=='positive_diversity','value']).pvalue)
g.figure.savefig(f"plots/read_diversity/all_samp_species_paired_zoomin.pdf")
plt.clf()


### Heterogenity vs bias ###
all_dat = pd.DataFrame()
for taxid in mpa_res.index:
    asms = taxid2asm(taxid,3)
    for asm in asms:
        # Load gene regions
        gene_annot = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
        #Filter short genes
        gene_annot['len'] = gene_annot['geneRight']-gene_annot['geneLeft']
        gene_annot = gene_annot[gene_annot['len'] > 500]
        gene_annot = gene_annot[(gene_annot.Strand == '-') & (gene_annot.Start > 750)] #Incase of fetching beyond start of genome
        gene_annot.index = np.array([re.match(r'ID=([^;]*)',x).group(1) for x in gene_annot['Desc']])
        reference_genome = FastaFile(f'strains/{asm}/{asm}.fna') #needed as global variable in calc_upstream_diversity
        for sample in samples:
            if np.isnan(mpa_res.loc[taxid,sample]):
                continue
            mat = pd.read_csv(f'matrices/{asm}_{sample}.csv', header = None, index_col = 0)
            scaled_mat = scale_mat(mat)
            # Load alignments
            aln_file = pysam.AlignmentFile(f'bams/{asm}_{sample}.sorted.bam', "rb")
            name_indexed = pysam.IndexedReads(aln_file)
            name_indexed.build()
            #Get read length
            for read in aln_file.fetch():
                read_len = len(read.get_forward_sequence())
                break
            gene_annot['upstream_diversity'] = [calc_upstream_diversity(x,aln_file,name_indexed,read_len=read_len) for x in np.array(gene_annot)]
            gene_annot['bias'] = list(map(calc_bias, np.array(scaled_mat.loc[gene_annot.index,:])))
            gene_annot = gene_annot.dropna()
            gene_annot['sample'] = sample
            gene_annot['asm'] = asm
            all_dat = all_dat.append(gene_annot[['sample','asm','upstream_diversity','bias',]])
all_dat.to_csv("plots/all_samp_bias_vs_read_diversity.csv")


all_dat = pd.read_csv("plots/all_samp_bias_vs_read_diversity.csv", index_col=0)
corr_res = []
for samp in set(all_dat['sample']):
    for asm in set(all_dat['asm']):
        dat = all_dat[(all_dat['sample']==samp)&(all_dat['asm']==asm)]
        if len(dat) == 0:
            continue
        res = stats.pearsonr(dat['bias'], dat['upstream_diversity'])
        corr_res.append([samp,asm,res[0],res[1]])
corr_res = pd.DataFrame(corr_res, columns = ['sample','asm','corr','pvalue'])
corr_res['species'] = corr_res['asm'].map(asm_2_species_dict)
corr_res = corr_res.groupby(['sample','species']).mean().reset_index()

corr_pivot = corr_res.pivot("sample", "species", "corr")
corr_pivot = corr_pivot.transpose()
g = sns.heatmap(corr_pivot, vmin=-0.3, vmax=0.3, cmap="coolwarm")
g.figure.savefig(f"plots/read_diversity/bias_vs_read_diversity_corr_heatmap.pdf", bbox_inches="tight")
plt.clf()

g = sns.boxplot(y=corr_res["species"],x=corr_res["corr"], color = 'tab:blue')
g.figure.savefig(f"plots/read_diversity/bias_vs_read_diversity_corr_boxplot.pdf", bbox_inches="tight")
plt.clf()

corr_res['adj_pval'] = multi.multipletests(corr_res["pvalue"], method = 'fdr_bh', alpha = 0.05)[1]
corr_res['log_pval'] = -np.log(corr_res["adj_pval"]) 
g = sns.boxplot(y=corr_res["species"],x=corr_res["log_pval"], color = 'tab:blue', showfliers=False)
plt.axvline(-np.log(0.05), color='r')
g.figure.savefig(f"plots/read_diversity/bias_vs_read_diversity_corr_pval_boxplot.pdf", bbox_inches="tight")
plt.clf()

stats.pearsonr(all_dat['bias'], all_dat['upstream_diversity'])
#(-0.15158442450241497, 0.0)

fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(
    data=all_dat,
    x="bias",
    y="upstream_diversity",
    color="k",
    ax=ax,
)
sns.kdeplot(
    data=all_dat,
    x="bias",
    y="upstream_diversity",
    levels=5,
    fill=True,
    alpha=0.6,
    cut=2,
    ax=ax,
)
