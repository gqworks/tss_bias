import pandas as pd
import re
import numpy as np
from pysam import FastaFile
import pysam
from align.calign import aligner
from align.matrix import DNAFULL
from Bio.Seq import Seq
import seaborn as sns
from scipy import stats
from tqdm import tqdm


# --- Gadi/portable bootstrap (added in refactor) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from tss_bias.runtime import bootstrap as _bootstrap, mpa_path as _mpa_path
_CFG = _bootstrap(chdir_work=False)
_MPA = _mpa_path(_CFG)
_HMP2_FQ = _CFG["cohorts"]["hmp2"]["fastq_dir"]
# --- end bootstrap ---
SMALL_BUFFER = 5

def get_tpm(file):
    dat = pd.read_table(file)
    dat = dat.drop_duplicates(subset='target_id')
    dat = dat[['target_id','tpm']]
    dat.index = dat['target_id']
    dat = dat.drop(columns = ['target_id'])
    return dat

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


#with open('sample_lists/Franzosa_8_samples') as f:
#    samples = f.read().splitlines()
#samples = [x.rsplit('\t',1)[0] for x in samples]

with open('sample_lists/hmp2_samples') as f:
    hmp2_samples = [ln for ln in f.read().splitlines() if ln.strip() and not ln.startswith('#')]
hmp2_samples = {x.split('\t')[0]:x.split('\t')[1] for x in hmp2_samples}

#Read species abundances
mpa_res = pd.read_table(_MPA, comment= '#')
mpa_res = mpa_res[(mpa_res['clade_name'].str.contains("s__"))&(~mpa_res['clade_name'].str.contains("t__"))]
mpa_res.index = [re.match('.*\|([^\|]*)',x).group(1) for x in mpa_res['clade_name']]
mpa_res = mpa_res.drop(columns=['clade_name'])

hmp2_mpa = mpa_res.loc[:,hmp2_samples.keys()]
hmp2_mpa = hmp2_mpa.loc[((hmp2_mpa > 0.5).sum(axis=1) >= 30),:] # Filter species with atleast 0.5% abundance in atleast 30 samples
hmp2_mpa = hmp2_mpa.where(hmp2_mpa > 0.5, np.nan)

hmp2_genomes = {'s__Phocaeicola_vulgatus': 'GCF_000012825.1_ASM1282v1',
 's__Agathobacter_rectalis': 'GCF_022453685.1_ASM2245368v1',
 's__Alistipes_putredinis': 'GCF_000154465.1_ASM15446v1',
 's__Faecalibacterium_prausnitzii': 'GCF_000154385.1_ASM15438v1',
 's__Bacteroides_uniformis': 'GCF_044361425.1_ASM4436142v1',
 's__Parabacteroides_distasonis': 'GCF_018279895.1_ASM1827989v1',
 's__Bacteroides_caccae': 'GCF_018292205.1_ASM1829220v1'}

all_df = pd.DataFrame()
for species in tqdm(hmp2_mpa.index):
    asm = hmp2_genomes[species]

    #Load gene regions
    reference_genome = FastaFile(f'refseq_genomes/{asm}/{asm}.fna')
    gene_annot = pd.read_csv(f'refseq_genomes/{asm}/{asm}_filtered_tss.csv')
    gene_annot.index = np.array([re.match(r'ID=([^;]*)',x).group(1) for x in gene_annot['Desc']])
    #Filter short genes
    gene_annot['len'] = gene_annot['geneRight']-gene_annot['geneLeft']
    gene_annot = gene_annot[gene_annot['len'] > 500]
    gene_annot = gene_annot[(gene_annot.Strand == '-') & (gene_annot.Start > 750)] #Incase of fetching beyond start of genome

    for samp in hmp2_samples.keys():
        if np.isnan(hmp2_mpa.loc[species,samp]):
            continue
        #Read TPM sample data
        tpm_file = get_tpm(f'kallisto_res/{asm}_{samp}/abundance.tsv')
        chrom_map = pd.read_table(f'refseq_genomes/{asm}/{asm}_filtered_tss.csv', sep = ',', header=0, index_col=0)
        chrom_map = chrom_map[~chrom_map.index.duplicated(keep='first')]
        chrom_map = {chrom:re.match(r'ID=(\d*)_',chrom_map.loc[chrom,'Desc']).group(1) for chrom in chrom_map.index}
        tpm_file.index = [x.replace(x.rsplit('_',1)[0],chrom_map.get(x.rsplit('_',1)[0],'na')) for x in tpm_file.index]
        tpm_dict = tpm_file.to_dict()
        gene_annot["tpm"] = [tpm_dict['tpm'].get(x,np.nan)for x in gene_annot.index]

        #Calc upstream diversity
        aln_file = pysam.AlignmentFile(f'bams/{asm}_{samp}.sorted.bam', "rb")
        name_indexed = pysam.IndexedReads(aln_file)
        name_indexed.build()
        for read in aln_file.fetch(): #Get read length
            read_len = len(read.get_forward_sequence())
            break
        gene_annot['upstream_diversity'] = [calc_upstream_diversity(x,aln_file,name_indexed,read_len=read_len) for x in np.array(gene_annot)]
        gene_annot['mapped_reads'] = np.sum([int(x.split('\t')[2]) for x in pysam.idxstats(f'bams/{asm}_{samp}.sorted.bam').split('\n')[:-1]])
        gene_annot['samp'] = samp
        gene_annot['asm'] = asm
        gene_annot['species'] = species
        all_df = pd.concat([all_df, gene_annot])
        
all_df = all_df.dropna()
all_df = all_df[all_df['tpm']>0]
all_df['mapped_reads'] = [np.sum([int(x.split('\t')[2]) for x in pysam.idxstats(f'bams/{row["asm"]}_{row["samp"]}.sorted.bam').split('\n')[:-1]]) for i,row in all_df.iterrows()]
all_df['norm_tpm'] = 1000000*all_df['tpm']/all_df['mapped_reads']
all_df['log_norm_tpm'] = np.log(all_df['norm_tpm'] + 0.001)
all_df.to_csv('plots/hmp2_sample_species_tpm_vs_hetero.csv')

all_df = pd.read_csv('plots/hmp2_sample_species_tpm_vs_hetero.csv', index_col = 0)
g = sns.regplot(data = all_df, x = 'upstream_diversity', y = 'log_norm_tpm')
print(stats.pearsonr(all_df['log_norm_tpm'], all_df['upstream_diversity']))
#(-0.10025090230262745, 4.149096420444886e-164)
# HMP2 (-0.14298772543336633, 0.0)

#for asm in list(set(all_df['asm'])):
#    for samp in list(set(all_df['samp'])):
#        subset = all_df[(all_df['asm']==asm)&(all_df['samp']==samp)]
#        if len(subset) > 0:
#            g = sns.regplot(data = subset, x = 'upstream_diversity', y = 'tpm')
#            g.figure.savefig(f"hetero_plots/hetero_vs_tpm_{asm}_{samp}.pdf", bbox_inches="tight")
#            plt.clf()