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


SMALL_BUFFER = 5

def taxid2asm(taxid,n=1):
    strain = refseq[(refseq['species_taxid']==int(taxid)) & (refseq['assembly_level']=='Complete Genome')]
    if len(strain) == 0:
        strain = refseq[(refseq['species_taxid']==int(taxid))]
    ftp = strain['ftp_path'][:n].tolist()
    asm = [x.rsplit('/',1)[1] for x in ftp]
    return(asm)

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

def asm_2_species(asm):
    name = refseq[refseq['ftp_path'].str.contains(asm)]['organism_name'].item()
    name = re.match(r'(^[^ ]+ [^ ]+).*',name).group(1)
    if name == '[Eubacterium] eligens':
        name = 'Lachnospira eligens'
    return name



with open('Franzosa_8_samples') as f:
    samples = f.read().splitlines()
samples = [x.rsplit('\t',1)[0] for x in samples]

#Read species abundances
mpa_res = pd.read_table('metaphlan3_res/merged_mpa.txt', comment= '#')
mpa_res = mpa_res[mpa_res['clade_name'].str.contains("s__")]
mpa_res.index = [int(re.match('.*\|([^\|]*)',x).group(1)) for x in mpa_res['NCBI_tax_id']]
mpa_res = mpa_res.drop(columns=['clade_name', 'NCBI_tax_id'])
mpa_res = mpa_res.loc[:,samples]
mpa_res = mpa_res.loc[((mpa_res > 1).sum(axis=1) >= 5),:] # Filter species with atleast 1% abundance in atleast 5 samples
mpa_res[mpa_res < 1] = np.nan
mpa_res = mpa_res.drop(39491) #Eubacterium Rectale no reference in database.

refseq = pd.read_table('src/assembly_summary_refseq.txt', header = 1)

all_df = pd.DataFrame()
for taxid in mpa_res.index:
    asms = taxid2asm(taxid,3)
    for asm in asms:
        #Load gene regions
        reference_genome = FastaFile(f'strains/{asm}/{asm}.fna')
        gene_annot = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
        gene_annot.index = np.array([re.match(r'ID=([^;]*)',x).group(1) for x in gene_annot['Desc']])
        #Filter short genes
        gene_annot['len'] = gene_annot['geneRight']-gene_annot['geneLeft']
        gene_annot = gene_annot[gene_annot['len'] > 500]
        gene_annot = gene_annot[(gene_annot.Strand == '-') & (gene_annot.Start > 750)] #Incase of fetching beyond start of genome

        for samp in samples:
            if np.isnan(mpa_res.loc[taxid,samp]):
                continue
            #Read TPM sample data
            tpm_file = get_tpm(f'kallisto_res/{asm}_{samp}/abundance.tsv')
            chrom_map = pd.read_table(f'strains/{asm}/{asm}_filtered_tss.csv', sep = ',', header=0, index_col=0)
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
            all_df = pd.concat([all_df, gene_annot])

all_df = all_df.dropna()
all_df = all_df[all_df['tpm']>0]
all_df['mapped_reads'] = [np.sum([int(x.split('\t')[2]) for x in pysam.idxstats(f'bams/{row["asm"]}_{row["samp"]}.sorted.bam').split('\n')[:-1]]) for i,row in all_df.iterrows()]
all_df['norm_tpm'] = 1000000*all_df['tpm']/all_df['mapped_reads']
all_df['log_norm_tpm'] = np.log(all_df['norm_tpm'] + 0.001)
asm_2_species_dict = {x:asm_2_species(x) for x in set(all_df['asm'].tolist())}
all_df['species'] = all_df['asm'].map(asm_2_species_dict)
all_df.to_csv('plots/all_sample_species_tpm_vs_hetero.csv')

g = sns.regplot(data = all_df, x = 'upstream_diversity', y = 'log_norm_tpm')
stats.pearsonr(all_df['log_norm_tpm'], all_df['upstream_diversity'])
#(-0.10025090230262745, 4.149096420444886e-164)

#for asm in list(set(all_df['asm'])):
#    for samp in list(set(all_df['samp'])):
#        subset = all_df[(all_df['asm']==asm)&(all_df['samp']==samp)]
#        if len(subset) > 0:
#            g = sns.regplot(data = subset, x = 'upstream_diversity', y = 'tpm')
#            g.figure.savefig(f"hetero_plots/hetero_vs_tpm_{asm}_{samp}.pdf", bbox_inches="tight")
#            plt.clf()