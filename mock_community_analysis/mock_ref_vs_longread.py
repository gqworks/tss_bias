import pyBigWig
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from Bio import SeqIO
import re
from scipy.stats import ttest_rel

from tqdm import tqdm
import random

from scipy.stats import mannwhitneyu

from scipy.stats import ttest_ind


def scale_mat(d):
    return (d-d.mean(axis=1)[:,None]) / d.std(axis=1)[:,None]

def parse_gff(gff, ref_len_dict):
    regions = []
    for contig in list(set(gff[0])):
        dat = np.array(gff[gff[0]==contig])
        end_of_contig = ref_len_dict[contig]        
        if dat[0,3] != 0:
            regions.append([0,dat[0,3], 'intergenic',contig])
        last_start = dat[0,3]
        for i in range(len(dat)):
            if i == len(dat)-1:
                regions.append([last_start,dat[i,4],'geneic',contig])
                regions.append([dat[i,4],end_of_contig,'intergenic',contig])
                break
            if dat[i,4] < dat[i+1,3]:
                regions.append([last_start,dat[i,4],'geneic',contig])
                regions.append([dat[i,4],dat[i+1,3],'intergenic',contig])
                last_start = dat[i+1,3] #increment last start
    return(regions)

def filter_overlap_genes(gff,ref_len_dict):
    filt_genes = []
    #for contig in list(ref_len_dict.keys()):
    for contig in list(set(gff[0])):
        #genes = gff[gff[:,0] == contig, :]
        genes = np.array(gff[gff[0]==contig])
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

def calc_bias(x):
    left_sum = np.sum(x[:500])
    right_sum = np.sum(x[500:])
    return(right_sum-left_sum)

def add_p_val(ax,lft,rgt,y,h,p):
    ax.plot([lft, lft, rgt, rgt], [y, y+h, y+h, y], lw=1.5, c='k')
    ax.text((lft + rgt) * .5, y+h, ('n.s.' if p > 0.15 else 'p < %.2g' if p > 0.001 else 'p < %.1g') % max(p+1e-20, 1e-20), ha='center', va='bottom', color='k')


#Load reference genome read coverage and prodigal annot
bw_ref = pyBigWig.open(f'beds/SRR8359173_all_ref.bw')
ref_len_dict = bw_ref.chroms()
ref_names = list(ref_len_dict.keys())
gff_ref = pd.read_table('mock_ref/merged_genomes_prodigal.gff', comment = '#', header=None)
fasta_dict = {rec.id : re.match(r'[^ ]* ([^ ]* [^ ]*)',rec.description).group(1) for rec in SeqIO.parse("mock_ref/merged_genomes.fna", "fasta")}
species_dict = {}
for key, value in fasta_dict.items():
    if value in species_dict:
        species_dict[value] = species_dict[value] + [key]
    else:
        species_dict[value] = [key]
all_species = list(species_dict.keys())

# Calculate gene percentage of long read reference
parsed_gff_ref = parse_gff(gff_ref,ref_len_dict)
parsed_gff_ref = np.array(parsed_gff_ref)
perc = []
for x in ref_names:
    dat = parsed_gff_ref[parsed_gff_ref[:,3] == x, :]
    genes_gff = dat[dat[:,2] == "geneic", :]
    gene_sum = np.sum((genes_gff[:,1].astype(int)) - genes_gff[:,0].astype(int))
    perc.append(gene_sum/ref_len_dict[x])
np.mean(perc) # 87.6%

# Calculate coverage bias
ref_species_bias_means = []
ref_sepcies_bias = pd.DataFrame()
for species in all_species:
    indexes = species_dict[species]
    species_gff = gff_ref[gff_ref[0].isin(indexes)]
    filt_genes = filter_overlap_genes(species_gff,ref_len_dict)
    mat = []
    for gene in filt_genes:
        strand = gene[5]
        start = gene[2] if strand == '+' else gene[3]
        read_cv = np.array(bw_ref.values(gene[0], start-500, start+500))
        read_cv = np.flip(read_cv) if strand == '-' else read_cv
        mat.append(read_cv)
    mat = pd.DataFrame(mat)
    ref_scaled_mat = scale_mat(mat)
    ref_scaled_mat = ref_scaled_mat.dropna()
    ref_bias_vec = list(map(calc_bias, np.array(ref_scaled_mat)))
    ref_species_bias_means = ref_species_bias_means + [np.mean(ref_bias_vec)]
    dat = pd.DataFrame(ref_bias_vec,columns = ["bias"])
    dat['species'] = species
    ref_sepcies_bias = pd.concat([ref_sepcies_bias,dat])
    
sns.boxplot(y = "species", x = "bias", data=ref_sepcies_bias) 
plt.savefig(f"plots/mock_all_species_TSS_bias.pdf", bbox_inches='tight')
plt.clf()



#Load longRead coverage and prodigal annot
bw_longRead = pyBigWig.open(f'beds/SRR8359173_hifi.bw')
longRead_dict = bw_longRead.chroms()
minimap_longRead_annot = pd.read_table(f'minimap_res/hifi_all_minimap2.paf', header=None, usecols=range(12))
minimap_longRead_annot['readName'] = [x.rsplit('_',1)[0] for x in minimap_longRead_annot[0]]
minimap_longRead_annot['species_prefix'] = [x.rsplit('_',1)[0] for x in minimap_longRead_annot[5]]
gff_longRead = pd.read_table(f'mock_hifi_ref/hifi_100K_sample_prodigal.gff', comment = '#', header=None)

longread_species_bias_means = []
for species in all_species:
    indexes = species_dict[species]
    #Filter longreads with matching genes with genome
    species_minimap_subset = minimap_longRead_annot[minimap_longRead_annot['species_prefix'].isin(indexes)]
    gff_longRead_subset = gff_longRead[gff_longRead[0].isin(species_minimap_subset['readName'])]
    gff_longRead_subset['len'] = gff_longRead_subset[0].map(longRead_dict)
    gff_longRead_subset = gff_longRead_subset[gff_longRead_subset['len']>10000]
    read_set = list(set(gff_longRead_subset[0]))
    gff_longRead_subset[3] = gff_longRead_subset[3] - 1 # change for 0-indexed
    gff_longRead_subset[4] = gff_longRead_subset[4] - 1 # change for 0-indexed

    filt_genes = []
    for longRead in read_set:
        read_len = longRead_dict[longRead]
        genes = pd.DataFrame(gff_longRead_subset[gff_longRead_subset[0]==longRead])
        filt_genes = filt_genes + filter_overlap_genes(genes,longRead_dict).tolist()
    filt_genes = np.array(filt_genes)

    # Get gene vs intergenic percentage of Long reads
    #parsed_longread_gff_ref = parse_gff(gff_longRead_subset,longRead_dict)
    #parsed_longread_gff_ref = np.array(parsed_longread_gff_ref)
    #perc = []
    #for x in tqdm(list(set(gff_longRead_subset[0]))):
    #    dat = parsed_longread_gff_ref[parsed_longread_gff_ref[:,3] == x, :]
    #    genes_gff = dat[dat[:,2] == "geneic", :]
    #    gene_sum = np.sum((genes_gff[:,1].astype(int)) - genes_gff[:,0].astype(int))
    #    perc.append(gene_sum/longRead_dict[x]) 
    #np.mean(perc) # 80.2%

    #Gene Coverage at TSS
    mat = []
    for gene in filt_genes:
        strand = gene[5]
        start = int(gene[2]) if strand == '+' else int(gene[3])
        read_cv = np.array(bw_longRead.values(gene[0], start-500, start+500))
        read_cv = np.flip(read_cv) if strand == '-' else read_cv
        mat.append(read_cv)
    mat = pd.DataFrame(mat)
    longread_scaled_mat = scale_mat(mat)
    longread_scaled_mat = longread_scaled_mat.dropna()
    longread_bias_vec = list(map(calc_bias, np.array(longread_scaled_mat)))
    longread_species_bias_means = longread_species_bias_means + [np.mean(longread_bias_vec)]

df = pd.concat([pd.Series(ref_species_bias_means),pd.Series(longread_species_bias_means)], axis=1)
df = df.rename(columns={0:'genome',1:'long read'})
df.to_csv("mock_all_genome_long_read_bias.csv")

df = pd.read_csv("mock_all_genome_long_read_bias.csv", index_col = 0)
df = df.dropna()
sns.boxplot(data= df, color = 'tab:blue')
plt.ylim(-40, 40)
add_p_val(plt,0,1,35,2,ttest_rel(df['genome'],df['long read']).pvalue) # Ttest_relResult(statistic=0.6736680757335564, pvalue=0.5090811929580812)
plt.savefig(f"plots/mock_all_species_bias_reference_vs_longread_boxplot.pdf")
plt.clf()