import re, os
import pandas as pd
import pyBigWig
import numpy as np
from pysam import FastaFile
import subprocess
import seaborn as sns
from matplotlib import pyplot as plt
import glob
import pysam
from io import StringIO
from scipy.stats import ttest_rel
from pathlib import Path

def scale_mat(d):
    mean_vals = d.mean(axis=1).to_numpy()
    std_vals = d.std(axis=1).to_numpy()
    return (d - mean_vals[:, None]) / std_vals[:, None]

def calc_bias(x):
    left_sum = np.sum(x[:500])
    right_sum = np.sum(x[500:])
    return(right_sum-left_sum)

def filter_overlap_genes(gff,ref_len_dict):
    filt_genes = []
    for contig in list(ref_len_dict.keys()):
        genes = gff[gff[:,0] == contig, :]
        tot_len = ref_len_dict[contig]
        for i in range(1,len(genes)-1): # Ignore first and last gene to avoid writing edge cases, also they most likely fall into the 500bp long read edge anyway
            left_pos = genes[i,3]
            right_pos = genes[i,4]
            strand = genes[i,6]
            start = left_pos if strand == '+' else right_pos
            if start-500 < 500 or start+500 > tot_len-500: #remove 500bp upstream/downstream of edges
                continue
            previous_gene_start = genes[i-1,3] if genes[i-1,6] == '+' else genes[i-1,4]
            next_gene_start = genes[i+1,3] if genes[i+1,6] == '+' else genes[i+1,4]
            if start-500 > previous_gene_start and start+500 < next_gene_start:
                filt_genes.append(genes[i])
    return(np.array(filt_genes))

def kraken_majority_annot(report):
    dat = pd.read_csv(report, sep = '\t', header = None)
    dat = dat[dat[3]=='S']
    taxid = dat.iloc[0,4]
    perc = dat.iloc[0,0]
    return(taxid, perc)

def find_best_bin(taxid,sample):
    tax_bins = []
    for i in glob.glob(f'assembled_ref/{sample}/kraken_res/*.report'):
        bin_no = re.match('.*(bin\.\d+)\.report',i).group(1)
        bin_taxid, perc = kraken_majority_annot(i)
        if bin_taxid == taxid:
            tax_bins.append([bin_no,perc])
    if len(tax_bins) == 0:
        return 0
    tax_bins = pd.DataFrame(tax_bins)
    top_bin = tax_bins.iloc[tax_bins[1].idxmax(),0]
    top_perc = tax_bins.iloc[tax_bins[1].idxmax(),1]
    if top_perc > 80:
        return top_bin
    else:
        return 0

def add_p_val(ax,lft,rgt,y,h,p):
    ax.plot([lft, lft, rgt, rgt], [y, y+h, y+h, y], lw=1.5, c='k')
    ax.text((lft + rgt) * .5, y+h, ('n.s.' if p > 0.15 else 'p < %.2g' if p > 0.001 else 'p < %.1g') % max(p+1e-20, 1e-20), ha='center', va='bottom', color='k')

    
### Franzosa dataset (8 metagenomics and metatranscriptomics matched samples) ###
#with open('Franzosa_8_samples') as f:
#    Franzosa_8 = f.read().splitlines()
#Franzosa_8 = {x.split('\t')[0]:x.split('\t')[1] for x in Franzosa_8}

# HMP2 Samples
with open('sample_lists/hmp2_samples') as f:
    hmp2_samples = f.read().splitlines()
hmp2_samples = {x.split('\t')[0]:x.split('\t')[2] for x in hmp2_samples}

#Read species abundances
mpa_res = pd.read_table('mpa4_res/merged_mpa.txt', comment= '#')
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

hmp2_taxid = {'s__Phocaeicola_vulgatus': 821,
 's__Agathobacter_rectalis': 39491,
 's__Alistipes_putredinis': 28117,
 's__Faecalibacterium_prausnitzii': 853,
 's__Bacteroides_uniformis': 820,
 's__Parabacteroides_distasonis': 823,
 's__Bacteroides_caccae': 47678}

# Assemble samples
#for sample, path in hmp2_samples.items():
#    subprocess.call(f'ml megahit && megahit -t 64 -1 {path}/{sample}_1.fastq.gz -2 {path}/{sample}_2.fastq.gz -o assembled_ref/{sample}', shell = True)
#    subprocess.call(f'bowtie2-build assembled_ref/{sample}/final.contigs.fa assembled_ref/{sample}/final.contigs.fa', shell=True)
'''
# Align reads to assembled reference and bin
for sample, path in hmp2_samples.items():
    if not os.path.exists(f'bams/{sample}_assembled_ref.sorted.bam'):
        subprocess.call(f'bowtie2 -p 64 -x assembled_ref/{sample}/final.contigs.fa -1 {path}/{sample}_1.fastq.gz -2 {path}/{sample}_2.fastq.gz |samtools view --threads 64 -b - |samtools sort - -o bams/{sample}_assembled_ref.sorted.bam --threads 64', shell = True)
        subprocess.call(f'samtools index bams/{sample}_assembled_ref.sorted.bam -@ 64', shell = True)

    # Get read depth of genome
    if not os.path.exists(f'beds/{sample}_assembled_ref.bw'):
        subprocess.call(f'bamCoverage --bam bams/{sample}_assembled_ref.sorted.bam -p 64 -o beds/{sample}_assembled_ref.bw -of bigwig', shell = True)
    
    # Annotate genes w/ prodigal
    if not os.path.exists(f'assembled_ref/{sample}/prodigal.gff'):
        subprocess.call(f'prodigal -i assembled_ref/{sample}/final.contigs.fa -o assembled_ref/{sample}/prodigal.gff -f gff -p meta -d assembled_ref/{sample}/prodigal.fna', shell = True)
    
    # Bin Contigs
    if not os.path.exists(f'assembled_ref/{sample}/depth.txt'):
        subprocess.call(f'ml MetaBAT && jgi_summarize_bam_contig_depths --outputDepth assembled_ref/{sample}/depth.txt bams/{sample}_assembled_ref.sorted.bam', shell = True)
        subprocess.call(f'ml MetaBAT && metabat2 -t 64 -i assembled_ref/{sample}/final.contigs.fa -a assembled_ref/{sample}/depth.txt -o assembled_ref/{sample}/metabat/bin', shell = True)
    
    # Annotate contig bins with Kraken2
    if not os.path.exists(f'assembled_ref/{sample}/kraken_res/'):
        os.mkdir(f'assembled_ref/{sample}/kraken_res/')
    for i in glob.glob(f'assembled_ref/{sample}/metabat/*.fa'):
        bin_no = re.match('.*(bin\.\d+)\.fa',i).group(1)
        subprocess.call(f'ml ncbi-blast/2.12.0 kraken/2.1.2 && kraken2 --threads 64 --db /groups/cgsd/hbyao/db/bfv/ --output assembled_ref/{sample}/kraken_res/{bin_no}.kra --report assembled_ref/{sample}/kraken_res/{bin_no}.report {i}', shell = True)  
'''

# Find number of reads aligned
#read_count_df = []
#for sample,path in hmp2_samples.items():
#    bamfile = f'bams/{sample}_assembled_ref.sorted.bam'
#    indStats = pd.read_csv(StringIO(pysam.idxstats(bamfile)), sep = '\t', header = None, names = ['contig', 'length', 'mapped', 'unmapped'])
#    tot_mapped = indStats.mapped.sum()
#    tot_unmapped = indStats.unmapped.sum()
#    read_count_df.append([sample,tot_mapped,tot_unmapped])

# Checkm2 Bins
#os.mkdir('checkm2_out')
for sample in hmp2_mpa.columns:
    bw_contig = pyBigWig.open(f'beds/{sample}_assembled_ref.bw')
    contig_dict = bw_contig.chroms()
    for species in hmp2_mpa.index:
        if np.isnan(hmp2_mpa.loc[species,sample]):
            continue
        asm = hmp2_genomes[species]
        taxid = hmp2_taxid[species]
        binno = find_best_bin(taxid,sample)
        if binno == 0:
            print(f'No high quality bin found for {taxid} in {sample}')
            continue
        bin_loc = f'assembled_ref/{sample}/metabat/{binno}.fa'
        subprocess.call(f'source activate checkm2 && checkm2 predict --threads 64 --remove_intermediates --input {bin_loc} --output-directory checkm2_out/{sample}_{asm}', shell = True)
        
rows = []
for f in Path("checkm2_out").rglob("quality_report.tsv"):
    df = pd.read_csv(f, sep="\t")
    df["source"] = f.parent.name
    rows.append(df)
merged = pd.concat(rows, ignore_index=True)
merged.to_csv("plots/checkm2_merged.csv", index=False)
merged.boxplot(["Completeness", "Contamination"])
plt.tight_layout()
plt.savefig("plots/binning_checkm2_boxplot.pdf", dpi=300, bbox_inches="tight")
plt.close()


# Calculate Bias in bins
checkm2_res = pd.read_csv("plots/checkm2_merged.csv")
bias_means = []
for sample in hmp2_mpa.columns:
    bw_contig = pyBigWig.open(f'beds/{sample}_assembled_ref.bw')
    contig_dict = bw_contig.chroms()
    for species in hmp2_mpa.index:
        if np.isnan(hmp2_mpa.loc[species,sample]):
            continue
        asm = hmp2_genomes[species]
        taxid = hmp2_taxid[species]

        binno = find_best_bin(taxid,sample)
        if binno == 0:
            print(f'No high quality bin found for {taxid} in {sample}')
            continue

        # CheckM2
        checkm_dat = checkm2_res[(checkm2_res['source']==f'{sample}_{asm}')&(checkm2_res['Name']==binno)]
        if len(checkm_dat) != 1:
            print(f'No checkM data found for {asm} in {sample}')
            continue
        
        completeness = checkm_dat['Completeness'].item()
        contamination = checkm_dat['Contamination'].item()
        if (completeness < 90) or contamination > 5:
            print(f'{binno} does not pass High Quality bin for {asm} in {sample}')
            continue
        
        # Reference genome bias
        ref_bias_vec = []
        mat = pd.read_csv(f'matrices/{asm}_{sample}.csv', header = None, index_col = 0)
        ref_avg_cv = mat.mean().mean()
        scaled_mat = scale_mat(mat)
        scaled_mat = scaled_mat.dropna()
        ref_mean = np.mean(list(map(calc_bias, np.array(scaled_mat))))

        #Get contigs from bin 
        contigs = FastaFile(f'assembled_ref/{sample}/metabat/{binno}.fa').references
        gff_contig = pd.read_table(f'assembled_ref/{sample}/prodigal.gff', comment = '#', header=None)
        gff_contig = gff_contig[gff_contig[0].isin(contigs)]
        gff_contig['partial'] = [re.match('.*partial=(\d\d)',x).group(1) for x in gff_contig[8]]
        gff_contig['contig_len'] = gff_contig[0].map(contig_dict)
        gff_contig['gene_len'] = gff_contig[4] - gff_contig[3]
        gff_contig[3] = gff_contig[3] - 1 # change for 0-indexed
        gff_contig[4] = gff_contig[4] - 1 # change for 0-indexed
        gff_contig = gff_contig[(gff_contig['gene_len']>500)&(gff_contig['partial']=='00')&(gff_contig['contig_len']>2000)]
        read_set = list(set(gff_contig[0]))
        
        # Filter overlapping genes
        filt_genes = []
        for contig in read_set:
            read_len = contig_dict[contig]
            genes = np.array(gff_contig[gff_contig[0]==contig])
            filt_genes = filt_genes + filter_overlap_genes(genes,contig_dict).tolist()
        filt_genes = pd.DataFrame(filt_genes)
        
        #Gene Coverage at TSS
        mat = []
        for gene in np.array(filt_genes):
            strand = gene[6]
            start = int(gene[3]) if strand == '+' else int(gene[4])
            read_cv = np.array(bw_contig.values(gene[0], start-500, start+500))
            read_cv = np.flip(read_cv) if strand == '-' else read_cv
            mat.append(read_cv)
        mat = pd.DataFrame(mat)
        contig_avg_cv = mat.mean().mean()
        contig_mat = scale_mat(mat)
        contig_mat = contig_mat.dropna()
        contig_mean = np.mean(list(map(calc_bias, np.array(contig_mat))))
        
        bias_means.append([asm,sample,ref_mean,contig_mean,ref_avg_cv,contig_avg_cv])
        
bias_means = pd.DataFrame(bias_means)
bias_means.to_csv("plots/hmp2_assembled_ref_bias_bins_checkM.csv")

bias_means = pd.read_csv("plots/hmp2_assembled_ref_bias_bins.csv", index_col = 0)
bias_means.columns = ['asm','sample','Reference genome','Assembled reference','Reference genome cv', 'Assembled reference cv']
sns.boxplot(data=bias_means[['Reference genome','Assembled reference']], showfliers=False, color = 'tab:blue')
add_p_val(plt,0,1,185,3,ttest_rel(bias_means['Reference genome'],bias_means['Assembled reference']).pvalue) # Ttest_relResult(statistic=-2.1892669309449704, pvalue=0.029981782391281703)
plt.ylabel('Average TSS bias')
plt.savefig(f"plots/hmp2_assembled_ref_boxplot_bins.pdf")
plt.clf()
sns.boxplot(data=bias_means[['Reference genome cv', 'Assembled reference cv']], showfliers=False, color = 'tab:blue')
add_p_val(plt,0,1,260,3,ttest_rel(bias_means['Reference genome cv'],bias_means['Assembled reference cv']).pvalue)
plt.savefig(f"plots/hmp2_assembled_ref_boxplot_bins_avg_cv.pdf")
plt.clf()


# Calculate Bias using contigs
bias_means = []
mapped_reads = []
for sample in hmp2_mpa.columns:
    for species in hmp2_mpa.index:
        if np.isnan(hmp2_mpa.loc[species,sample]):
            continue
        asm = hmp2_genomes[species]
        
        # Align Reference genome genes to assembled reference genes in order to match species
        if not os.path.exists(f'minimap_res/{asm}_{sample}_assembled.paf'):
            subprocess.call(f'minimap2 refseq_genomes/{asm}/{asm}_prodigal.fna assembled_ref/{sample}/prodigal.fna > minimap_res/{asm}_{sample}_assembled.paf', shell = True)
        
        # Reference genome bias
        ref_bias_vec = []
        mat = pd.read_csv(f'matrices/{asm}_{sample}.csv', header = None, index_col = 0)
        ref_avg_cv = mat.mean().mean()
        scaled_mat = scale_mat(mat)
        scaled_mat = scaled_mat.dropna()
        ref_no_genes = scaled_mat.shape[0]
        ref_mean = np.mean(list(map(calc_bias, np.array(scaled_mat))))

        # Assembled contig reference
        bw_contig = pyBigWig.open(f'beds/{sample}_assembled_ref.bw')
        contig_dict = bw_contig.chroms()
        minimap_contig_annot = pd.read_table(f'minimap_res/{asm}_{sample}_assembled.paf', header=None, usecols=range(12))
        minimap_contig_annot = minimap_contig_annot[minimap_contig_annot[11]==60] # Filter high quality
        minimap_contig_annot['readName'] = [x.rsplit('_',1)[0] for x in minimap_contig_annot[0]]
        
        # Filter contigs with matching genes with genome
        gff_contig = pd.read_table(f'assembled_ref/{sample}/prodigal.gff', comment = '#', header=None)
        gff_contig = gff_contig[gff_contig[5]>50]
        gff_contig = gff_contig[gff_contig[0].isin(minimap_contig_annot['readName'])]
        gff_contig['partial'] = [re.match('.*partial=(\d\d)',x).group(1) for x in gff_contig[8]]
        gff_contig['len'] = gff_contig[0].map(contig_dict)
        gff_contig['gene_len'] = gff_contig[4] - gff_contig[3]
        gff_contig = gff_contig[gff_contig['len']>10000]
        read_set = list(set(gff_contig[0]))
        gff_contig[3] = gff_contig[3] - 1 # change for 0-indexed
        gff_contig[4] = gff_contig[4] - 1 # change for 0-indexed
        
        if len(read_set) == 0:
            print(f'Skipping {asm} in sample {sample} due to no sufficient length contigs')
            continue
        # Filter overlapping genes
        filt_genes = []
        for contig in read_set:
            genes = np.array(gff_contig[gff_contig[0]==contig])
            filt_genes = filt_genes + filter_overlap_genes(genes,contig_dict).tolist()
        filt_genes = pd.DataFrame(filt_genes)
        filt_genes = filt_genes[(filt_genes[11]>500)&(filt_genes[9]=='00')] #Filter genes shorter than 500bps and are not partial
        #filt_genes = filt_genes.sort_values(5, ascending = False).head(ref_no_genes) # Take top N genes, where N is the number of genes from reference. For fairer comparison
        
        #Gene Coverage at TSS
        mat = []
        for gene in np.array(filt_genes):
            strand = gene[6]
            start = int(gene[3]) if strand == '+' else int(gene[4])
            read_cv = np.array(bw_contig.values(gene[0], start-500, start+500))
            read_cv = np.flip(read_cv) if strand == '-' else read_cv
            mat.append(read_cv)
        mat = pd.DataFrame(mat)
        contig_avg_cv = mat.mean().mean()
        contig_mat = scale_mat(mat)
        contig_mat = contig_mat.dropna()
        contig_no_genes = contig_mat.shape[0]
        contig_mean = np.mean(list(map(calc_bias, np.array(contig_mat))))
        
        bias_means.append([asm,sample,ref_mean,contig_mean])
        mapped_reads.append([asm,sample,ref_no_genes,contig_no_genes,ref_avg_cv,contig_avg_cv])
bias_means = pd.DataFrame(bias_means)
bias_means.to_csv("plots/hmp2_assembled_ref_bias.csv")

mapped_reads = pd.DataFrame(mapped_reads)
sns.boxplot(data=mapped_reads[[4,5]], showfliers=False, color = 'tab:blue')
add_p_val(plt,0,1,105,3,ttest_rel(mapped_reads[4],mapped_reads[5]).pvalue)
plt.savefig(f"plots/hmp2_assembled_ref_boxplot_avg_cv.pdf")
plt.clf()

bias_means = pd.read_csv("plots/hmp2_assembled_ref_bias.csv", header = 0, names = ['asm','sample','ref_bias','contig_bias'])
sns.boxplot(data=bias_means, showfliers=False, color = 'tab:blue')
add_p_val(plt,0,1,185,5,ttest_rel(bias_means['ref_bias'],bias_means['contig_bias']).pvalue) # Ttest_relResult(statistic=-7.110207413007883, pvalue=9.293765094220285e-12)
plt.savefig(f"plots/hmp2_assembled_ref_boxplot.pdf")
plt.clf()
