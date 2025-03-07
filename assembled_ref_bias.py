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

def taxid2asm(taxid,n=1):
    strain = refseq[(refseq['species_taxid']==int(taxid)) & (refseq['assembly_level']=='Complete Genome')]
    if len(strain) == 0:
        strain = refseq[(refseq['species_taxid']==int(taxid))]
    ftp = strain['ftp_path'][:n].tolist()
    asm = [x.rsplit('/',1)[1] for x in ftp]
    return(asm)

def asm_2_species(asm):
    name = refseq[refseq['ftp_path'].str.contains(asm)]['organism_name'].item()
    return re.match(r'(^[^ ]+ [^ ]+).*',name).group(1)

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

    
### Read NCBI Refseq table ###
refseq = pd.read_table('src/assembly_summary_refseq.txt', header = 1)

### Franzosa dataset (8 metagenomics and metatranscriptomics matched samples) ###
with open('Franzosa_8_samples') as f:
    Franzosa_8 = f.read().splitlines()
Franzosa_8 = {x.split('\t')[0]:x.split('\t')[1] for x in Franzosa_8}

# Assemble samples
#for sample,path in Franzosa_8.items():
#    subprocess.call(f'ml megahit && megahit -t 64 -1 {path}_1.fastq.gz -2 {path}_2.fastq.gz -o assembled_ref/{sample}', shell = True)
#    subprocess.call(f'bowtie2-build assembled_ref/{sample}/final.contigs.fa assembled_ref/{sample}/final.contigs.fa', shell=True)

# Align reads to assembled reference and bin
for sample,path in Franzosa_8.items():
    if not os.path.exists(f'bams/{sample}_assembled_ref.sorted.bam'):
        subprocess.call(f'bowtie2 -p 64 -x assembled_ref/{sample}/final.contigs.fa -1 {path}_1.fastq.gz -2 {path}_2.fastq.gz |samtools view --threads 64 -b - |samtools sort - -o bams/{sample}_assembled_ref.sorted.bam --threads 64', shell = True)
        subprocess.call(f'samtools index bams/{sample}_assembled_ref.sorted.bam -@ 64', shell = True)

    # Get read depth of genome
    #if not os.path.exists(f'beds/{sample}_assembled_ref.bw'):
    #    subprocess.call(f'bamCoverage --bam bams/{sample}_assembled_ref.sorted.bam -p 64 -o beds/{sample}_assembled_ref.bw -of bigwig', shell = True)
    
    # Annotate genes w/ prodigal
    #if not os.path.exists(f'assembled_ref/{sample}/prodigal.gff'):
    #    subprocess.call(f'prodigal -i assembled_ref/{sample}/final.contigs.fa -o assembled_ref/{sample}/prodigal.gff -f gff -p meta -d assembled_ref/{sample}/prodigal.fna', shell = True)
    
    # Bin Contigs
    #subprocess.call(f'ml MetaBAT && jgi_summarize_bam_contig_depths --outputDepth assembled_ref/{sample}/depth.txt bams/{sample}_assembled_ref.sorted.bam', shell = True)
    #subprocess.call(f'ml MetaBAT && metabat2 -t 64 -i assembled_ref/{sample}/final.contigs.fa -a assembled_ref/{sample}/depth.txt -o assembled_ref/{sample}/metabat/bin', shell = True)
    
    # Annotate contig bins with Kraken2
    #if not os.path.exists(f'assembled_ref/{sample}/kraken_res/'):
    #    os.mkdir(f'assembled_ref/{sample}/kraken_res/')
    #for i in glob.glob(f'assembled_ref/{sample}/metabat/*.fa'):
    #    bin_no = re.match('.*(bin\.\d+)\.fa',i).group(1)
    #    subprocess.call(f'ml ncbi-blast kraken && kraken2 --threads 64 --db /groups/cgsd/hbyao/db/bfv/ --output assembled_ref/{sample}/kraken_res/{bin_no}.kra --report assembled_ref/{sample}/kraken_res/{bin_no}.report {i}', shell = True)  

# Find number of reads aligned
read_count_df = []
for sample,path in Franzosa_8.items():
    bamfile = f'bams/{sample}_assembled_ref.sorted.bam'
    indStats = pd.read_csv(StringIO(pysam.idxstats(bamfile)), sep = '\t', header = None, names = ['contig', 'length', 'mapped', 'unmapped'])
    tot_mapped = indStats.mapped.sum()
    tot_unmapped = indStats.unmapped.sum()
    read_count_df.append([sample,tot_mapped,tot_unmapped])
    
# Get Refseq NCBI genomes
mpa_res = pd.read_table('metaphlan3_res/merged_mpa.txt', comment= '#')
mpa_res = mpa_res[mpa_res['clade_name'].str.contains("s__")]
mpa_res.index = [int(re.match('.*\|([^\|]*)',x).group(1)) for x in mpa_res['NCBI_tax_id']]
mpa_res = mpa_res.drop(columns=['clade_name', 'NCBI_tax_id'])
Franzosa_mpa = mpa_res.loc[:,Franzosa_8.keys()]
Franzosa_mpa = Franzosa_mpa.loc[((Franzosa_mpa > 1).sum(axis=1) >= 5),:] # Filter species with atleast 1% abundance in atleast 5 samples
Franzosa_mpa[Franzosa_mpa < 1] = np.nan
Franzosa_mpa = Franzosa_mpa.drop(39491)

# Calculate Bias using bins
bias_means = []
for sample in Franzosa_mpa.columns:
    bw_contig = pyBigWig.open(f'beds/{sample}_assembled_ref.bw')
    contig_dict = bw_contig.chroms()
    for taxid in Franzosa_mpa.index:
        if np.isnan(Franzosa_mpa.loc[taxid,sample]):
            continue
        asm = taxid2asm(taxid,1)[0]
        binno = find_best_bin(taxid,sample)
        if binno == 0:
            print(f'No high quality bin found for {taxid} in {sample}')
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
bias_means.to_csv("plots/assembled_ref_bias_bins.csv")

bias_means = pd.read_csv("plots/assembled_ref_bias_bins.csv", index_col = 0)
bias_means.columns = ['asm','sample','Reference genome','Assembled reference','Reference genome cv', 'Assembled reference cv']
sns.boxplot(data=bias_means[['Reference genome','Assembled reference']], showfliers=False, color = 'tab:blue')
add_p_val(plt,0,1,95,3,ttest_rel(bias_means['Reference genome'],bias_means['Assembled reference']).pvalue)
plt.ylabel('Average TSS bias')
plt.savefig(f"plots/assembled_ref_boxplot_bins.pdf")
plt.clf()
sns.boxplot(data=bias_means[[4,5]], showfliers=False, color = 'tab:blue')
add_p_val(plt,0,1,80,3,ttest_rel(bias_means[4],bias_means[5]).pvalue)
plt.savefig(f"plots/assembled_ref_boxplot_bins_avg_cv.pdf")
plt.clf()


# Calculate Bias using contigs
bias_means = []
mapped_reads = []
for sample in Franzosa_mpa.columns:
    for taxid in Franzosa_mpa.index:
        if np.isnan(Franzosa_mpa.loc[taxid,sample]):
            continue
        asm = taxid2asm(taxid,1)[0]
        # Align Reference genome genes to assembled reference genes in order to match species
        if not os.path.exists(f'minimap_res/{asm}_{sample}_assembled.paf'):
            subprocess.call(f'minimap2 strains/{asm}/{asm}_prodigal.fna assembled_ref/{sample}/prodigal.fna > minimap_res/{asm}_{sample}_assembled.paf', shell = True)
        
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
        gff_contig = gff_contig[gff_contig['len']>20000]
        read_set = list(set(gff_contig[0]))
        gff_contig[3] = gff_contig[3] - 1 # change for 0-indexed
        gff_contig[4] = gff_contig[4] - 1 # change for 0-indexed

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
bias_means.to_csv("plots/assembled_ref_bias.csv")

mapped_reads = pd.DataFrame(mapped_reads)
sns.boxplot(data=mapped_reads[[4,5]], showfliers=False, color = 'tab:blue')
add_p_val(plt,0,1,105,3,ttest_rel(mapped_reads[4],mapped_reads[5]).pvalue)
plt.savefig(f"plots/assembled_ref_boxplot_avg_cv.pdf")
plt.clf()

#bias_means = pd.read_csv("plots/assembled_ref_bias.csv", header = 0, names = ['asm','sample','ref_bias','contig_bias'])
sns.boxplot(data=bias_means, showfliers=False, color = 'tab:blue')
#plt.ylim(-35, 120)
add_p_val(plt,0,1,105,3,ttest_rel(bias_means[2],bias_means[3]).pvalue) # Ttest_relResult(statistic=4.0750025977880835, pvalue=0.0003262267921443576)
plt.savefig(f"plots/assembled_ref_boxplot.pdf")
plt.clf()
