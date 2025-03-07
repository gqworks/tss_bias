import re, os
import pandas as pd
import pyBigWig
import numpy as np
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import pysam

def taxid2asm(taxid,n=1):
    strain = refseq[(refseq['species_taxid']==int(taxid)) & (refseq['assembly_level']=='Complete Genome')]
    if len(strain) == 0:
        strain = refseq[(refseq['species_taxid']==int(taxid))]
    ftp = strain['ftp_path'][:n].tolist()
    asm = [x.rsplit('/',1)[1] for x in ftp]
    return(asm)
        
def calc_cov_mp(asm, sample, DNA_loc, mp):
    ### Align reads ##
    if not os.path.exists(f'bams/{asm}_{sample}_mp{mp}.sorted.bam'):
        subprocess.call(f'bowtie2 -p 64 -x strains/{asm}/{asm}.fna -1 {DNA_loc}_1.fastq.gz -2 {DNA_loc}_2.fastq.gz --mp {mp} |samtools view --threads 64 -b - |samtools sort - -o bams/{asm}_{sample}_mp{mp}.sorted.bam --threads 64', shell=True)
        
        subprocess.call(f'samtools index bams/{asm}_{sample}_mp{mp}.sorted.bam -@ 64', shell = True)

    #Get read depth of genome
    if not os.path.exists(f'beds/{asm}_{sample}_mp{mp}.bw'):
        subprocess.call(f'bamCoverage --bam bams/{asm}_{sample}_mp{mp}.sorted.bam -p 64 -o beds/{asm}_{sample}_mp{mp}.bw -of bigwig', shell = True)

    ### Generate matrix ###
    if not os.path.exists(f'matrices/{asm}_{sample}_mp{mp}.csv'):
        bp_depth = pyBigWig.open(f'beds/{asm}_{sample}_mp{mp}.bw')
        genome_dict = bp_depth.chroms()
        genomes = list(genome_dict.keys())
        tss_dat = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv', header = 0)
        tss_dat = tss_dat.to_numpy()
        locus_tag = np.array([re.match(r'^ID=([^;]*)',x).group(1) for x in tss_dat[:,4]])
        mat = []
        for row in tss_dat:
            chrom = row[0]
            if (row[5] <= 550) or (row[5] + 550 >= genome_dict[chrom]):
                window = [0]*1001
            else:
                window = bp_depth.values(chrom,int(row[5])-501,int(row[5])+500)
                if row[3]=="-": window.reverse()
            mat.append(window)
        mat = np.array(mat)
        x = np.hstack((locus_tag[np.newaxis].T,mat))
        np.savetxt(f'matrices/{asm}_{sample}_mp{mp}.csv',x,fmt='%s',delimiter=',')

def scale_mat(d):
    return (d-d.mean(axis=1)[:,None]) / d.std(axis=1)[:,None]

def calc_bias(x):
    left_sum = np.sum(x[:500])
    right_sum = np.sum(x[500:])
    return(right_sum-left_sum)

def add_p_val(ax,lft,rgt,y,h,p):
    ax.plot([lft, lft, rgt, rgt], [y, y+h, y+h, y], lw=1.5, c='k')
    ax.text((lft + rgt) * .5, y+h, ('n.s.' if p > 0.15 else 'p < %.2g' if p > 0.001 else 'p < %.1g') % max(p+1e-20, 1e-20), ha='center', va='bottom', color='k')

def get_tpm(file):
    dat = pd.read_table(file)
    dat = dat.drop_duplicates(subset='target_id')
    dat = dat[['target_id','tpm']]
    dat.index = dat['target_id']
    dat = dat.drop(columns = ['target_id'])
    return dat

READ_LEN = 150
SMALL_BUFFER = 5
def downstream_NM(bam_file, gene_annot):
    NM_scores = []
    for gene in np.array(gene_annot):
        if gene[3] == '+':
            left = int(gene[5]) + READ_LEN - SMALL_BUFFER
            right = int(gene[5]) + 500
        else:
            left = int(gene[5]) - 501
            right = int(gene[5]) - READ_LEN + SMALL_BUFFER
        NM_set = []
        for read in bam_file.fetch(gene[0],left,right):
            if not read.is_unmapped and read.mapping_quality>=40:
                NM_set.append(read.get_cigar_stats()[0][10])
        if len(NM_set) == 0:
            NM_scores.append(np.nan)
        else:
            NM_scores.append(np.mean(NM_set))
    return(NM_scores)

def upstream_NM(bam_file, gene_annot):
    NM_scores = []
    for gene in np.array(gene_annot):
        if gene[3] == '+':
            left = int(gene[5]) - 500 
            right = int(gene[5]) - READ_LEN + SMALL_BUFFER
        else:
            left = int(gene[5]) + READ_LEN - SMALL_BUFFER
            right = int(gene[5]) + 500
        NM_set = []
        for read in bam_file.fetch(gene[0],left,right):
            if not read.is_unmapped and read.mapping_quality>=40:
                NM_set.append(read.get_cigar_stats()[0][10])
        if len(NM_set) == 0:
            NM_scores.append(np.nan)
        else:
            NM_scores.append(np.mean(NM_set))
    return(NM_scores)

def window_NM(bam_file, gene_annot):
    NM_scores = []
    for gene in np.array(gene_annot):
        left = int(gene[5]) - 501
        right = int(gene[5]) + 500

        NM_set = []
        for read in bam_file.fetch(gene[0],left,right):
            if not read.is_unmapped and read.mapping_quality>=40:
                NM_set.append(read.get_cigar_stats()[0][10])
        if len(NM_set) == 0:
            continue
        NM_scores.append(np.mean(NM_set))
    return(NM_scores)

def unique_MAPQ(bam_file, gene_annot):
    count = 0
    for gene in np.array(gene_annot):
        left = int(gene[5]) - 501
        right = int(gene[5]) + 500
        for read in bam_file.fetch(gene[0],left,right):
            if(not read.is_unmapped and read.mapping_quality>=40):
                count += 1
    return(count)


### Read NCBI Refseq table ###
refseq = pd.read_table('src/assembly_summary_refseq.txt', header = 1)

### Read species abundances ###
mpa_res = pd.read_table('metaphlan3_res/merged_mpa.txt', comment= '#')
mpa_res = mpa_res[mpa_res['clade_name'].str.contains("s__")]
mpa_res.index = [int(re.match('.*\|([^\|]*)',x).group(1)) for x in mpa_res['NCBI_tax_id']]
mpa_res = mpa_res.drop(columns=['clade_name', 'NCBI_tax_id'])

### Franzosa dataset (8 metagenomics and metatranscriptomics matched samples) ###
with open('Franzosa_8_samples') as f:
    Franzosa_8 = f.read().splitlines()
Franzosa_8 = {x.split('\t')[0]:x.split('\t')[1] for x in Franzosa_8}
Franzosa_mpa = mpa_res.loc[:,Franzosa_8.keys()]
Franzosa_mpa = Franzosa_mpa.loc[((Franzosa_mpa > 1).sum(axis=1) >= 5),:] 
Franzosa_mpa[Franzosa_mpa < 1] = np.nan
Franzosa_mpa = Franzosa_mpa.drop(39491) #Eubacterium Rectale no reference in database.
            
mp_parameters = ['8','4','2']
for sample in Franzosa_mpa.columns:
    DNA_loc = Franzosa_8[sample]
    for taxid in Franzosa_mpa.index:
        asm = taxid2asm(taxid,1)[0]
        for mp in mp_parameters:
            calc_cov_mp(asm,sample,DNA_loc,mp)

# Check Bias and plot
all_bias = []
all_cv = []
for sample in Franzosa_mpa.columns:
    for taxid in Franzosa_mpa.index:
        if np.isnan(Franzosa_mpa.loc[taxid,sample]):
            continue
        asm = taxid2asm(taxid,1)[0]
        
        # MP=8 TSS bias
        mat = pd.read_table(f'matrices/{asm}_{sample}_mp8.csv', sep = ',', header=None, index_col=0)
        scaled_mat = scale_mat(mat)
        scaled_mat = scaled_mat.fillna(0)
        mp8_bias = np.mean(list(map(calc_bias, np.array(scaled_mat))))
        mp8_cv = mat.mean().mean()
        
        # MP=6 bias
        mat = pd.read_table(f'matrices/{asm}_{sample}.csv', sep = ',', header=None, index_col=0)
        scaled_mat = scale_mat(mat)
        scaled_mat = scaled_mat.fillna(0)
        mp6_bias = np.mean(list(map(calc_bias, np.array(scaled_mat))))
        mp6_cv = mat.mean().mean()
        
        # MP=4 TSS bias
        mat = pd.read_table(f'matrices/{asm}_{sample}_mp4.csv', sep = ',', header=None, index_col=0)
        scaled_mat = scale_mat(mat)
        scaled_mat = scaled_mat.fillna(0)
        mp4_bias = np.mean(list(map(calc_bias, np.array(scaled_mat))))
        mp4_cv = mat.mean().mean()
        
        # MP=2 TSS bias
        mat = pd.read_table(f'matrices/{asm}_{sample}_mp2.csv', sep = ',', header=None, index_col=0)
        scaled_mat = scale_mat(mat)
        scaled_mat = scaled_mat.fillna(0)
        mp2_bias = np.mean(list(map(calc_bias, np.array(scaled_mat))))
        mp2_cv = mat.mean().mean()
        
        all_bias.append([sample, asm, mp8_bias, mp6_bias, mp4_bias, mp2_bias])
        all_cv.append([sample, asm, mp8_cv, mp6_cv, mp4_cv, mp2_cv])
        
all_bias = pd.DataFrame(all_bias, columns = ['sample','asm','8','6 (default)','4','2'])        
all_bias = all_bias.melt(id_vars = ['sample','asm'])

sns.boxplot(data = all_bias, x = 'variable', y= 'value', order= ['8','6 (default)','4','2'], showfliers=False)
add_p_val(plt,0,1,80,1,ttest_rel(all_bias.loc[(all_bias.variable=='8'),'value'],all_bias.loc[(all_bias.variable=='6 (default)'),'value']).pvalue)
add_p_val(plt,1,2,90,1,ttest_rel(all_bias.loc[(all_bias.variable=='6 (default)'),'value'],all_bias.loc[(all_bias.variable=='4'),'value']).pvalue)
add_p_val(plt,2,3,85,1,ttest_rel(all_bias.loc[(all_bias.variable=='4'),'value'],all_bias.loc[(all_bias.variable=='2'),'value']).pvalue)
plt.xlabel('Mismatch penalty')
plt.ylabel('TSS Bias')
plt.savefig(f"plots/relaxed_aligner/all_bias_mp.pdf")
plt.clf()

np.mean(all_bias.loc[(all_bias.variable=='8'),'value']) # 17.57
np.mean(all_bias.loc[(all_bias.variable=='6 (default)'),'value']) # 37.09
np.mean(all_bias.loc[(all_bias.variable=='4'),'value']) # 35.56
np.mean(all_bias.loc[(all_bias.variable=='2'),'value']) # 31.64

all_cv = pd.DataFrame(all_cv, columns = ['sample','asm','8','6 (default)','4','2'])        
all_cv = all_cv.melt(id_vars = ['sample','asm'])

sns.boxplot(data = all_cv, x = 'variable', y= 'value', order= ['8','6 (default)','4','2'], showfliers=False)
add_p_val(plt,0,1,70,1,ttest_rel(all_cv.loc[(all_cv.variable=='8'),'value'],all_cv.loc[(all_bias.variable=='6 (default)'),'value']).pvalue)
add_p_val(plt,1,2,75,1,ttest_rel(all_cv.loc[(all_cv.variable=='6 (default)'),'value'],all_cv.loc[(all_bias.variable=='4'),'value']).pvalue)
add_p_val(plt,2,3,80,1,ttest_rel(all_cv.loc[(all_cv.variable=='4'),'value'],all_cv.loc[(all_bias.variable=='2'),'value']).pvalue)
plt.xlabel('Mismatch penalty')
plt.ylabel('Average TSS window coverage')
plt.savefig(f"plots/relaxed_aligner/all_cv_mp.pdf")
plt.clf()

np.mean(all_cv.loc[(all_cv.variable=='8'),'value']) # 9
np.mean(all_cv.loc[(all_cv.variable=='6 (default)'),'value']) # 26.37
np.mean(all_cv.loc[(all_cv.variable=='4'),'value']) # 27.48
np.mean(all_cv.loc[(all_cv.variable=='2'),'value']) # 28.26

### Check number of mismatches ###
mapq_df = []
for sample in Franzosa_mpa.columns:
    for taxid in Franzosa_mpa.index:
        if np.isnan(Franzosa_mpa.loc[taxid,sample]):
            continue
        asm = taxid2asm(taxid,1)[0]
        gene_annot = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
        gene_annot = gene_annot[(gene_annot.Strand == '-') & (gene_annot.Start > 750)] #Incase of fetching beyond start of genome
        
        bam_file = pysam.AlignmentFile(f'bams/{asm}_{sample}.sorted.bam', "rb")
        unique_count = unique_MAPQ(bam_file,gene_annot)
        NM_avg = np.mean(window_NM(bam_file, gene_annot))
        mapq_df.append([sample, asm, '6 (default)', unique_count, NM_avg])
        for mp in mp_parameters:
            bam_file = pysam.AlignmentFile(f'bams/{asm}_{sample}_mp{mp}.sorted.bam', "rb")
            unique_count = unique_MAPQ(bam_file,gene_annot)
            NM_avg = np.mean(window_NM(bam_file, gene_annot))
            mapq_df.append([sample, asm, mp, unique_count, NM_avg])
mapq_df = pd.DataFrame(mapq_df, columns = ['sample','asm','mp','mean_MAPQ', 'unique_count', 'NM_avg'])     

sns.boxplot(data = mapq_df, x = 'mp', y= 'unique_count', order= ['8', '6 (default)','4','2'], showfliers=False)
add_p_val(plt,0,1,475000,10000,ttest_rel(mapq_df.loc[(mapq_df.mp=='8'),'unique_count'],mapq_df.loc[(mapq_df.mp=='6 (default)'),'unique_count']).pvalue)
add_p_val(plt,1,2,500000,10000,ttest_rel(mapq_df.loc[(mapq_df.mp=='6 (default)'),'unique_count'],mapq_df.loc[(mapq_df.mp=='4'),'unique_count']).pvalue)
add_p_val(plt,2,3,525000,10000,ttest_rel(mapq_df.loc[(mapq_df.mp=='4'),'unique_count'],mapq_df.loc[(mapq_df.mp=='2'),'unique_count']).pvalue)
plt.xlabel('Mismatch penalty')
plt.ylabel('Total unique reads')
plt.tight_layout()
plt.savefig(f"plots/relaxed_aligner/all_unique_reads_mp.pdf")
plt.clf()

sns.boxplot(data = mapq_df, x = 'mp', y= 'NM_avg', order= ['8', '6 (default)','4','2'], showfliers=False)
add_p_val(plt,0,1,4,0.1,ttest_rel(mapq_df.loc[(mapq_df.mp=='8'),'NM_avg'],mapq_df.loc[(mapq_df.mp=='6 (default)'),'NM_avg']).pvalue)
add_p_val(plt,1,2,4.5,0.1,ttest_rel(mapq_df.loc[(mapq_df.mp=='6 (default)'),'NM_avg'],mapq_df.loc[(mapq_df.mp=='4'),'NM_avg']).pvalue)
add_p_val(plt,2,3,5,0.1,ttest_rel(mapq_df.loc[(mapq_df.mp=='4'),'NM_avg'],mapq_df.loc[(mapq_df.mp=='2'),'NM_avg']).pvalue)
plt.xlabel('Mismatch penalty')
plt.ylabel('Average substition count')
plt.tight_layout()
plt.savefig("plots/relaxed_aligner/all_NM_score.pdf")
plt.clf()

# Higher conservation of sequence in highly expressed genes.
# Check read NM score of reads downstream of TSS across expression group
all_metrics = pd.DataFrame()
for sample in Franzosa_mpa.columns:
    for taxid in Franzosa_mpa.index:
        if np.isnan(Franzosa_mpa.loc[taxid,sample]):
            continue
        asm = taxid2asm(taxid,1)[0]
        
        #Read TPM sample data
        tpm_file = get_tpm(f'kallisto_res/{asm}_{sample}/abundance.tsv')
        chrom_map = pd.read_table(f'strains/{asm}/{asm}_filtered_tss.csv', sep = ',', header=0, index_col=0)
        chrom_map = chrom_map[~chrom_map.index.duplicated(keep='first')]
        chrom_map = {chrom:re.match(r'ID=(\d*)_',chrom_map.loc[chrom,'Desc']).group(1) for chrom in chrom_map.index}
        tpm_file.index = [x.replace(x.rsplit('_',1)[0],chrom_map.get(x.rsplit('_',1)[0],'na')) for x in tpm_file.index]
        tpm_dict = tpm_file.to_dict()

        #Get gene No. mismatch
        gene_annot = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
        gene_annot = gene_annot[(gene_annot.Strand == '-') & (gene_annot.Start > 750)] #Incase of fetching beyond start of genome
        gene_annot['gene_id'] = [re.match(r'ID=(\d+_\d+);',x).group(1) for x in gene_annot['Desc']]
        bam_file = pysam.AlignmentFile(f'bams/{asm}_{sample}_mp2.sorted.bam', "rb")
        gene_annot['downstream_NM'] = downstream_NM(bam_file, gene_annot)
        gene_annot['upstream_NM'] = upstream_NM(bam_file, gene_annot)
        gene_annot['tpm'] = [tpm_dict['tpm'].get(x,np.nan) for x in gene_annot.gene_id]
        tol_mapped_reads = np.sum([int(x.split('\t')[2]) for x in pysam.idxstats(f'bams/{asm}_{sample}.sorted.bam').split('\n')[:-1]])
        gene_annot['norm_tpm'] = 1000000*gene_annot['tpm']/tol_mapped_reads
        gene_annot['tpm_group'] = np.where(gene_annot['norm_tpm'] == 0, "zero", np.where(gene_annot['norm_tpm'] < 100, 'low', np.where(gene_annot['norm_tpm'] < 1000, 'mid', 'high')))
        
        summary = gene_annot[['tpm_group','downstream_NM','upstream_NM']].groupby('tpm_group').mean().reset_index()
        summary['asm'] = asm
        summary['sample'] = sample
        all_metrics = pd.concat([all_metrics,summary])

sns.boxplot(data=all_metrics, y="downstream_NM", x="tpm_group", order = ['zero','low','mid','high'], showfliers=False)
add_p_val(plt,0,1,7,0.2,ttest_rel(all_metrics.loc[(all_metrics.tpm_group=='zero'),'downstream_NM'],all_metrics.loc[(all_metrics.tpm_group=='low'),'downstream_NM']).pvalue)
add_p_val(plt,1,2,6,0.2,ttest_rel(all_metrics.loc[(all_metrics.tpm_group=='low'),'downstream_NM'],all_metrics.loc[(all_metrics.tpm_group=='mid'),'downstream_NM']).pvalue)
add_p_val(plt,2,3,5,0.2,ttest_rel(all_metrics.loc[(all_metrics.tpm_group=='mid'),'downstream_NM'],all_metrics.loc[(all_metrics.tpm_group=='high'),'downstream_NM']).pvalue)
add_p_val(plt,0,2,8,0.2,ttest_rel(all_metrics.loc[(all_metrics.tpm_group=='zero'),'downstream_NM'],all_metrics.loc[(all_metrics.tpm_group=='mid'),'downstream_NM']).pvalue)
add_p_val(plt,0,3,9,0.2,ttest_rel(all_metrics.loc[(all_metrics.tpm_group=='zero'),'downstream_NM'],all_metrics.loc[(all_metrics.tpm_group=='high'),'downstream_NM']).pvalue)
plt.xlabel('Expression group')
plt.ylabel('Average downstream substition count')
plt.tight_layout()
plt.savefig("plots/relaxed_aligner/all_exp_vs_downstream_NM.pdf")
plt.clf()

all_metrics[all_metrics["tpm_group"]=='zero']['downstream_NM'].mean() #2.9014570972154563
all_metrics[all_metrics["tpm_group"]=='low']['downstream_NM'].mean() #2.4870814296038137
all_metrics[all_metrics["tpm_group"]=='mid']['downstream_NM'].mean() #2.4614125073619504
all_metrics[all_metrics["tpm_group"]=='high']['downstream_NM'].mean() #2.4177797687674443

sns.boxplot(data=all_metrics, y="upstream_NM", x="tpm_group", order = ['zero','low','mid','high'], showfliers=False)
add_p_val(plt,0,1,7,0.2,ttest_rel(all_metrics.loc[(all_metrics.tpm_group=='zero'),'upstream_NM'],all_metrics.loc[(all_metrics.tpm_group=='low'),'upstream_NM']).pvalue)
add_p_val(plt,1,2,6,0.2,ttest_rel(all_metrics.loc[(all_metrics.tpm_group=='low'),'upstream_NM'],all_metrics.loc[(all_metrics.tpm_group=='mid'),'upstream_NM']).pvalue)
add_p_val(plt,2,3,5,0.2,ttest_rel(all_metrics.loc[(all_metrics.tpm_group=='mid'),'upstream_NM'],all_metrics.loc[(all_metrics.tpm_group=='high'),'upstream_NM']).pvalue)
add_p_val(plt,0,2,8,0.2,ttest_rel(all_metrics.loc[(all_metrics.tpm_group=='zero'),'upstream_NM'],all_metrics.loc[(all_metrics.tpm_group=='mid'),'upstream_NM']).pvalue)
add_p_val(plt,0,3,9,0.2,ttest_rel(all_metrics.loc[(all_metrics.tpm_group=='zero'),'upstream_NM'],all_metrics.loc[(all_metrics.tpm_group=='high'),'upstream_NM']).pvalue)
plt.xlabel('Expression group')
plt.ylabel('Average upstream substition count')
plt.tight_layout()
plt.savefig("plots/relaxed_aligner/all_exp_vs_upstream_NM.pdf")
plt.clf()

all_metrics[all_metrics["tpm_group"]=='zero']['upstream_NM'].mean() #2.9954283245669124
all_metrics[all_metrics["tpm_group"]=='low']['upstream_NM'].mean() #2.50103148652234
all_metrics[all_metrics["tpm_group"]=='mid']['upstream_NM'].mean() #2.4851037375438825
all_metrics[all_metrics["tpm_group"]=='high']['upstream_NM'].mean() #2.460051226537761


