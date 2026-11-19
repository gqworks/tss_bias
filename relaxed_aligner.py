import re, os
import pandas as pd
import pyBigWig
import numpy as np
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
import pysam
        
def calc_cov_mp(asm, sample, DNA_loc, mp):
    ### Align reads ##
    if not os.path.exists(f'bams/{asm}_{sample}_mp{mp}.sorted.bam'):
        subprocess.call(f'bowtie2 -p 64 -x refseq_genomes/{asm}/{asm}.fna -1 {DNA_loc}_1.fastq.gz -2 {DNA_loc}_2.fastq.gz --mp {mp} |samtools view --threads 64 -b - |samtools sort - -o bams/{asm}_{sample}_mp{mp}.sorted.bam --threads 64', shell=True)
        
        subprocess.call(f'samtools index bams/{asm}_{sample}_mp{mp}.sorted.bam -@ 64', shell = True)

    #Get read depth of genome
    if not os.path.exists(f'beds/{asm}_{sample}_mp{mp}.bw'):
        subprocess.call(f'bamCoverage --bam bams/{asm}_{sample}_mp{mp}.sorted.bam -p 64 -o beds/{asm}_{sample}_mp{mp}.bw -of bigwig', shell = True)

    ### Generate matrix ###
    if not os.path.exists(f'matrices/{asm}_{sample}_mp{mp}.csv'):
        bp_depth = pyBigWig.open(f'beds/{asm}_{sample}_mp{mp}.bw')
        genome_dict = bp_depth.chroms()
        genomes = list(genome_dict.keys())
        tss_dat = pd.read_csv(f'refseq_genomes/{asm}/{asm}_filtered_tss.csv', header = 0)
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
    mean_vals = d.mean(axis=1).to_numpy()
    std_vals = d.std(axis=1).to_numpy()
    return (d - mean_vals[:, None]) / std_vals[:, None]

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
mp_parameters = ['8','4','2']


for sample, path in hmp2_samples.items():
    DNA_loc = f'{path}/{sample}'
    for species in hmp2_mpa.index:
        if np.isnan(hmp2_mpa.loc[species,sample]):
            continue
        asm = hmp2_genomes[species]
        for mp in mp_parameters:
            calc_cov_mp(asm,sample,DNA_loc,mp)

# Check Bias and plot
all_bias = []
all_cv = []
all_cv_split = []
for sample in hmp2_mpa.columns:
    for species in hmp2_mpa.index:
        if np.isnan(hmp2_mpa.loc[species,sample]):
            continue
        asm = hmp2_genomes[species]
        
        # MP=8 TSS bias
        mat = pd.read_table(f'matrices/{asm}_{sample}_mp8.csv', sep = ',', header=None, index_col=0)
        scaled_mat = scale_mat(mat)
        scaled_mat = scaled_mat.fillna(0)
        mp8_bias = np.mean(list(map(calc_bias, np.array(scaled_mat))))
        mp8_cv = mat.mean().mean()
        mp8_cv_up = mat.iloc[:,:500].mean().mean()
        mp8_cv_down = mat.iloc[:,500:].mean().mean()
        
        # MP=6 bias
        mat = pd.read_table(f'matrices/{asm}_{sample}.csv', sep = ',', header=None, index_col=0)
        scaled_mat = scale_mat(mat)
        scaled_mat = scaled_mat.fillna(0)
        mp6_bias = np.mean(list(map(calc_bias, np.array(scaled_mat))))
        mp6_cv = mat.mean().mean()
        mp6_cv_up = mat.iloc[:,:500].mean().mean()
        mp6_cv_down = mat.iloc[:,500:].mean().mean()
        
        # MP=4 TSS bias
        mat = pd.read_table(f'matrices/{asm}_{sample}_mp4.csv', sep = ',', header=None, index_col=0)
        scaled_mat = scale_mat(mat)
        scaled_mat = scaled_mat.fillna(0)
        mp4_bias = np.mean(list(map(calc_bias, np.array(scaled_mat))))
        mp4_cv = mat.mean().mean()
        mp4_cv_up = mat.iloc[:,:500].mean().mean()
        mp4_cv_down = mat.iloc[:,500:].mean().mean()
        
        # MP=2 TSS bias
        mat = pd.read_table(f'matrices/{asm}_{sample}_mp2.csv', sep = ',', header=None, index_col=0)
        scaled_mat = scale_mat(mat)
        scaled_mat = scaled_mat.fillna(0)
        mp2_bias = np.mean(list(map(calc_bias, np.array(scaled_mat))))
        mp2_cv = mat.mean().mean()
        mp2_cv_up = mat.iloc[:,:500].mean().mean()
        mp2_cv_down = mat.iloc[:,500:].mean().mean()
        
        all_bias.append([sample, asm, mp8_bias, mp6_bias, mp4_bias, mp2_bias])
        all_cv.append([sample, asm, mp8_cv, mp6_cv, mp4_cv, mp2_cv])
        all_cv_split.append([sample, asm, 'upstream', mp8_cv_up, mp6_cv_up, mp4_cv_up, mp2_cv_up])
        all_cv_split.append([sample, asm, 'downstream', mp8_cv_down, mp6_cv_down, mp4_cv_down, mp2_cv_down])
        
all_bias = pd.DataFrame(all_bias, columns = ['sample','asm','8','6 (default)','4','2'])
all_bias.to_csv('plots/hmp2_relaxed_aligner_bias.csv',index=False)
all_bias = pd.read_csv('plots/hmp2_relaxed_aligner_bias.csv')
avg_all_bias = all_bias.drop(columns=['asm']).groupby('sample', as_index=False).mean()
avg_all_bias = avg_all_bias.melt(id_vars=["sample"])

sns.boxplot(data = avg_all_bias, x = 'variable', y= 'value', order= ['8','6 (default)','4','2'], showfliers=False)
add_p_val(plt,0,1,90,1,ttest_rel(avg_all_bias.loc[(avg_all_bias.variable=='8'),'value'],avg_all_bias.loc[(avg_all_bias.variable=='6 (default)'),'value']).pvalue)
add_p_val(plt,1,2,85,1,ttest_rel(avg_all_bias.loc[(avg_all_bias.variable=='6 (default)'),'value'],avg_all_bias.loc[(avg_all_bias.variable=='4'),'value']).pvalue)
add_p_val(plt,2,3,80,1,ttest_rel(avg_all_bias.loc[(avg_all_bias.variable=='4'),'value'],avg_all_bias.loc[(avg_all_bias.variable=='2'),'value']).pvalue)
plt.xlabel('Mismatch penalty')
plt.ylabel('TSS Bias')
plt.savefig(f"plots/relaxed_aligner/hmp2_bias_mp_avg.pdf")
plt.clf()

ttest_rel(avg_all_bias.loc[(avg_all_bias.variable=='6 (default)'),'value'],avg_all_bias.loc[(avg_all_bias.variable=='2'),'value']) # Ttest_relResult(statistic=14.773761515211575, pvalue=4.808036904037771e-24)
np.mean(avg_all_bias.loc[(avg_all_bias.variable=='8'),'value']) # 41.86330575304684
np.mean(avg_all_bias.loc[(avg_all_bias.variable=='6 (default)'),'value']) # 41.867687286038084
np.mean(avg_all_bias.loc[(avg_all_bias.variable=='4'),'value']) # 40.71469869031312
np.mean(avg_all_bias.loc[(avg_all_bias.variable=='2'),'value']) # 36.310408635893566

all_cv = pd.DataFrame(all_cv, columns = ['sample','asm','8','6 (default)','4','2'])       
all_cv.to_csv('plots/hmp2_relaxed_aligner_cv.csv',index=False)
all_cv = pd.read_csv('plots/hmp2_relaxed_aligner_cv.csv')
avg_all_cv = all_cv.drop(columns=['asm']).groupby('sample', as_index=False).mean()
avg_all_cv = avg_all_cv.melt(id_vars=["sample"])

sns.boxplot(data = avg_all_cv, x = 'variable', y= 'value', order= ['8','6 (default)','4','2'], showfliers=False)
add_p_val(plt,0,1,170,1,ttest_rel(avg_all_cv.loc[(avg_all_cv.variable=='8'),'value'],avg_all_cv.loc[(avg_all_cv.variable=='6 (default)'),'value']).pvalue)
add_p_val(plt,1,2,185,1,ttest_rel(avg_all_cv.loc[(avg_all_cv.variable=='6 (default)'),'value'],avg_all_cv.loc[(avg_all_cv.variable=='4'),'value']).pvalue)
add_p_val(plt,2,3,200,1,ttest_rel(avg_all_cv.loc[(avg_all_cv.variable=='4'),'value'],avg_all_cv.loc[(avg_all_cv.variable=='2'),'value']).pvalue)
plt.xlabel('Mismatch penalty')
plt.ylabel('Average TSS window coverage')
plt.savefig(f"plots/relaxed_aligner/hmp2_cv_mp_avg.pdf")
plt.clf()

ttest_rel(avg_all_cv.loc[(avg_all_cv.variable=='6 (default)'),'value'],avg_all_cv.loc[(avg_all_cv.variable=='2'),'value']) # Ttest_relResult(statistic=-11.027066379269675, pvalue=1.9653953756699703e-17)
np.median(avg_all_cv.loc[(avg_all_cv.variable=='8'),'value']) # 37.60730732869865
np.median(avg_all_cv.loc[(avg_all_cv.variable=='6 (default)'),'value']) # 40.5310943133888
np.median(avg_all_cv.loc[(avg_all_cv.variable=='4'),'value']) # 44.16155495046232
np.median(avg_all_cv.loc[(avg_all_cv.variable=='2'),'value']) # 48.18723326096659

all_cv_split = pd.DataFrame(all_cv_split, columns = ['sample','asm','direction','8','6 (default)','4','2'])       
all_cv_split.to_csv('plots/hmp2_relaxed_aligner_cv_split.csv',index=False)
all_cv_split = pd.read_csv('plots/hmp2_relaxed_aligner_cv_split.csv')
avg_all_cv_split = all_cv_split.drop(columns=['asm']).groupby(['sample','direction'], as_index=False).mean()
avg_all_cv_split = avg_all_cv_split.melt(id_vars=['sample','direction'])

sns.boxplot(data = avg_all_cv_split, x = 'variable', y= 'value', hue= 'direction', order= ['8','6 (default)','4','2'], showfliers=False)
ttest_rel(avg_all_cv_split.loc[(avg_all_cv_split.variable=='6 (default)')&(avg_all_cv_split.direction=='upstream'),'value'],
          avg_all_cv_split.loc[(avg_all_cv_split.variable=='4')&(avg_all_cv_split.direction=='upstream'),'value']) 
ttest_rel(avg_all_cv_split.loc[(avg_all_cv_split.variable=='6 (default)')&(avg_all_cv_split.direction=='downstream'),'value'],
          avg_all_cv_split.loc[(avg_all_cv_split.variable=='4')&(avg_all_cv_split.direction=='downstream'),'value']) 

### Check number of mismatches ###
mapq_df = []
for sample in hmp2_mpa.columns:
    for species in hmp2_mpa.index:
        if np.isnan(hmp2_mpa.loc[species,sample]):
            continue
        asm = hmp2_genomes[species]

        gene_annot = pd.read_csv(f'refseq_genomes/{asm}/{asm}_filtered_tss.csv')
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
mapq_df = pd.DataFrame(mapq_df, columns = ['sample','asm','mp', 'unique_count', 'NM_avg'])     
mapq_df.to_csv('plots/hmp2_relaxed_aligner_mapq.csv',index=False)

mapq_df = pd.read_csv('plots/hmp2_relaxed_aligner_mapq.csv')
avg_mapq_df = mapq_df.drop(columns=['asm']).groupby(['sample','mp'], as_index=False).mean()

sns.boxplot(data = avg_mapq_df, x = 'mp', y= 'unique_count', order= ['8', '6 (default)','4','2'], showfliers=False)
add_p_val(plt,0,1,1150000,20000,ttest_rel(avg_mapq_df.loc[(avg_mapq_df.mp=='8'),'unique_count'],avg_mapq_df.loc[(avg_mapq_df.mp=='6 (default)'),'unique_count']).pvalue)
add_p_val(plt,1,2,1250000,20000,ttest_rel(avg_mapq_df.loc[(avg_mapq_df.mp=='6 (default)'),'unique_count'],avg_mapq_df.loc[(avg_mapq_df.mp=='4'),'unique_count']).pvalue)
add_p_val(plt,2,3,1350000,20000,ttest_rel(avg_mapq_df.loc[(avg_mapq_df.mp=='4'),'unique_count'],avg_mapq_df.loc[(avg_mapq_df.mp=='2'),'unique_count']).pvalue)
plt.xlabel('Mismatch penalty')
plt.ylabel('Total unique reads')
plt.tight_layout()
plt.savefig(f"plots/relaxed_aligner/hmp2_unique_reads_mp_avg.pdf")
plt.clf()

sns.boxplot(data = avg_mapq_df, x = 'mp', y= 'NM_avg', order= ['8', '6 (default)','4','2'], showfliers=False)
add_p_val(plt,0,1,3.5,0.1,ttest_rel(avg_mapq_df.loc[(avg_mapq_df.mp=='8'),'NM_avg'],avg_mapq_df.loc[(avg_mapq_df.mp=='6 (default)'),'NM_avg']).pvalue)
add_p_val(plt,1,2,4,0.1,ttest_rel(avg_mapq_df.loc[(avg_mapq_df.mp=='6 (default)'),'NM_avg'],avg_mapq_df.loc[(avg_mapq_df.mp=='4'),'NM_avg']).pvalue)
add_p_val(plt,2,3,4.5,0.1,ttest_rel(avg_mapq_df.loc[(avg_mapq_df.mp=='4'),'NM_avg'],avg_mapq_df.loc[(avg_mapq_df.mp=='2'),'NM_avg']).pvalue)
plt.xlabel('Mismatch penalty')
plt.ylabel('Average substition count')
plt.tight_layout()
plt.savefig("plots/relaxed_aligner/hmp2_NM_score_avg.pdf")
plt.clf()

# Higher conservation of sequence in highly expressed genes.
# Check read NM score of reads downstream of TSS across expression group
all_metrics = pd.DataFrame()
for sample in hmp2_mpa.columns:
    for species in hmp2_mpa.index:
        if np.isnan(hmp2_mpa.loc[species,sample]):
            continue
        asm = hmp2_genomes[species]
        
        #Read TPM sample data
        tpm_file = get_tpm(f'kallisto_res/{asm}_{sample}/abundance.tsv')
        chrom_map = pd.read_table(f'refseq_genomes/{asm}/{asm}_filtered_tss.csv', sep = ',', header=0, index_col=0)
        chrom_map = chrom_map[~chrom_map.index.duplicated(keep='first')]
        chrom_map = {chrom:re.match(r'ID=(\d*)_',chrom_map.loc[chrom,'Desc']).group(1) for chrom in chrom_map.index}
        tpm_file.index = [x.replace(x.rsplit('_',1)[0],chrom_map.get(x.rsplit('_',1)[0],'na')) for x in tpm_file.index]
        tpm_dict = tpm_file.to_dict()

        #Get gene No. mismatch
        gene_annot = pd.read_csv(f'refseq_genomes/{asm}/{asm}_filtered_tss.csv')
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
all_metrics = all_metrics.dropna()
all_metrics.to_csv('plots/hmp2_relaxed_aligner_NMvsTPM.csv')

all_metrics = pd.read_csv('plots/hmp2_relaxed_aligner_NMvsTPM.csv', index_col=0)
avg_all_metrics = all_metrics.drop(columns=['asm']).groupby(['sample','tpm_group'], as_index=False).mean()

sns.boxplot(data=avg_all_metrics, y="downstream_NM", x="tpm_group", order = ['zero','low','mid','high'], showfliers=False)
add_p_val(plt,0,1,6,0.2,ttest_ind(avg_all_metrics.loc[(avg_all_metrics.tpm_group=='zero'),'downstream_NM'],avg_all_metrics.loc[(avg_all_metrics.tpm_group=='low'),'downstream_NM']).pvalue)
add_p_val(plt,1,2,6,0.2,ttest_ind(avg_all_metrics.loc[(avg_all_metrics.tpm_group=='low'),'downstream_NM'],avg_all_metrics.loc[(avg_all_metrics.tpm_group=='mid'),'downstream_NM']).pvalue)
add_p_val(plt,2,3,6,0.2,ttest_ind(avg_all_metrics.loc[(avg_all_metrics.tpm_group=='mid'),'downstream_NM'],avg_all_metrics.loc[(avg_all_metrics.tpm_group=='high'),'downstream_NM']).pvalue)
add_p_val(plt,0,2,7,0.2,ttest_ind(avg_all_metrics.loc[(avg_all_metrics.tpm_group=='zero'),'downstream_NM'],avg_all_metrics.loc[(avg_all_metrics.tpm_group=='mid'),'downstream_NM']).pvalue)
add_p_val(plt,0,3,8,0.2,ttest_ind(avg_all_metrics.loc[(avg_all_metrics.tpm_group=='zero'),'downstream_NM'],avg_all_metrics.loc[(avg_all_metrics.tpm_group=='high'),'downstream_NM']).pvalue)
plt.xlabel('Expression group')
plt.ylabel('Average downstream substition count')
plt.tight_layout()
plt.savefig("plots/relaxed_aligner/hmp2_exp_vs_downstream_NM_avg.pdf")
plt.clf()

avg_all_metrics[avg_all_metrics["tpm_group"]=='zero']['downstream_NM'].mean() # 2.349374099184168
avg_all_metrics[avg_all_metrics["tpm_group"]=='low']['downstream_NM'].mean() # 2.068957466754067
avg_all_metrics[avg_all_metrics["tpm_group"]=='mid']['downstream_NM'].mean() # 2.176172467360027
avg_all_metrics[avg_all_metrics["tpm_group"]=='high']['downstream_NM'].mean() # 2.3203900513937095

sns.boxplot(data=avg_all_metrics, y="upstream_NM", x="tpm_group", order = ['zero','low','mid','high'], showfliers=False)
add_p_val(plt,0,1,5.5,0.2,ttest_ind(avg_all_metrics.loc[(avg_all_metrics.tpm_group=='zero'),'upstream_NM'],avg_all_metrics.loc[(avg_all_metrics.tpm_group=='low'),'upstream_NM']).pvalue)
add_p_val(plt,1,2,5.5,0.2,ttest_ind(avg_all_metrics.loc[(avg_all_metrics.tpm_group=='low'),'upstream_NM'],avg_all_metrics.loc[(avg_all_metrics.tpm_group=='mid'),'upstream_NM']).pvalue)
add_p_val(plt,2,3,5.5,0.2,ttest_ind(avg_all_metrics.loc[(avg_all_metrics.tpm_group=='mid'),'upstream_NM'],avg_all_metrics.loc[(avg_all_metrics.tpm_group=='high'),'upstream_NM']).pvalue)
add_p_val(plt,0,2,6.5,0.2,ttest_ind(avg_all_metrics.loc[(avg_all_metrics.tpm_group=='zero'),'upstream_NM'],avg_all_metrics.loc[(avg_all_metrics.tpm_group=='mid'),'upstream_NM']).pvalue)
add_p_val(plt,0,3,7.5,0.2,ttest_ind(avg_all_metrics.loc[(avg_all_metrics.tpm_group=='zero'),'upstream_NM'],avg_all_metrics.loc[(avg_all_metrics.tpm_group=='high'),'upstream_NM']).pvalue)
plt.xlabel('Expression group')
plt.ylabel('Average upstream substition count')
plt.tight_layout()
plt.savefig("plots/relaxed_aligner/hmp2_exp_vs_upstream_NM_avg.pdf")
plt.clf()

avg_all_metrics[avg_all_metrics["tpm_group"]=='zero']['upstream_NM'].mean() # 2.3803567756238095
avg_all_metrics[avg_all_metrics["tpm_group"]=='low']['upstream_NM'].mean() # 2.0949764255384054
avg_all_metrics[avg_all_metrics["tpm_group"]=='mid']['upstream_NM'].mean() # 2.19785775793678
avg_all_metrics[avg_all_metrics["tpm_group"]=='high']['upstream_NM'].mean() # 2.29087021116697
