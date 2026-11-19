import pandas as pd
import re, os
import numpy as np
from Bio import SeqIO
import matplotlib.pyplot as plt
import subprocess
import pyBigWig
from tqdm import tqdm
from scipy.stats import pearsonr
import seaborn as sns
import statsmodels.stats.multitest as multi

def get_gc_null(tss_dat,fasta_sequences):
    null_gc = []
    for chrom in list(set(tss_dat.Genome)):
        random_tss = randomize(np.array(tss_dat[tss_dat.Genome==chrom]['Start']))
        strand_perc = sum((tss_dat.Genome==chrom)&(tss_dat.Strand=='+'))/sum(tss_dat.Genome==chrom)
        genome_len = len(fasta_sequences[chrom].seq)
        for i in random_tss:
            if i <= 501 or i >= genome_len-500:
                continue
            left_perc = 100*len(re.findall("G|C",str(fasta_sequences[chrom].seq[i-500:i])))/500
            right_perc = 100*len(re.findall("G|C",str(fasta_sequences[chrom].seq[i:i+500])))/500
            if np.random.binomial(1,strand_perc)==1: 
                null_gc.append(right_perc-left_perc)
            else:
                null_gc.append(left_perc-right_perc)
    return(np.mean(null_gc))

def randomize(intervals):
    intervals = intervals.astype(int)
    point_diffs = intervals[1:] - intervals[:-1]
    max_len = intervals[-1]
    new_points = np.concatenate((np.array(point_diffs).cumsum(),[max_len]))
    new_points = (new_points + np.random.randint(0,max_len))%max_len
    return new_points

def get_simu_null(simu_bw,tss_dat,fasta_sequences):
    null_simu_mat = []
    for chrom in list(set(tss_dat.Genome)):
        random_tss = randomize(np.array(tss_dat[tss_dat.Genome==chrom]['Start']))
        strand_perc = sum((tss_dat.Genome==chrom)&(tss_dat.Strand=='+'))/sum(tss_dat.Genome==chrom)
        genome_len = len(fasta_sequences[chrom].seq)
        for i in random_tss:
            if i <= 501 or i >= genome_len-500:
                continue
            window = simu_bw.values(chrom,i-501,i+500)
            if np.random.binomial(1, strand_perc) == 0: window.reverse()
            null_simu_mat.append(window)
    return(pd.DataFrame(null_simu_mat))

def scale_mat(d):
    mean_vals = d.mean(axis=1).to_numpy()
    std_vals = d.std(axis=1).to_numpy()
    return (d - mean_vals[:, None]) / std_vals[:, None]

def calc_bias(x):
    left_sum = np.sum(x[:500])
    right_sum = np.sum(x[500:])
    return(right_sum-left_sum)

def get_fixed_null(depth,annot):
    genome_dict = depth.chroms()
    null_mat = []
    for chrom in list(set(annot.Genome)):
        random_tss = randomize(np.array(annot[annot.Genome==chrom]['Start']))
        strand_perc = sum((annot.Genome==chrom)&(annot.Strand=='+'))/sum(annot.Genome==chrom)
        for i in random_tss:
            if i <= 500 or i >= genome_dict[chrom]-500:
                continue
            window = depth.values(chrom,i-501,i+500)
            if np.random.binomial(1, strand_perc) == 0: window.reverse()
            null_mat.append(window)
    return(pd.DataFrame(null_mat))

def get_simu_bias_mat(asm, scale = True):
    fna = f'refseq_genomes/{asm}/{asm}.fna'
    fasta_sequences = SeqIO.to_dict(SeqIO.parse(fna, "fasta"))
    tss_dat = pd.read_csv(f'refseq_genomes/{asm}/{asm}_filtered_tss.csv')
    tss_dat_np = tss_dat.to_numpy()
    simu_bw = pyBigWig.open(f'beds/{asm}_simu.bw')
    simu_mat = []
    for i in tss_dat_np:
        genome_len = len(fasta_sequences[i[0]].seq)
        start = i[5]
        strand = i[3]
        if start <= 501 or start >= genome_len-500:
            window = [0]*1001
        else:
            window = simu_bw.values(i[0],start-501,start+500)
        if i[3]=="-": window.reverse()
        simu_mat.append(window)
    simu_mat = pd.DataFrame(simu_mat)
    simu_mat.index = [re.match(r'ID=([^;]*);.*',x[4]).group(1) for x in tss_dat_np]
    if scale:
        scaled_mat = scale_mat(simu_mat)
        scaled_mat = scaled_mat.fillna(0)
        return(scaled_mat)
    else:
        return(simu_mat)

def plot_heatmap(mat,title,vmax,dpi=300):
    g = sns.clustermap(mat, col_cluster = False, row_cluster = False, xticklabels = False, yticklabels=False, figsize=(10, 25), vmax=vmax, cmap = 'Oranges')
    g.savefig(title, dpi = dpi)

def sort_tss_bias(x):
    #input pandas matrix
    x['bias'] = [calc_bias(i) for i in np.array(x)]
    index = x.sort_values(['bias'], ascending = False).index
    return(index)


###Initialise Data
with open('sample_lists/franzosa_RNALater_samples') as f:
    franzosa_samples = f.read().splitlines()
franzosa_samples = {x.split('\t')[0]:x.split('\t')[1] for x in franzosa_samples}

with open('sample_lists/hmp2_samples') as f:
    hmp2_samples = f.read().splitlines()
hmp2_samples = {x.split('\t')[0]:x.split('\t')[1] for x in hmp2_samples}

#Species abundances
mpa_res = pd.read_table('mpa4_res/merged_mpa.txt', comment= '#')
mpa_res = mpa_res[(mpa_res['clade_name'].str.contains("s__"))&(~mpa_res['clade_name'].str.contains("t__"))]
mpa_res.index = [re.match('.*\|([^\|]*)',x).group(1) for x in mpa_res['clade_name']]
mpa_res = mpa_res.drop(columns=['clade_name'])

franzosa_mpa = mpa_res.loc[:,franzosa_samples.keys()]
franzosa_mpa = franzosa_mpa.loc[((franzosa_mpa > 0.5).sum(axis=1) >= 5),:] # Filter species with atleast 0.5% abundance in atleast 5 samples
franzosa_mpa = franzosa_mpa.where(franzosa_mpa > 0.5, np.nan)

franzosa_genomes = {'s__Phocaeicola_vulgatus': 'GCF_000012825.1_ASM1282v1',
 's__Agathobacter_rectalis': 'GCF_022453685.1_ASM2245368v1',
 's__Alistipes_putredinis': 'GCF_000154465.1_ASM15446v1',
 's__Faecalibacterium_prausnitzii': 'GCF_000154385.1_ASM15438v1',
 's__Bacteroides_uniformis': 'GCF_044361425.1_ASM4436142v1',
 's__Phascolarctobacterium_faecium': 'GCF_003945365.1_PFJ30894_01',
 's__Parabacteroides_distasonis': 'GCF_018279895.1_ASM1827989v1',
 's__Gemmiger_formicilis': 'GCF_900167555.1_IMG-taxon_2590828806_annotated_assembly',
 's__Fusicatenibacter_saccharivorans': 'GCF_027667425.1_ASM2766742v1'}

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
'''
##########
### GC ###
##########
### GC content Null distribution ###
all_pvals = []
for species in hmp2_genomes.keys():
    asm = hmp2_genomes[species]
    fna = f'refseq_genomes/{asm}/{asm}.fna'
    fasta_sequences = SeqIO.to_dict(SeqIO.parse(fna, "fasta"))
    tss_dat = pd.read_csv(f'refseq_genomes/{asm}/{asm}_filtered_tss.csv')
    tss_dat_np = tss_dat.to_numpy()
    gc_bias = []
    for i in tss_dat_np:
        genome_len = len(fasta_sequences[i[0]].seq)
        start = i[5]
        strand = i[3]
        if start <= 501 or start >= genome_len-500:
            continue
        left_perc = 100*len(re.findall("G|C",str(fasta_sequences[i[0]].seq[start-500:start])))/500
        right_perc = 100*len(re.findall("G|C",str(fasta_sequences[i[0]].seq[start:start+500])))/500
        if strand == '+': 
            gc_bias.append(right_perc-left_perc)
        else:
            gc_bias.append(left_perc-right_perc)
    real_bias = np.mean(gc_bias)
    null_bias = []
    for i in range(500):
        null_bias.append(get_gc_null(tss_dat,fasta_sequences))
    pval = (sum(null_bias > real_bias)+1)/len(null_bias)
    all_pvals.append([asm,pval])
    plt.suptitle('GC null distribution')
    plt.hist(null_bias, bins=50)
    plt.axvline(x=real_bias, color='r')
    plt.text(1, 10, f'p={pval}', size=12, color='black',weight='bold')
    plt.savefig(f"plots/null_distributions/{asm}_gc.pdf")
    plt.clf()
all_pvals = pd.DataFrame(all_pvals)
all_pvals.to_csv('plots/null_distributions/hmp2_genome_pvals_gc.csv')

##################
### SIMULATED ####
##################
###Simulated Reads null distribution
for species in hmp2_genomes.keys():
    asm = hmp2_genomes[species]
    fna_file = f'refseq_genomes/{asm}/{asm}.fna'
    if not os.path.exists(f'refseq_genomes/{asm}/pe_simu1.fq'):
        subprocess.call(f'ml ART && art_illumina -na -ss HS25 -i {fna_file} -p -l 150 --fcov 30 -m 200 -s 10 -o refseq_genomes/{asm}/pe_simu', shell= True)
    if not os.path.exists(f'bams/{asm}_simu.sorted.bam'):
        subprocess.call(f'bowtie2 -p 64 -x {fna_file} -1 refseq_genomes/{asm}/pe_simu1.fq -2 refseq_genomes/{asm}/pe_simu2.fq |samtools view --threads 64 -b - |samtools sort - -o bams/{asm}_simu.sorted.bam --threads 64', shell=True)
        subprocess.call(f'samtools index bams/{asm}_simu.sorted.bam -@ 64',shell=True)
    if not os.path.exists(f'beds/{asm}_simu.bw'):
        subprocess.call(f'bamCoverage --bam bams/{asm}_simu.sorted.bam -p 64 -o beds/{asm}_simu.bw -of bigwig', shell = True)

all_pvals = []
for species in hmp2_genomes.keys():
    asm = hmp2_genomes[species]
    scaled_mat = get_simu_bias_mat(asm)
    simu_bias = list(map(calc_bias, np.array(scaled_mat)))
    simu_bw = pyBigWig.open(f'beds/{asm}_simu.bw')
    tss_dat = pd.read_csv(f'refseq_genomes/{asm}/{asm}_filtered_tss.csv')
    fna = f'refseq_genomes/{asm}/{asm}.fna'
    fasta_sequences = SeqIO.to_dict(SeqIO.parse(fna, "fasta"))
    null_bias = []
    for i in tqdm(range(500)):
        null_mat = get_simu_null(simu_bw,tss_dat,fasta_sequences)
        scaled_null = scale_mat(null_mat)
        scaled_null = scaled_null.fillna(0)
        null_bias.append(np.mean(list(map(calc_bias, np.array(scaled_null)))))
    pval = (sum(null_bias > np.mean(simu_bias))+1)/len(null_bias)
    all_pvals.append([asm,pval])
    plt.suptitle('Simulated reads null distribution')
    plt.hist(null_bias, bins=50)
    plt.axvline(x=np.mean(simu_bias), color='r')
    plt.text(-7, 3, f'p={pval}', size=12, color='white',weight='bold')
    plt.savefig(f"plots/null_distributions/{asm}_simu.pdf")
    plt.clf()
all_pvals = pd.DataFrame(all_pvals)
all_pvals.to_csv('plots/null_distributions/hmp2_pvals_simu.csv')


all_pvals = pd.read_csv('plots/null_distributions/hmp2_pvals_simu.csv', index_col=0)
all_pvals['species'] = [list(hmp2_genomes.keys())[list(hmp2_genomes.values()).index(x)] for x in all_pvals['0']]
all_pvals = all_pvals.groupby(['species']).mean().reset_index()
g = sns.violinplot(data = all_pvals, x = '1', color = 'tab:blue', cut = 0, bw=0.2, inner="point", inner_kws=dict(markersize=15))
g.figure.savefig(f"plots/null_distributions/hmp2_species_simu_pval_violinplot.pdf", bbox_inches="tight")
plt.clf()


#Simu heatmap example
asm = 'GCF_000154385.1_ASM15438v1'
mat = get_simu_bias_mat(asm, scale = False)
scaled_mat = scale_mat(mat)
scaled_mat = scaled_mat.fillna(0)
sorted_mat = mat.loc[sort_tss_bias(scaled_mat),:]
vmax = np.percentile(sorted_mat.to_numpy().flatten(),95)
plot_heatmap(sorted_mat, f'plots/coverage_heatmaps/{asm}_simu.png',vmax=vmax)

simu_bw = pyBigWig.open(f'beds/{asm}_simu.bw')
tss_dat = pd.read_csv(f'refseq_genomes/{asm}/{asm}_filtered_tss.csv')
fasta_sequences = SeqIO.to_dict(SeqIO.parse(f'refseq_genomes/{asm}/{asm}.fna', "fasta"))
simu_null_mat = get_simu_null(simu_bw,tss_dat,fasta_sequences)
scaled_mat = scale_mat(simu_null_mat)
scaled_mat = scaled_mat.fillna(0)
sorted_mat = simu_null_mat.loc[sort_tss_bias(scaled_mat),:]
plot_heatmap(sorted_mat, f'plots/coverage_heatmaps/{asm}_simu_null.png',vmax=vmax)

###################################
####### REAL READ COVERAGE ########
###################################
###Fixed intervals null permutations
all_pvals = []
for sample in hmp2_mpa.columns:
    for species in hmp2_mpa.index:
        if np.isnan(hmp2_mpa.loc[species,sample]):
            continue
        asm = hmp2_genomes[species]
        #Calculate mean sample TSS bias
        mat = pd.read_table(f'matrices/{asm}_{sample}.csv', sep = ',', header=None, index_col=0)
        scaled_mat = scale_mat(mat)
        scaled_mat = scaled_mat.fillna(0)
        real_bias = list(map(calc_bias, np.array(scaled_mat)))

        #Calculate null TSS bias distribution
        null_bias = []
        tss_dat = pd.read_csv(f'refseq_genomes/{asm}/{asm}_filtered_tss.csv')
        bw_depth = pyBigWig.open(f'beds/{asm}_{sample}.bw')
        for i in range(500):
            null_mat = get_fixed_null(bw_depth,tss_dat)
            scaled_null = scale_mat(null_mat)
            scaled_null = scaled_null.fillna(0)
            null_bias.append(np.mean(list(map(calc_bias, np.array(scaled_null)))))
        pval = (sum(null_bias > np.mean(real_bias))+1)/len(null_bias)
        plt.suptitle(f'{sample} {asm}')
        plt.hist(null_bias, bins=50)
        plt.axvline(x=np.mean(real_bias), color='r')
        plt.text(-5, 3, f'p={pval}', size=12, color='white',weight='bold')
        plt.savefig(f"plots/null_distributions/{asm}_{sample}.pdf")
        plt.clf()

        #Correlate scores with bias from simulated reads
        simu_scaled_mat = get_simu_bias_mat(asm)
        simu_bias = list(map(calc_bias, np.array(simu_scaled_mat)))
        simu_corr = pearsonr(simu_bias,real_bias)
        all_pvals.append([asm,sample,pval,simu_corr])

all_pvals = pd.DataFrame(all_pvals)
all_pvals.to_csv('plots/null_distributions/hmp2_pvals_sample.csv')

all_pvals = pd.read_csv('plots/null_distributions/hmp2_pvals_sample.csv', index_col=0, header = 0, names=['asm','sample','pval','pearson'])
all_pvals['corr'] = [float(re.match(r'\(([^,]*),.*',x).group(1)) for x in all_pvals['pearson']]
all_pvals['species'] = [list(hmp2_genomes.keys())[list(hmp2_genomes.values()).index(x)] for x in all_pvals['asm']]
all_pvals = all_pvals.groupby(['sample','species']).mean().reset_index()
all_pvals['adj_pval'] = multi.multipletests(all_pvals.pval, method = 'fdr_bh', alpha = 0.05)[1]
all_pvals['neg_log'] = -all_pvals['adj_pval']

g = sns.boxplot(data = all_pvals, y = 'species', x = 'adj_pval', orient = 'h', color = 'tab:blue', showfliers=False)
plt.xlabel("p-value (log-scaled)")
plt.xscale('log')
plt.axvline(0.05, color='r')
g.figure.savefig(f"plots/null_distributions/hmp2_species_pval_boxplot.pdf", bbox_inches="tight")
plt.clf()

g = sns.boxplot(data = all_pvals, y = 'species', x = 'corr', orient = 'h', color = 'tab:blue')
g.figure.savefig(f"plots/null_distributions/hmp2_species_simu_corr_boxplot.pdf", bbox_inches="tight")
plt.clf()

###Plot heatmaps #Change to just 1 example
sample = 'SRR5947807'
asm = 'GCF_000154385.1_ASM15438v1'
mat = pd.read_table(f'matrices/{asm}_{sample}.csv', sep = ',', header=None, index_col=0)
scaled_mat = scale_mat(mat)
scaled_mat = scaled_mat.fillna(0)
sorted_mat = mat.loc[sort_tss_bias(scaled_mat),:]
sorted_mat = sorted_mat[sorted_mat.sum(axis=1) > 0]
vmax = np.percentile(sorted_mat.to_numpy().flatten(),95)
plot_heatmap(sorted_mat, f'plots/coverage_heatmaps/{asm}_{sample}.png', vmax = vmax, dpi = 300)

#Null heatmap
bw_depth = pyBigWig.open(f'beds/{asm}_{sample}.bw')
tss_dat = pd.read_csv(f'refseq_genomes/{asm}/{asm}_filtered_tss.csv')
null_mat = get_fixed_null(bw_depth,tss_dat)
scaled_null = scale_mat(null_mat)
scaled_null = scaled_null.fillna(0)
sorted_null = null_mat.loc[sort_tss_bias(scaled_null),:]
plot_heatmap(sorted_null, f'plots/coverage_heatmaps/{asm}_{sample}_null.png',vmax=vmax, dpi = 300)
'''
#########################################
####### Validation in Long reads ########
#########################################
with open('sample_lists/pacbio_idlist') as f:
    pacbio_samples = f.read().splitlines()
pacbio_samples = {x.split('\t')[0]:x.split('\t')[1] for x in pacbio_samples}

with open('sample_lists/nanopore_idlist') as f:
    nanopore_samples = f.read().splitlines()
nanopore_samples = {x.split('\t')[0]:x.split('\t')[1] for x in nanopore_samples}
merged_samples = {**hmp2_samples, **pacbio_samples, **nanopore_samples}

pacbio_abund = pd.read_csv('src/pacbio_merged_mpa.tsv', sep = '\t', header = 0, index_col = 0, names = sorted(pacbio_samples.keys()))
pacbio_abund = pacbio_abund[(pacbio_abund.index.str.contains("s__"))&(pacbio_abund.index.str.contains("k__"))]
pacbio_abund.index = [re.match('.*\|([^\|]*)',x).group(1) for x in pacbio_abund.index]
pacbio_abund = pacbio_abund[pacbio_abund.mean(axis=1).sort_values(ascending=False)>500]
pacbio_abund = 100*pacbio_abund/pacbio_abund.sum(axis=0)

nanopore_abund = pd.read_csv('src/nanopore_merged_mpa.tsv', sep = '\t', header = 0, index_col = 0, names = sorted(nanopore_samples.keys()))
nanopore_abund = nanopore_abund[(nanopore_abund.index.str.contains("s__"))&(nanopore_abund.index.str.contains("k__"))]
nanopore_abund.index = [re.match('.*\|([^\|]*)',x).group(1) for x in nanopore_abund.index]
nanopore_abund = nanopore_abund[nanopore_abund.mean(axis=1).sort_values(ascending=False)>500]
nanopore_abund = 100*nanopore_abund/nanopore_abund.sum(axis=0)

merged_abund = pd.concat([hmp2_mpa,pacbio_abund,nanopore_abund], axis=1)
# Filter species with atleast 0.5% abundance in atleast 4 samples in each dataset
merged_abund = merged_abund.loc[((merged_abund[hmp2_samples.keys()] > 0.5).sum(axis=1) >= 30) & 
                                ((merged_abund[pacbio_samples.keys()] > 0.5).sum(axis=1) >= 4) & 
                                ((merged_abund[nanopore_samples.keys()] > 0.5).sum(axis=1) >= 4),:]
merged_abund = merged_abund.where(merged_abund > 0.5, np.nan)

selected_genomes = {'s__Phocaeicola_vulgatus': 'GCF_000012825.1_ASM1282v1',
 's__Bacteroides_uniformis': 'GCF_044361425.1_ASM4436142v1',
 's__Agathobacter_rectalis': 'GCF_022453685.1_ASM2245368v1',
 's__Alistipes_putredinis': 'GCF_000154465.1_ASM15446v1',
 's__Faecalibacterium_prausnitzii': 'GCF_000154385.1_ASM15438v1'}
longread_samples = list(pacbio_samples.keys())+list(nanopore_samples.keys())

all_LR_pvals = []
with open('plots/null_distributions/all_longread_pvals_new2.txt', 'a') as f:
    for sample in longread_samples:
        for species in merged_abund.index:
            if np.isnan(merged_abund.loc[species,sample]):
                continue
            asm = selected_genomes[species]

            #Import TSS positions
            tss_dat = pd.read_csv(f'refseq_genomes/{asm}/{asm}_filtered_tss.csv')
            tss_dat.index = np.array([re.match(r'ID=([^;]*)',x).group(1) for x in tss_dat['Desc']])
            tss_dat['len'] = tss_dat['geneRight']-tss_dat['geneLeft']
            tss_dat_long = tss_dat[tss_dat['len'] > 1000] #Filter short genes as long reads will cover past it
            bw_depth = pyBigWig.open(f'beds/{asm}_{sample}.bw')

            #All genes
            mat = pd.read_table(f'matrices/{asm}_{sample}.csv', sep = ',', header=None, index_col=0)
            mat_all = mat.loc[tss_dat.index,]
            scaled_mat = scale_mat(mat_all)
            scaled_mat = scaled_mat.fillna(0)
            real_bias = list(map(calc_bias, np.array(scaled_mat)))

            #sorted_mat = mat_all.loc[sort_tss_bias(scaled_mat),:]
            #sorted_mat = sorted_mat.loc[sorted_mat.sum(axis=1)!=0,:]
            #vmax = np.percentile(sorted_mat.to_numpy().flatten(),95)
            #plot_heatmap(sorted_mat, f'plots/coverage_heatmaps/{asm}_{sample}.png', vmax = vmax)

            null_bias = []
            for _ in range(100):
                null_mat = get_fixed_null(bw_depth,tss_dat)
                scaled_null = scale_mat(null_mat)
                scaled_null = scaled_null.fillna(0)
                null_bias.append(np.mean(list(map(calc_bias, np.array(scaled_null)))))
            pval = (sum(null_bias > np.mean(real_bias))+1)/len(null_bias)

            #Long genes
            mat_long = mat.loc[tss_dat_long.index,]
            scaled_mat = scale_mat(mat_long)
            scaled_mat = scaled_mat.fillna(0)
            real_bias = list(map(calc_bias, np.array(scaled_mat)))

            #sorted_mat = mat_long.loc[sort_tss_bias(scaled_mat),:]
            #sorted_mat = sorted_mat.loc[sorted_mat.sum(axis=1)!=0,:]
            #vmax = np.percentile(sorted_mat.to_numpy().flatten(),95)
            #plot_heatmap(sorted_mat, f'plots/coverage_heatmaps/{asm}_{sample}_long.png', vmax = vmax)

            null_bias = []
            for _ in range(100):
                null_mat = get_fixed_null(bw_depth,tss_dat_long)
                scaled_null = scale_mat(null_mat)
                scaled_null = scaled_null.fillna(0)
                null_bias.append(np.mean(list(map(calc_bias, np.array(scaled_null)))))
            pval_long = (sum(null_bias > np.mean(real_bias))+1)/len(null_bias)

            all_LR_pvals.append([asm,sample,pval,pval_long])
            f.write(f'{asm},{sample},{pval},{pval_long}\n')
            f.flush()
            
all_LR_pvals = pd.DataFrame(all_LR_pvals)
all_LR_pvals.to_csv('plots/null_distributions/all_longread_pvals_new2.csv')


all_LR_pvals = pd.read_csv('plots/null_distributions/all_longread_pvals_new2.csv', index_col = 0, header= 0, names = ['Strain','Sample','pval_all','pval_long'])
all_LR_pvals['adj_pval_all'] = multi.multipletests(all_LR_pvals.pval_all, method = 'fdr_bh', alpha = 0.05)[1]
all_LR_pvals['adj_pval_long'] = multi.multipletests(all_LR_pvals.pval_long, method = 'fdr_bh', alpha = 0.05)[1]
all_LR_pvals = pd.melt(all_LR_pvals, id_vars = ['Strain','Sample'], value_vars = ['adj_pval_all','adj_pval_long'], var_name='type', value_name='pval')
all_LR_pvals['species'] =  [list(selected_genomes.keys())[list(selected_genomes.values()).index(x)] for x in all_LR_pvals['Strain']]
all_LR_pvals = all_LR_pvals.groupby(['species','Sample','type']).mean().reset_index()
sum(all_LR_pvals[all_LR_pvals.type=='adj_pval_all']['pval'] < 0.05) # 221/309
sum(all_LR_pvals[all_LR_pvals.type=='adj_pval_long']['pval'] < 0.05) # 288/309
all_LR_pvals['pval_log'] = -np.log10(all_LR_pvals['pval'])

g = sns.boxplot(data = all_LR_pvals, y = 'species', x = 'pval', orient = 'h', hue = 'type', showfliers=False)
plt.axvline(0.05, color='r')
plt.xlabel("p-value (log-scaled)")
plt.xscale('log')
g.figure.savefig(f"plots/null_distributions/longread_pval_boxplot_new2.pdf", bbox_inches="tight")
plt.clf()