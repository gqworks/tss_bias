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

def taxid2asm(taxid,n=1):
    strain = refseq[(refseq['species_taxid']==int(taxid)) & (refseq['assembly_level']=='Complete Genome')]
    if len(strain) == 0:
        strain = refseq[(refseq['species_taxid']==int(taxid))]
    ftp = strain['ftp_path'][:n].tolist()
    asm = [x.rsplit('/',1)[1] for x in ftp]
    return(asm)

def asm_2_species(asm):
    name = refseq[refseq['ftp_path'].str.contains(asm)]['organism_name'].item()
    name = re.match(r'(^[^ ]+ [^ ]+).*',name).group(1)
    if name == '[Eubacterium] eligens':
        name = 'Lachnospira eligens'
    return name

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
    return (d-d.mean(axis=1)[:,None]) / d.std(axis=1)[:,None]

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
    fna = f'strains/{asm}/{asm}.fna'
    fasta_sequences = SeqIO.to_dict(SeqIO.parse(fna, "fasta"))
    tss_dat = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
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
with open('Franzosa_8_samples') as f:
    Franzosa_8 = f.read().splitlines()
Franzosa_8 = {x.split('\t')[0]:x.split('\t')[1] for x in Franzosa_8}

#Read species abundances
mpa_res = pd.read_table('metaphlan3_res/merged_mpa.txt', comment= '#')
mpa_res = mpa_res[mpa_res['clade_name'].str.contains("s__")]
mpa_res.index = [int(re.match('.*\|([^\|]*)',x).group(1)) for x in mpa_res['NCBI_tax_id']]
mpa_res = mpa_res.drop(columns=['clade_name', 'NCBI_tax_id'])
Franzosa_mpa = mpa_res.loc[:,Franzosa_8.keys()]
Franzosa_mpa = Franzosa_mpa.loc[((Franzosa_mpa > 1).sum(axis=1) >= 5),:] # Filter species with atleast 1% abundance in atleast 5 samples
Franzosa_mpa[Franzosa_mpa < 1] = np.nan
Franzosa_mpa = Franzosa_mpa.drop(39491) #Eubacterium Rectale no reference in database.

refseq = pd.read_table('src/assembly_summary_refseq.txt', header = 1)

all_strains = []
for taxid in Franzosa_mpa.index:
    asms = taxid2asm(taxid,3)
    all_strains = all_strains + asms
dataset_asm_2_species = {x:asm_2_species(x) for x in all_strains}

'''
##########
### GC ###
##########
### GC content Null distribution ###
all_pvals = []
for asm in all_strains:
    fna = f'strains/{asm}/{asm}.fna'
    fasta_sequences = SeqIO.to_dict(SeqIO.parse(fna, "fasta"))
    tss_dat = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
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
all_pvals.to_csv('plots/null_distributions/all_pvals_gc.csv')

##################
### SIMULATED ####
##################
###Simulated Reads null distribution
for asm in all_strains:
    fna_file = f'strains/{asm}/{asm}.fna'
    if not os.path.exists(f'strains/{asm}/pe_simu1.fq'):
        subprocess.call(f'ml ART && art_illumina -na -ss HS25 -i {fna_file} -p -l 150 --fcov 30 -m 200 -s 10 -o strains/{asm}/pe_simu', shell= True)
    if not os.path.exists(f'bams/{asm}_simu.sorted.bam'):
        subprocess.call(f'bowtie2 -x {fna_file} -1 strains/{asm}/pe_simu1.fq -2 strains/{asm}/pe_simu2.fq |samtools view --threads 64 -b - |samtools sort - -o bams/{asm}_simu.sorted.bam --threads 64', shell=True)
        subprocess.call(f'samtools index bams/{asm}_simu.sorted.bam -@ 64',shell=True)
    if not os.path.exists(f'beds/{asm}_simu.bw'):
        subprocess.call(f'bamCoverage --bam bams/{asm}_simu.sorted.bam -p 64 -o beds/{asm}_simu.bw -of bigwig', shell = True)

all_pvals = []
for asm in all_strains:
    scaled_mat = get_simu_bias_mat(asm)
    simu_bias = list(map(calc_bias, np.array(scaled_mat)))
    simu_bw = pyBigWig.open(f'beds/{asm}_simu.bw')
    tss_dat = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
    fna = f'strains/{asm}/{asm}.fna'
    fasta_sequences = SeqIO.to_dict(SeqIO.parse(fna, "fasta"))
    null_bias = []
    for i in tqdm(range(500)):
        null_mat = get_simu_null(simu_bw,tss_dat,fasta_sequences)
        scaled_null = scale_func2(null_mat)
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
all_pvals.to_csv('plots/null_distributions/all_pvals_simu.csv')


all_pvals = pd.read_csv('plots/null_distributions/all_pvals_simu.csv', index_col=0)
all_pvals['species'] = all_pvals['0'].map(dataset_asm_2_species)
all_pvals = all_pvals.groupby(['species']).mean().reset_index()
g = sns.violinplot(data = all_pvals, x = '1', color = 'tab:blue', width = 0.1)
g.figure.savefig(f"plots/null_distributions/all_species_simu_pval_violinplot.pdf", bbox_inches="tight")
plt.clf()


#Simu heatmap example
asm = 'GCF_003312465.1_ASM331246v1'
mat = get_simu_bias_mat(asm, scale = False)
scaled_mat = scale_mat(mat)
scaled_mat = scaled_mat.fillna(0)
sorted_mat = mat.loc[sort_tss_bias(scaled_mat),:]
vmax = np.percentile(sorted_mat.to_numpy().flatten(),95)
plot_heatmap(sorted_mat, f'plots/coverage_heatmaps/{asm}_simu.png',vmax=vmax)

simu_bw = pyBigWig.open(f'beds/{asm}_simu.bw')
tss_dat = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
fasta_sequences = SeqIO.to_dict(SeqIO.parse(f'strains/{asm}/{asm}.fna', "fasta"))
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
for sample in Franzosa_mpa.columns:
    for taxid in Franzosa_mpa.index:
        if np.isnan(Franzosa_mpa.loc[taxid,sample]):
            continue
        strains = taxid2asm(taxid,3)
        for asm in strains:
            #Calculate mean sample TSS bias
            mat = pd.read_table(f'matrices/{asm}_{sample}.csv', sep = ',', header=None, index_col=0)
            scaled_mat = scale_mat(mat)
            scaled_mat = scaled_mat.fillna(0)
            real_bias = list(map(calc_bias, np.array(scaled_mat)))
            
            #Calculate null TSS bias distribution
            null_bias = []
            tss_dat = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
            bw_depth = pyBigWig.open(f'beds/{asm}_{sample}.bw')
            for i in range(500):
                null_mat = get_fixed_null(bw_depth,tss_dat)
                scaled_null = scale_mat(null_mat)
                scaled_null = scaled_null.fillna(0)
                null_bias.append(np.mean(list(map(calc_bias, np.array(scaled_null)))))
            pval = (sum(null_bias > np.mean(real_bias))+1)/len(null_bias)
            #plt.suptitle(f'{sample} {asm}')
            #plt.hist(null_bias, bins=50)
            #plt.axvline(x=np.mean(real_bias), color='r')
            #plt.text(-5, 3, f'p={pval}', size=12, color='white',weight='bold')
            #plt.savefig(f"plots/null_distributions/{asm}_{sample}.pdf")
            #plt.clf()
            
            #Correlate scores with bias from simulated reads
            simu_scaled_mat = get_simu_bias_mat(asm)
            simu_bias = list(map(calc_bias, np.array(simu_scaled_mat)))
            simu_corr = pearsonr(simu_bias,real_bias)
            all_pvals.append([asm,sample,pval,simu_corr])
        
all_pvals = pd.DataFrame(all_pvals)
all_pvals.to_csv('plots/null_distributions/all_pvals_sample.csv')

all_pvals = pd.read_csv('plots/null_distributions/all_pvals_sample.csv', index_col=0, header = 0, names=['asm','sample','pval','pearson'])
all_pvals['corr'] = [float(re.match(r'\(([^,]*),.*',x).group(1)) for x in all_pvals['pearson']]
all_pvals['species'] = all_pvals['asm'].map(dataset_asm_2_species)
all_pvals = all_pvals.groupby(['sample','species']).mean().reset_index()
all_pvals['adj_pval'] = multi.multipletests(all_pvals.pval, method = 'fdr_bh', alpha = 0.05)[1]
all_pvals['neg_log'] = -np.log(all_pvals['adj_pval'])

g = sns.boxplot(data = all_pvals, y = 'species', x = 'neg_log', orient = 'h', color = 'tab:blue', showfliers=False)
plt.xlabel("-log(p-value)")
plt.axvline(-np.log(0.05), color='r')
g.figure.savefig(f"plots/null_distributions/all_samp_species_pval_boxplot.pdf", bbox_inches="tight")
plt.clf()

g = sns.boxplot(data = all_pvals, y = 'species', x = 'corr', orient = 'h', color = 'tab:blue')
g.figure.savefig(f"plots/null_distributions/all_samp_species_simu_corr_boxplot.pdf", bbox_inches="tight")
plt.clf()

###Plot heatmaps #Change to just 1 example
sample = 'X311245214_RNAlater'
asm = 'GCF_002586945.1_ASM258694v1'
#Sample Heatmap
mat = pd.read_table(f'matrices/{asm}_{sample}.csv', sep = ',', header=None, index_col=0)
scaled_mat = scale_mat(mat)
scaled_mat = scaled_mat.fillna(0)
sorted_mat = mat.loc[sort_tss_bias(scaled_mat),:]
vmax = np.percentile(sorted_mat.to_numpy().flatten(),95)
plot_heatmap(sorted_mat, f'plots/coverage_heatmaps/{asm}_{sample}.png', vmax = vmax, dpi = 300)

#Null heatmap
bw_depth = pyBigWig.open(f'beds/{asm}_{sample}.bw')
tss_dat = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
null_mat = get_fixed_null(bw_depth,tss_dat)
scaled_null = scale_mat(null_mat)
scaled_null = scaled_null.fillna(0)
sorted_null = null_mat.loc[sort_tss_bias(scaled_null),:]
plot_heatmap(sorted_null, f'plots/coverage_heatmaps/{asm}_{sample}_null.png',vmax=vmax, dpi = 300)
'''
#########################################
####### Validation in Long reads ########
#########################################
with open('pacbio_idlist') as f:
    pacbio_samples = f.read().splitlines()
pacbio_samples = {x.split('\t')[0]:x.split('\t')[1] for x in pacbio_samples}

with open('nanopore_idlist') as f:
    nanopore_samples = f.read().splitlines()
nanopore_samples = {x.split('\t')[0]:x.split('\t')[1] for x in nanopore_samples}
merged_samples = {**Franzosa_8, **pacbio_samples, **nanopore_samples}

pacbio_abund = pd.read_csv('src/pacbio_merged_k2_raw.csv',header = 0, index_col = 0)
pacbio_abund = 100*pacbio_abund/pacbio_abund.sum(axis=0)
pacbio_abund.index = pacbio_abund.index.values
nanopore_abund = pd.read_csv('src/nanopore_merged_k2_raw.csv',header = 0, index_col = 0)
nanopore_abund = 100*nanopore_abund/nanopore_abund.sum(axis=0)

merged_abund = pd.concat([mpa_res[Franzosa_8.keys()],pacbio_abund,nanopore_abund], axis=1)
merged_abund = merged_abund.loc[((merged_abund[Franzosa_8.keys()] > 0.5).sum(axis=1) >= 4) & ((merged_abund[pacbio_samples.keys()] > 0.5).sum(axis=1) >= 4) & ((merged_abund[nanopore_samples.keys()] > 0.5).sum(axis=1) >= 4),:] 
merged_abund[merged_abund < 0.5] = np.nan
merged_abund = merged_abund.drop(39491) #Eubacterium Rectale no reference in database.


all_LR_pvals = []
with open('plots/null_distributions/all_longread_pvals.txt', 'a') as f:
    for sample in merged_abund.columns:
        for taxid in merged_abund.index:
            if np.isnan(merged_abund.loc[taxid,sample]):
                continue
            strains = taxid2asm(taxid,3)
            for asm in strains:
                #Import TSS positions
                tss_dat = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
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
                for i in range(500):
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
                for i in range(500):
                    null_mat = get_fixed_null(bw_depth,tss_dat_long)
                    scaled_null = scale_mat(null_mat)
                    scaled_null = scaled_null.fillna(0)
                    null_bias.append(np.mean(list(map(calc_bias, np.array(scaled_null)))))
                pval_long = (sum(null_bias > np.mean(real_bias))+1)/len(null_bias)

                all_LR_pvals.append([asm,sample,pval,pval_long])
                f.write(f'{asm},{sample},{pval},{pval_long}\n')
            
all_LR_pvals = pd.DataFrame(all_LR_pvals)
all_LR_pvals.to_csv('plots/null_distributions/all_longread_pvals.csv')


all_LR_pvals = pd.read_csv('plots/null_distributions/all_longread_pvals.csv', index_col = 0, header= 0, names = ['Strain','Sample','pval_all','pval_long'])
all_LR_pvals['adj_pval_all'] = multi.multipletests(all_LR_pvals.pval_all, method = 'fdr_bh', alpha = 0.05)[1]
all_LR_pvals['adj_pval_long'] = multi.multipletests(all_LR_pvals.pval_long, method = 'fdr_bh', alpha = 0.05)[1]

all_LR_pvals = pd.melt(all_LR_pvals, id_vars = ['Strain','Sample'], value_vars = ['adj_pval_all','adj_pval_long'], var_name='type', value_name='pval')

all_strains = list(set(all_LR_pvals['Strain']))
platform_asm_2_species = {x:asm_2_species(x) for x in all_strains}
all_LR_pvals['species'] = all_LR_pvals['Strain'].map(platform_asm_2_species)
all_LR_pvals = all_LR_pvals.groupby(['species','Sample','type']).mean().reset_index()
sum(all_LR_pvals[all_LR_pvals.type=='adj_pval_all']['pval'] < 0.05) # 27
sum(all_LR_pvals[all_LR_pvals.type=='adj_pval_long']['pval'] < 0.05) # 73
all_LR_pvals['pval_log'] = -np.log(all_LR_pvals['pval'])

g = sns.boxplot(data = all_LR_pvals, y = 'species', x = 'pval_log', orient = 'h', hue = 'type', showfliers=False)
plt.axvline(-np.log(0.05), color='r')
g.figure.savefig(f"plots/null_distributions/longread_pval_boxplot.pdf", bbox_inches="tight")
plt.clf()
