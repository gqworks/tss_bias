import pandas as pd
import re, os
import numpy as np
from Bio import SeqIO
import matplotlib.pyplot as plt
import pyBigWig
import seaborn as sns
import statsmodels.stats.multitest as multi
import subprocess

def get_gc_null(tss_dat_np,fasta_sequences):
    null_gc = []
    for chrom in list(set(tss_dat_np[:,0])):
        subset = tss_dat_np[np.in1d(tss_dat_np[:,0],chrom)]
        random_tss = randomize(subset[:,6])
        strand_perc = sum(subset[:,4]=='+')/len(subset)
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

def scale_mat(d):
    return (d-d.mean(axis=1)[:,None]) / d.std(axis=1)[:,None]

def calc_bias(x):
    left_sum = np.sum(x[:500])
    right_sum = np.sum(x[500:])
    return(right_sum-left_sum)

def get_fixed_null(depth,tss_dat_np):
    genome_dict = depth.chroms()
    null_mat = []
    for chrom in list(set(tss_dat_np[:,0])):
        subset = tss_dat_np[np.in1d(tss_dat_np[:,0],chrom)]
        random_tss = randomize(subset[:,6])
        strand_perc = sum(subset[:,4]=='+')/len(subset)
        for i in random_tss:
            if i <= 500 or i >= genome_dict[chrom]-500:
                continue
            window = depth.values(chrom,i-501,i+500)
            if np.random.binomial(1, strand_perc) == 0: window.reverse()
            null_mat.append(window)
    return(pd.DataFrame(null_mat))

def get_tpm(file):
    dat = pd.read_table(file)
    dat = dat.drop_duplicates(subset='target_id')
    dat = dat[['target_id','tpm']]
    dat.index = dat['target_id']
    dat = dat.drop(columns = ['target_id'])
    return dat

def remove_TSS_overlap(dat):
    #Remove overlaping TSS -TSS doesnt overlap with TSS
    non_overlap = []
    #First case
    if float(dat[0,6])+500 < float(dat[1,6]):
        non_overlap.append(dat[0])
    #Mid case
    for i in range(1,len(dat)-1):
        if float(dat[i,6])-500 > float(dat[i-1,6]) and float(dat[i,6])+500 < float(dat[i+1,6]):
            non_overlap.append(dat[i])
    #Last case
    if float(dat[len(dat)-1,6])-500 > float(dat[len(dat)-2,6]):
        non_overlap.append(dat[len(dat)-1])
    non_overlap = pd.DataFrame(non_overlap, columns = ["Genome","left","right","score",'Strand',"desc","Start"])
    return(non_overlap.to_numpy())

###Initialise Data
'''
mpa_res = pd.read_table('metaphlan3_res/merged_mpa.txt', comment= '#')
mpa_res = mpa_res[mpa_res['clade_name'].str.contains("s__")]
mpa_res.index = [int(re.match('.*\|([^\|]*)',x).group(1)) for x in mpa_res['NCBI_tax_id']]
mpa_res = mpa_res.drop(columns=['clade_name', 'NCBI_tax_id'])

with open('Franzosa_8_samples') as f:
    Franzosa_8 = f.read().splitlines()
Franzosa_8 = {x.split('\t')[0]:x.split('\t')[1] for x in Franzosa_8}
Franzosa_mpa = mpa_res.loc[:,Franzosa_8.keys()]
Franzosa_mpa = Franzosa_mpa.loc[((Franzosa_mpa > 1).sum(axis=1) >= 5),:] # Filter species with atleast 1% abundance in atleast 5 samples
Franzosa_mpa[Franzosa_mpa < 1] = np.nan
Franzosa_mpa = Franzosa_mpa.drop(39491) #Eubacterium Rectale no reference in database.

# Test of F. prausnitzii GCF_902388275.1_UHGG_MGYG-HGUT-02545
asm = 'GCF_902388275.1_UHGG_MGYG-HGUT-02545'

selected_genomes =  ['GCF_002586945.1_ASM258694v1', # 853 Faecalibacterium prausnitzii
                     'GCF_000012825.1_ASM1282v1', # 821 Phocaeicola vulgatus
                     'GCF_000012845.1_ASM1284v1', # 823 Parabacteroides distasonis
                     'GCF_020735745.1_ASM2073574v1', # 39485 Lachnospira eligens
                     'GCF_018289375.1_ASM1828937v1'] # Bacteroides uniformis
'''
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
 #'s__Alistipes_putredinis': 'GCF_000154465.1_ASM15446v1',
 #'s__Faecalibacterium_prausnitzii': 'GCF_000154385.1_ASM15438v1',
 's__Bacteroides_uniformis': 'GCF_044361425.1_ASM4436142v1',
 's__Parabacteroides_distasonis': 'GCF_018279895.1_ASM1827989v1',
 's__Bacteroides_caccae': 'GCF_018292205.1_ASM1829220v1'}

# Run Promotech
all_dat = []
all_differences = []
for species, asm in hmp2_genomes.items():
    if not os.path.exists(f'promotech_res/{asm}'):
        subprocess.call([f'source activate promotech_env && python /home/gordonq/Promotech-master/promotech.py -pg -m RF-HOT -f refseq_genomes/{asm}/{asm}.fna -o promotech_res/{asm}'], shell = True)
    if not os.path.exists(f'promotech_res/{asm}/genome_predictions.csv'):
        subprocess.call([f'source activate promotech_env && cd /home/gordonq/Promotech-master && python /home/gordonq/Promotech-master/promotech.py -g -t 0.7 -m RF-HOT -i /groups/cgsd/gordonq/TSS_depth/main/promotech_res/{asm} -o /groups/cgsd/gordonq/TSS_depth/main/promotech_res/{asm}'], shell = True)

    ### Prepare predicted promoters ###
    tss_dat = pd.read_csv(f'promotech_res/{asm}/genome_predictions.csv', sep = '\t')
    prodigal_tss = pd.read_csv(f'refseq_genomes/{asm}/{asm}_filtered_tss.csv')
    tss_dat = tss_dat.sort_values(by=['score'], ascending = False).head(len(prodigal_tss))
    #tss_dat = tss_dat[tss_dat.score>=0.8]
    tss_dat['tss'] = [x['end'] if x['strand'] == '+' else x['start'] for i, x in tss_dat.iterrows()]
    tss_dat['chrom'] = tss_dat.iloc[0,0].split(',')[0]
    tss_dat_np = tss_dat.to_numpy()
    tss_dat_np = remove_TSS_overlap(tss_dat_np)

    # Compare difference of predicted promoters to prodigal TSSs
    differences = []
    for pos in tss_dat_np[:,6]:
        closest = min(prodigal_tss.Start, key=lambda x:abs(x-pos))
        diff = abs(closest-pos)
        differences.append(diff)
    all_differences = all_differences + differences
    plt.hist(differences, bins=50, range = (0,1000))
    plt.xlabel("Distance to closest TSS (bp)")
    plt.ylabel("No. promoters")
    plt.xticks(np.arange(0, 1000, 100))
    plt.savefig(f"plots/promoter_prediction/{asm}_distance_from_prodigal_positions.pdf")
    plt.clf()
    
    ### GC content Null distribution ###
    fna = f'refseq_genomes/{asm}/{asm}.fna'
    fasta_sequences = SeqIO.to_dict(SeqIO.parse(fna, "fasta"))
    gc_bias = []
    for i in tss_dat_np:
        genome_len = len(fasta_sequences[i[0]].seq)
        start = i[6]
        strand = i[4]
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
    for i in range(1000):
        null_bias.append(get_gc_null(tss_dat_np,fasta_sequences))
    gc_pval = (sum(null_bias > real_bias)+1)/len(null_bias)
    plt.suptitle('GC null distribution')
    plt.hist(null_bias, bins=50)
    plt.axvline(x=real_bias, color='r')
    plt.text(1, 10, f'p={gc_pval}', size=12, color='black',weight='bold')
    plt.savefig(f"plots/promoter_prediction/{asm}_gc_distribution.pdf")
    plt.clf()

    ###################################
    ####### REAL READ COVERAGE ########
    ###################################
    all_pvals = []
    for sample in hmp2_samples.keys():
        if not os.path.exists(f'beds/{asm}_{sample}.bw'):
            continue
        # Calculate matrix
        bp_depth = pyBigWig.open(f'beds/{asm}_{sample}.bw')
        genome_dict = bp_depth.chroms()

        mat = []
        for row in tss_dat_np:
            chrom = row[0]
            if (row[6] <= 550) or (row[6] + 550 >= genome_dict[chrom]):
                window = [0]*1001
            else:
                window = bp_depth.values(chrom,int(row[6])-501,int(row[6])+500)
                if row[4]=="-": window.reverse()
            mat.append(window)
        mat = pd.DataFrame(mat)

        #Calculate mean sample TSS bias
        scaled_mat = scale_mat(mat)
        scaled_mat = scaled_mat.fillna(0)
        real_bias = list(map(calc_bias, np.array(scaled_mat)))

        #Calculate null TSS bias distribution
        null_bias = []
        for i in range(500):
            null_mat = get_fixed_null(bp_depth,tss_dat_np)
            scaled_null = scale_mat(null_mat)
            scaled_null = scaled_null.fillna(0)
            null_bias.append(np.mean(list(map(calc_bias, np.array(scaled_null)))))
        pval = (sum(null_bias > np.mean(real_bias))+1)/len(null_bias)
        plt.suptitle(f'{sample} {asm}')
        plt.hist(null_bias, bins=50)
        plt.axvline(x=np.mean(real_bias), color='r')
        plt.text(-5, 3, f'p={pval}', size=12, color='white',weight='bold')
        plt.savefig(f"plots/promoter_prediction/{asm}_{sample}_null_distribution.pdf")
        plt.clf()

        all_dat.append([asm,sample,pval,gc_pval])
        
all_dat = pd.DataFrame(all_dat, columns = ['asm','sample','pval','gc_pval'])
all_dat['adj_pval'] = multi.multipletests(all_dat['pval'], method = 'fdr_bh', alpha = 0.05)[1]
all_dat['neg_log'] = -np.log10(all_dat['adj_pval'])
all_dat.to_csv("plots/promoter_prediction/all_pvals.csv")

all_dat = pd.read_csv('plots/promoter_prediction/all_pvals.csv', index_col = 0)
all_dat['neg_log'] = -np.log10(all_dat['adj_pval'])
g = sns.boxplot(data = all_dat, y = 'asm', x = 'neg_log', orient = 'h', showfliers=False)
plt.axvline(-np.log10(0.05), color='r')
plt.xlabel("Difference between true bias and null distribution\n(p-value)")
plt.ylabel("")
g.figure.savefig(f"plots/promoter_prediction/pval_boxplot.pdf", bbox_inches="tight")
plt.clf()


plt.hist(all_differences, bins=50, range = (0,1000))
plt.xlabel("Distance to closest TSS (bp)")
plt.ylabel("No. promoters")
plt.xticks(np.arange(0, 1000, 100))
plt.savefig(f"plots/promoter_prediction/all_distance_from_prodigal_positions.pdf")
plt.clf()
