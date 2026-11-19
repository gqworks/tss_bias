import subprocess
import pandas as pd
import numpy as np
import re, os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from Bio import SeqIO

def scale_mat(d):
    mean_vals = d.mean(axis=1).to_numpy()
    std_vals = d.std(axis=1).to_numpy()
    return (d - mean_vals[:, None]) / std_vals[:, None]

def calc_bias(x):
    left_sum = np.sum(x[:500])
    right_sum = np.sum(x[500:])
    return(right_sum-left_sum)

def get_bias(asm,sample):
    if not os.path.exists(f'matrices/{asm}_{sample}.csv'):
        return np.nan
    mat = pd.read_table(f'matrices/{asm}_{sample}.csv', sep = ',', header=None, index_col=0)
    scaled_mat = scale_mat(mat)
    scaled_mat = scaled_mat.fillna(0)
    bias = np.mean(list(map(calc_bias, np.array(scaled_mat))))
    return(bias)

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

# Create database sketch of all reference genomes
#all_path = [f'refseq_genomes/{hmp2_genomes[x]}/{hmp2_genomes[x]}.fna' for x in hmp2_mpa.index]
#subprocess.call(f'sylph sketch {" ".join(all_path)} -o sketches/hmp2_database', shell = True)

# Create database of TSS regions of all reference genomes
for species, asm in hmp2_genomes.items():
    bed_file = pd.read_csv(f'refseq_genomes/{asm}/{asm}_filtered_tss.csv')
    bed_file['TSSLeft'] = bed_file['Start']-501
    #bed_file['TSSRight'] = bed_file['Start']+500
    bed_file = bed_file[['Genome','TSSLeft','Start']]
    bed_file = bed_file[bed_file['TSSLeft']>0]
    bed_file.to_csv(f'refseq_genomes/{asm}/{asm}_tss.bed', sep = '\t', header = False, index = False)
    subprocess.call(f'bedtools getfasta -fi refseq_genomes/{asm}/{asm}.fna -bed refseq_genomes/{asm}/{asm}_tss.bed -fo refseq_genomes/{asm}/{asm}_tss.fna', shell = True)
all_path = [f'refseq_genomes/{hmp2_genomes[x]}/{hmp2_genomes[x]}_tss.fna' for x in hmp2_mpa.index]
subprocess.call(f'sylph sketch {" ".join(all_path)} -o sketches/hmp2_tss_database', shell = True)

# Create sample read sketch database
all_df = pd.DataFrame()
for sample,path in hmp2_samples.items():
    #if not os.path.exists(f'sketches/{sample}_ani_queries.tsv'):
    #    subprocess.call(f'sylph sketch -t 32 -1 {path}/{sample}_1.fastq.gz -2 {path}/{sample}_2.fastq.gz -d sketches', shell = True)
    #    subprocess.call(f'sylph query sketches/hmp2_database.syldb sketches/{sample}_1.fastq.gz.paired.sylsp -t 3 > sketches/{sample}_ani_queries.tsv', shell = True)
    #if not os.path.exists(f'sketches/{sample}_tss_ani_queries.tsv'):
    subprocess.call(f'sylph query sketches/hmp2_tss_database.syldb sketches/{sample}_1.fastq.gz.paired.sylsp -t 3 > sketches/{sample}_tss_ani_queries.tsv', shell = True)
    sample_species = hmp2_mpa.index[~np.isnan(hmp2_mpa.loc[:,sample])].to_list()
    asms = [hmp2_genomes[x] for x in sample_species]
    ani_query = pd.read_csv(f'sketches/{sample}_tss_ani_queries.tsv', sep = '\t')
    ani_query['asm'] = [x.split('/')[1] for x in ani_query.Genome_file]
    ani_query = ani_query[ani_query.asm.isin(asms)]
    ani_query['bias'] = [get_bias(x, sample) for x in asms]
    ani_query['containment_ind_perc'] = [eval(x) for x in ani_query.Containment_ind]
    all_df = pd.concat([all_df,ani_query])
all_df.to_csv("plots/hmp2_containment_hashing_tss_results_redo.csv")

# Use sylph profile instead
#all_abund = pd.DataFrame()
#for sample,path in hmp2_samples.items():
#    if not os.path.exists(f'sketches/{sample}_abundances.tsv'):
#        subprocess.call(f'sylph profile -u sketches/hmp2_database.syldb sketches/{sample}_1.fastq.gz.paired.sylsp -t 3 > sketches/{sample}_abundances.tsv', shell = True)
#    abundances = pd.read_csv(f'sketches/{sample}_abundances.tsv', sep = '\t')
#    abundances['asm'] = [x.split('/')[1] for x in abundances.Genome_file]
#    abundances['bias'] = [get_bias(x, sample) for x in abundances['asm']]
#    abundances['containment_ind_perc'] = [eval(x) for x in abundances.Containment_ind]
#    all_abund = pd.concat([all_abund,abundances]) 
#all_abund.to_csv("plots/hmp2_containment_hashing_abundances.csv")

all_df = pd.read_csv("plots/hmp2_containment_hashing_results.csv", index_col = 0)
all_df = pd.read_csv("plots/hmp2_containment_hashing_tss_results_redo.csv", index_col = 0)
all_df['species'] = [list(hmp2_genomes.keys())[list(hmp2_genomes.values()).index(x)] for x in all_df['asm']]
ani_stat = stats.pearsonr(all_df['bias'], all_df['Adjusted_ANI']) #(0.13589068933100376, 0.11469363113946436) HMP2: (0.3847837192016083, 5.038950810759524e-12)
stats.pearsonr(all_df['bias'], all_df['containment_ind_perc']) #(0.10876884673748101, 0.20749374489787165) HMP2: (0.5150553264049424, 1.0015543146150225e-21)

sns.lmplot(data=all_df, x="bias", y="Adjusted_ANI", hue="species")
plt.ylabel("Containment ANI", fontsize=12)
plt.xlabel("TSS Bias", fontsize=12)
plt.text(50, 92.5, 'R = ' + str(round(ani_stat[0],3)) + '  p = ' + str(5.04e-12), fontsize=12)
plt.savefig(f"plots/containment_hashing/hmp2_tss.pdf")
plt.clf()

#Average per species Stats
for spe in set(all_df.species):
    dat = all_df[all_df.species==spe]
    sns.regplot(data = dat, x="bias", y="Adjusted_ANI")
    plt.savefig(f"plots/containment_hashing/hmp2_{spe}.pdf")
    plt.clf()

#Investigate <95% ANI samples
odd_samples = [re.match('.*primary_seq/([^_]*)',x).group(1) for x in all_df[all_df['Adjusted_ANI']<95]['Sample_file'].to_list()]
hmp2_mpa.loc['s__Faecalibacterium_prausnitzii',odd_samples]

all_abund = pd.read_csv("plots/hmp2_containment_hashing_abundances.csv", index_col = 0)
all_df_merged = all_df.set_index(['Sample_file', 'Genome_file']).join(all_abund.set_index(['Sample_file', 'Genome_file'])['Sequence_abundance'])
all_df_merged = all_df_merged.reset_index() 
all_df_merged = all_df_merged.dropna(subset=['Sequence_abundance','species'])
ani_stat = stats.pearsonr(all_df_merged['bias'], all_df_merged['Adjusted_ANI']) # (0.32607129266821433, 2.777293428938e-08)
ani_stat = stats.spearmanr(all_df_merged['bias'], all_df_merged['Adjusted_ANI']) # SpearmanrResult(correlation=0.35027510452480976, pvalue=2.042996339866692e-09)
sns.lmplot(data=all_df_merged, x="bias", y="Adjusted_ANI", hue="species")
plt.ylabel("Containment ANI", fontsize=12)
plt.xlabel("TSS Bias", fontsize=12)
plt.text(50, 96, 'R = ' + str(round(ani_stat[0],3)) + '  p = ' + str(2.78e-8), fontsize=12)
plt.savefig(f"plots/containment_hashing/hmp2_sylph_abund.pdf")
plt.clf()

#Average per species
for spe in set(all_df_merged.species):
    dat = all_df_merged[all_df_merged.species==spe]
    ani_stat = stats.pearsonr(dat['bias'], dat['Adjusted_ANI'])
    print(f'{spe}: R = {str(round(ani_stat[0],3))} p = {str(ani_stat[1])}')
    #sns.regplot(data = dat, x="bias", y="Adjusted_ANI")
    #plt.savefig(f"plots/containment_hashing/hmp2_{spe}.pdf")
    #plt.clf()


for spe in set(avg_all_df_merged.species):
    dat = avg_all_df_merged[avg_all_df_merged.species==spe]
    sns.regplot(data = dat, x="bias", y="Adjusted_ANI")
    plt.savefig(f"plots/containment_hashing/hmp2_{spe}_sylph.pdf")
    plt.clf()

for spe in set(all_df.species):
    dat = all_df[all_df.species==spe]
    sns.regplot(data = dat, x="bias", y="Adjusted_ANI")
    plt.savefig(f"plots/containment_hashing/hmp2_{spe}_tss.pdf")
    plt.clf()

'''
### Perform hashing only of uniquely aligned reads
# Extract uniquely mapped reads
if not os.path.exists('unique_reads/'):
        os.mkdir('unique_reads/')

for sample in Franzosa_mpa.columns:
    for taxid in Franzosa_mpa.index:
        if np.isnan(Franzosa_mpa.loc[taxid,sample]):
            continue
        asm = taxid2asm(taxid,1)[0]
        subprocess.call(f'samtools view --threads 64 -b -q 40 bams/{asm}_{sample}_mp2.sorted.bam| samtools sort -n - -o unique_reads/{asm}_{sample}_mp2_nsorted.bam --threads 64', shell = True)
        subprocess.call(f'bedtools bamtofastq -i unique_reads/{asm}_{sample}_mp2_nsorted.bam -fq unique_reads/{asm}_{sample}_mp2_R1.fastq -fq2 unique_reads/{asm}_{sample}_mp2_R2.fastq', shell = True)


# Create sample read sketch database
all_df = pd.DataFrame()
for sample,path in Franzosa_8.items():
    taxids = Franzosa_mpa.index[~np.isnan(Franzosa_mpa.loc[:,sample])].to_list()
    for taxid in taxids:
        if np.isnan(Franzosa_mpa.loc[taxid,sample]):
            continue
        asm = taxid2asm(taxid,1)[0]

        subprocess.call(f'sylph sketch -1 unique_reads/{asm}_{sample}_mp2_R1.fastq -2 unique_reads/{asm}_{sample}_mp2_R2.fastq -d sketches', shell = True)
        subprocess.call(f'sylph query sketches/database.syldb sketches/{asm}_{sample}_mp2_R1.fastq.paired.sylsp -t 3 > sketches/{asm}_{sample}_mp2_ani_queries.tsv', shell = True)
        ani_query = pd.read_csv(f'sketches/{asm}_{sample}_mp2_ani_queries.tsv', sep = '\t')
        ani_query['asm'] = [x.split('/')[1] for x in ani_query.Genome_file]
        ani_query = ani_query[ani_query['asm']==asm]
        ani_query['bias'] = get_bias(asm, sample)
        ani_query['containment_ind_perc'] = [eval(x) for x in ani_query.Containment_ind]
        all_df = pd.concat([all_df,ani_query])

sns.regplot(data = all_df, x="bias", y="Adjusted_ANI")
stats.pearsonr(all_df['bias'], all_df['Adjusted_ANI']) #(0.29596484840115095, 0.00046811349395038936)
plt.savefig(f"plots/containment_hashing/all_adj_ANI_unique_mp2.pdf")
plt.clf()

sns.regplot(data = all_df, x="bias", y="containment_ind_perc")
stats.pearsonr(all_df['bias'], all_df['containment_ind_perc']) #(0.5704018126121043, 2.8406744549043237e-05)
plt.savefig(f"plots/containment_hashing/all_containment_ind_unique_mp2.pdf")
plt.clf()
            
for asm in set(all_df.asm):
    dat = all_df[all_df.asm==asm]
    sns.regplot(data = dat, x="bias", y="Adjusted_ANI")
    print(asm, stats.pearsonr(dat['bias'], dat['Adjusted_ANI']))
    plt.savefig(f"plots/containment_hashing/{asm}_unique_reads.pdf")
    plt.clf()
    
    sns.regplot(data = dat, x="bias", y="containment_ind_perc")
    print(asm, stats.pearsonr(dat['bias'], dat['containment_ind_perc']))
    plt.savefig(f"plots/containment_hashing/{asm}_unique_reads.pdf")
    plt.clf()
'''

# % of TSS windows to entire genome
all_lens = []
for spe, asm in hmp2_genomes.items():
    tss_length = sum(len(record.seq) for record in SeqIO.parse(f'refseq_genomes/{asm}/{asm}_tss.fna', "fasta"))
    genome_length = sum(len(record.seq) for record in SeqIO.parse(f'refseq_genomes/{asm}/{asm}.fna', "fasta"))
    all_lens.append(tss_length/genome_length)
    print(tss_length/genome_length)
sum(all_lens)/len(all_lens) # 0.49656386344023845
