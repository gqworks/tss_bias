import subprocess
import pandas as pd
import numpy as np
import re, os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def taxid2asm(taxid,n=1):
    strain = refseq[(refseq['species_taxid']==int(taxid)) & (refseq['assembly_level']=='Complete Genome')]
    if len(strain) == 0:
        strain = refseq[(refseq['species_taxid']==int(taxid))]
    ftp = strain['ftp_path'][:n].tolist()
    asm = [x.rsplit('/',1)[1] for x in ftp]
    return(asm)

def scale_mat(d):
    return (d-d.mean(axis=1)[:,None]) / d.std(axis=1)[:,None]

def calc_bias(x):
    left_sum = np.sum(x[:500])
    right_sum = np.sum(x[500:])
    return(right_sum-left_sum)

def get_bias(asm,sample):
    mat = pd.read_table(f'matrices/{asm}_{sample}.csv', sep = ',', header=None, index_col=0)
    scaled_mat = scale_mat(mat)
    scaled_mat = scaled_mat.fillna(0)
    bias = np.mean(list(map(calc_bias, np.array(scaled_mat))))
    return(bias)

def add_p_val(ax,lft,rgt,y,h,p):
    ax.plot([lft, lft, rgt, rgt], [y, y+h, y+h, y], lw=1.5, c='k')
    ax.text((lft + rgt) * .5, y+h, ('n.s.' if p > 0.15 else 'p < %.2g' if p > 0.001 else 'p < %.1g') % max(p+1e-20, 1e-20), ha='center', va='bottom', color='k')

def asm_2_species(asm):
    name = refseq[refseq['ftp_path'].str.contains(asm)]['organism_name'].item()
    name = re.match(r'(^[^ ]+ [^ ]+).*',name).group(1)
    if name == '[Eubacterium] eligens':
        name = 'Lachnospira eligens'
    return name

### Read NCBI Refseq table ###
refseq = pd.read_table('src/assembly_summary_refseq.txt', header = 1)

### Franzosa dataset (8 metagenomics and metatranscriptomics matched samples) ###
with open('Franzosa_8_samples') as f:
    Franzosa_8 = f.read().splitlines()
Franzosa_8 = {x.split('\t')[0]:x.split('\t')[1] for x in Franzosa_8}

### Read species abundances ###
mpa_res = pd.read_table('metaphlan3_res/merged_mpa.txt', comment= '#')
mpa_res = mpa_res[mpa_res['clade_name'].str.contains("s__")]
mpa_res.index = [int(re.match('.*\|([^\|]*)',x).group(1)) for x in mpa_res['NCBI_tax_id']]
mpa_res = mpa_res.drop(columns=['clade_name', 'NCBI_tax_id'])
Franzosa_mpa = mpa_res.loc[:,Franzosa_8.keys()]
Franzosa_mpa = Franzosa_mpa.loc[((Franzosa_mpa > 1).sum(axis=1) >= 5),:] 
Franzosa_mpa[Franzosa_mpa < 1] = np.nan
Franzosa_mpa = Franzosa_mpa.drop(39491) #Eubacterium Rectale no reference in database.
'''
# Create database sketch of all reference genomes
all_asms = []
for taxid in Franzosa_mpa.index:
    asms = taxid2asm(taxid,3)
    all_asms = all_asms + asms
all_path = [f'strains/{x}/{x}.fna' for x in all_asms]
subprocess.call(f'sylph sketch {" ".join(all_path)} -o sketches/database', shell = True)

# Create sample read sketch database
all_df = pd.DataFrame()
for sample,path in Franzosa_8.items():
    path_id = path.split('/')[-1]
    if not os.path.exists(f'sketches/{sample}_ani_queries.tsv'):
        subprocess.call(f'sylph sketch -1 {path}_1.fastq.gz -2 {path}_2.fastq.gz -d sketches', shell = True)
        subprocess.call(f'sylph query sketches/database.syldb sketches/{path_id}_1.fastq.gz.paired.sylsp -t 3 > sketches/{sample}_ani_queries.tsv', shell = True)
    taxids = Franzosa_mpa.index[~np.isnan(Franzosa_mpa.loc[:,sample])].to_list()
    asms = []
    for taxid in taxids:
        asms = asms+ taxid2asm(taxid,3)
    ani_query = pd.read_csv(f'sketches/{sample}_ani_queries.tsv', sep = '\t')
    ani_query['asm'] = [x.split('/')[1] for x in ani_query.Genome_file]
    ani_query = ani_query[ani_query.asm.isin(asms)]
    ani_query['bias'] = [get_bias(x, sample) for x in asms]
    ani_query['containment_ind_perc'] = [eval(x) for x in ani_query.Containment_ind]
    all_df = pd.concat([all_df,ani_query])
    
ani_stat = stats.pearsonr(all_df['bias'], all_df['Adjusted_ANI']) #(0.13589068933100376, 0.11469363113946436)
stats.pearsonr(all_df['bias'], all_df['containment_ind_perc']) #(0.10876884673748101, 0.20749374489787165)

sns.regplot(data = all_df, x="bias", y="Adjusted_ANI")
plt.ylabel("Containment ANI", fontsize=12)
plt.xlabel("TSS Bias", fontsize=12)
plt.text(15, 95.5, 'R = ' + str(round(ani_stat[0],3)) + '  p = ' + str(round(ani_stat[1],3)), fontsize=12)
plt.savefig(f"plots/containment_hashing/all.pdf")
plt.clf()

for asm in set(all_df.asm):
    dat = all_df[all_df.asm==asm]
    sns.regplot(data = dat, x="bias", y="Adjusted_ANI")
    print(asm, stats.pearsonr(dat['bias'], dat['Adjusted_ANI']))

dataset_asm_2_species = {x:asm_2_species(x) for x in all_asms}
all_df['species'] = all_df['asm'].map(dataset_asm_2_species)
for spe in set(all_df.species):
    dat = all_df[all_df.species==spe]
    sns.regplot(data = dat, x="bias", y="Adjusted_ANI")
    plt.savefig(f"plots/containment_hashing/{spe}.pdf")
    plt.clf()

#Average per species
avg_all_df = all_df[['Sample_file','species','bias','Adjusted_ANI']].groupby(['Sample_file','species']).mean().reset_index()
for spe in set(avg_all_df.species):
    dat = avg_all_df[avg_all_df.species==spe]
    sns.regplot(data = dat, x="bias", y="Adjusted_ANI")
    plt.savefig(f"plots/containment_hashing/{spe}.pdf")
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
    #sns.regplot(data = dat, x="bias", y="Adjusted_ANI")
    #print(asm, stats.pearsonr(dat['bias'], dat['Adjusted_ANI']))
    #plt.savefig(f"plots/containment_hashing/{asm}_unique_reads.pdf")
    #plt.clf()
    
    sns.regplot(data = dat, x="bias", y="containment_ind_perc")
    print(asm, stats.pearsonr(dat['bias'], dat['containment_ind_perc']))
    #plt.savefig(f"plots/containment_hashing/{asm}_unique_reads.pdf")
    #plt.clf()

