import re, os
import pandas as pd
import pyBigWig
import numpy as np
from pysam import FastaFile
import subprocess

def prepare_species_folder(ftp):
    asm = ftp.rsplit('/',1)[1]
    tax_path = f'refseq_genomes/{asm}'
    if not os.path.exists(tax_path):
        os.mkdir(tax_path)
        #Download genome ncbi
        subprocess.call(f'wget {ftp}/{asm}_genomic.fna.gz -O {tax_path}/{asm}.fna.gz', shell= True)
        subprocess.call(f'gzip -df {tax_path}/{asm}.fna.gz', shell= True)

        #Predict genes with prodigal
        subprocess.call(f'prodigal -i {tax_path}/{asm}.fna -o {tax_path}/{asm}_prodigal.gff -d {tax_path}/{asm}_prodigal.fna -f gff', shell= True)

        ###TSS filtering - Remove overlaping TSS (operon start), Allow overlaping with gene bodies
        tss_dat = pd.read_csv(f'{tax_path}/{asm}_prodigal.gff', sep = '\t', header=None, comment = '#')
        tss_dat.drop(columns=[1,2,5,7], inplace = True)
        tss_dat['TSS'] = [x[3] if x[6] == '+' else x[4] for i,x in tss_dat.iterrows()]
        tss_dat = np.array(tss_dat)
        #Remove overlaping TSS -TSS doesnt overlap with TSS
        non_overlap = []
        for chrom in list(set(tss_dat[:,0])):
            subset = tss_dat[np.in1d(tss_dat[:,0],chrom)]
            if(len(subset) == 1):
                non_overlap.append(subset[0])
                continue
            #First case
            if float(subset[0,5])+500 < float(subset[1,5]):
                non_overlap.append(subset[0])
            #Mid case
            for i in range(1,len(subset)-1):
                if float(subset[i,5])-500 > float(subset[i-1,5]) and float(subset[i,5])+500 < float(subset[i+1,5]):
                    non_overlap.append(subset[i])
            #Last case
            if float(subset[len(subset)-1,5])-500 > float(subset[len(subset)-2,5]):
                non_overlap.append(subset[len(subset)-1])
        non_overlap = pd.DataFrame(non_overlap, columns = ["Genome","geneLeft","geneRight","Strand","Desc","Start"])
        non_overlap.to_csv(f'{tax_path}/{asm}_filtered_tss.csv', index = False)

        #Index genome
        genome_fna = f'{tax_path}/{asm}.fna'
        subprocess.call(f'bowtie2-build {genome_fna} {genome_fna}', shell=True)
    return(asm)

def calc_cov(asm, sample, DNA_loc, seq_type):
    ### Align reads ##
    if not os.path.exists(f'bams/{asm}_{sample}.sorted.bam'):
        if seq_type == "pacbio":
            if not os.path.exists(f'refseq_genomes/{asm}/{asm}.fna.bwt'): #Index for BWA
                subprocess.call(f'bwa index refseq_genomes/{asm}/{asm}.fna', shell=True)
            subprocess.call(f'bwa mem -t 64 -x pacbio refseq_genomes/{asm}/{asm}.fna {DNA_loc}/{sample}.fastq.gz |samtools view --threads 64 -b - |samtools sort - -o bams/{asm}_{sample}.sorted.bam --threads 64', shell=True)
        elif seq_type == "nanopore":
            if not os.path.exists(f'refseq_genomes/{asm}/{asm}.fna.bwt'): #Index for BWA
                subprocess.call(f'bwa index refseq_genomes/{asm}/{asm}.fna', shell=True)
            subprocess.call(f'bwa mem -t 64 -x ont2d refseq_genomes/{asm}/{asm}.fna {DNA_loc}/{sample}.fastq.gz |samtools view --threads 64 -b - |samtools sort - -o bams/{asm}_{sample}.sorted.bam --threads 64', shell=True)
        elif seq_type == "illumina":
            subprocess.call(f'bowtie2 -p 64 -x refseq_genomes/{asm}/{asm}.fna -1 {DNA_loc}/{sample}_1.fastq.gz -2 {DNA_loc}/{sample}_2.fastq.gz |samtools view --threads 64 -b - |samtools sort - -o bams/{asm}_{sample}.sorted.bam --threads 64', shell=True)
        else:
            print("Invalid sequence type")
            return(-1)
        subprocess.call(f'samtools index bams/{asm}_{sample}.sorted.bam -@ 64', shell = True)

    #Get read depth of genome
    if not os.path.exists(f'beds/{asm}_{sample}.bw'):
        subprocess.call(f'bamCoverage --bam bams/{asm}_{sample}.sorted.bam -p 64 -o beds/{asm}_{sample}.bw -of bigwig', shell = True)

    ### Generate matrix ###
    if not os.path.exists(f'matrices/{asm}_{sample}.csv'):
        bp_depth = pyBigWig.open(f'beds/{asm}_{sample}.bw')
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
        np.savetxt(f'matrices/{asm}_{sample}.csv',x,fmt='%s',delimiter=',')

if not os.path.exists('refseq_genomes/'):
        os.mkdir('refseq_genomes/')
if not os.path.exists('bams/'):
        os.mkdir('bams/')        
if not os.path.exists('matrices/'):
        os.mkdir('matrices/')
if not os.path.exists('beds/'):
        os.mkdir('beds/')

### Read NCBI Refseq table ###
refseq = pd.read_table('src/assembly_summary_refseq.txt', header = 1, low_memory=False)

names = pd.read_csv('src/names.dmp', sep='\t|\t', engine='python', header=None)
names = names.iloc[:, [0, 2, 6]]  # taxid, name, name class
names.columns = ['taxid', 'name', 'name_class']
names['name_class'] = names['name_class'].str.strip()
scientific_names = names[names['name_class'] == 'scientific name']
def get_species_name(taxid):
    result = scientific_names[scientific_names['taxid'] == taxid]['name']
    if not result.empty:
        return f's__{re.sub(" ","_",result.values[0])}'
    else:
           None

### Read species abundances ###
mpa_res = pd.read_table('mpa4_res/merged_mpa.txt', comment= '#')
mpa_res = mpa_res[(mpa_res['clade_name'].str.contains("s__"))&(~mpa_res['clade_name'].str.contains("t__"))]
mpa_res.index = [re.match('.*\|([^\|]*)',x).group(1) for x in mpa_res['clade_name']]
mpa_res = mpa_res.drop(columns=['clade_name'])

### Franzosa dataset (8 metagenomics and metatranscriptomics matched samples) ###
with open('sample_lists/franzosa_RNALater_samples') as f:
    franzosa_samples = f.read().splitlines()
franzosa_samples = {x.split('\t')[0]:x.split('\t')[1] for x in franzosa_samples}
franzosa_mpa = mpa_res.loc[:,franzosa_samples.keys()]
franzosa_mpa = franzosa_mpa.loc[((franzosa_mpa > 0.5).sum(axis=1) >= 5),:] # Filter species with atleast 0.5% abundance in atleast 5 samples
franzosa_mpa = franzosa_mpa.where(franzosa_mpa > 0.5, np.nan)
'''
selected_genomes = dict.fromkeys(franzosa_mpa.index)
for key in selected_genomes.keys():
    species_name = re.sub('_', ' ', re.sub('s__','',key))
    assembly = refseq[
    (refseq['organism_name'].str.contains(species_name)) &
    (refseq['refseq_category'].isin(['reference genome', 'representative genome']))
    ]
    selected_genomes[key] = assembly['ftp_path'].tolist()

selected_genomes = {'s__Phocaeicola_vulgatus': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/012/825/GCF_000012825.1_ASM1282v1'],
 's__Agathobacter_rectalis': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/022/453/685/GCF_022453685.1_ASM2245368v1'],
 's__Alistipes_putredinis': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/154/465/GCF_000154465.1_ASM15446v1'],
 's__Faecalibacterium_prausnitzii': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/154/385/GCF_000154385.1_ASM15438v1'],
 's__Bacteroides_uniformis': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/044/361/425/GCF_044361425.1_ASM4436142v1'],
 's__Phascolarctobacterium_faecium': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/003/945/365/GCF_003945365.1_PFJ30894_01'],
 's__Parabacteroides_distasonis': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/018/279/895/GCF_018279895.1_ASM1827989v1'],
 's__Gemmiger_formicilis': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/900/167/555/GCF_900167555.1_IMG-taxon_2590828806_annotated_assembly'],
 's__Fusicatenibacter_saccharivorans': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/027/667/425/GCF_027667425.1_ASM2766742v1']}

seq_type = 'illumina'
for sample in franzosa_mpa.columns:
    DNA_loc = franzosa_samples[sample]
    for species in franzosa_mpa.index:
        if np.isnan(franzosa_mpa.loc[species,sample]):
            continue
        ftps = selected_genomes[species]
        for ftp in ftps:
            asm = prepare_species_folder(ftp)
            calc_cov(asm,sample,DNA_loc,seq_type)
'''
        
### HMP2 dataset ###
with open('sample_lists/hmp2_samples') as f:
    hmp2_samples = f.read().splitlines()
hmp2_samples = {x.split('\t')[0]:x.split('\t')[1] for x in hmp2_samples}
hmp2_mpa = mpa_res.loc[:,hmp2_samples.keys()]
hmp2_mpa = hmp2_mpa.loc[((hmp2_mpa > 0.5).sum(axis=1) >= 30),:] # Filter species with atleast 0.5% abundance in atleast 30 samples
hmp2_mpa = hmp2_mpa.where(hmp2_mpa > 0.5, np.nan)
'''
selected_genomes = dict.fromkeys(hmp2_mpa.index)
for key in selected_genomes.keys():
    species_name = re.sub('_', ' ', re.sub('s__','',key))
    assembly = refseq[
    (refseq['organism_name'].str.contains(species_name)) &
    (refseq['refseq_category'].isin(['reference genome', 'representative genome']))
    ]
    selected_genomes[key] = assembly['ftp_path'].tolist()

selected_genomes = {'s__Phocaeicola_vulgatus': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/012/825/GCF_000012825.1_ASM1282v1'],
 's__Agathobacter_rectalis': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/022/453/685/GCF_022453685.1_ASM2245368v1'],
 's__Alistipes_putredinis': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/154/465/GCF_000154465.1_ASM15446v1'],
 's__Faecalibacterium_prausnitzii': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/154/385/GCF_000154385.1_ASM15438v1'],
 's__Bacteroides_uniformis': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/044/361/425/GCF_044361425.1_ASM4436142v1'],
 's__Parabacteroides_distasonis': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/018/279/895/GCF_018279895.1_ASM1827989v1'],
 's__Bacteroides_caccae': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/018/292/205/GCF_018292205.1_ASM1829220v1']}

seq_type = 'illumina'
for sample in hmp2_mpa.columns:
    DNA_loc = hmp2_samples[sample]
    for species in hmp2_mpa.index:
        if np.isnan(hmp2_mpa.loc[species,sample]):
            continue
        ftps = selected_genomes[species]
        for ftp in ftps:
            asm = prepare_species_folder(ftp)
            calc_cov(asm,sample,DNA_loc,seq_type)


### Alternate Illumina datasets ###
with open('sample_lists/arg_samples') as f:
    arg_samples = f.read().splitlines()
arg_samples = {x.split('\t')[0]:x.split('\t')[1] for x in arg_samples}

with open('sample_lists/hyp_samples') as f:
    hyp_samples = f.read().splitlines()
hyp_samples = {x.split('\t')[0]:x.split('\t')[1] for x in hyp_samples}

merged_samples = {**franzosa_samples, **hmp2_samples, **arg_samples, **hyp_samples}
alt_illumina_mpa = mpa_res.loc[:,merged_samples.keys()]
alt_illumina_mpa = alt_illumina_mpa.loc[((alt_illumina_mpa[franzosa_samples.keys()] > 0.5).sum(axis=1) >= 5) & 
                               ((alt_illumina_mpa[hmp2_samples.keys()] > 0.5).sum(axis=1) >= 30) &
                               ((alt_illumina_mpa[arg_samples.keys()] > 0.5).sum(axis=1) >= 5) & 
                               ((alt_illumina_mpa[hyp_samples.keys()] > 0.5).sum(axis=1) >= 5),:]
alt_illumina_mpa = alt_illumina_mpa.where(alt_illumina_mpa > 0.5, np.nan)

selected_genomes = dict.fromkeys(alt_illumina_mpa.index)
for key in selected_genomes.keys():
    species_name = re.sub('_', ' ', re.sub('s__','',key))
    assembly = refseq[
    (refseq['organism_name'].str.contains(species_name)) &
    (refseq['refseq_category'].isin(['reference genome', 'representative genome']))
    ]
    selected_genomes[key] = assembly['ftp_path'].tolist()

selected_genomes = {'s__Phocaeicola_vulgatus': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/012/825/GCF_000012825.1_ASM1282v1'],
 's__Agathobacter_rectalis': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/022/453/685/GCF_022453685.1_ASM2245368v1'],
 's__Alistipes_putredinis': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/154/465/GCF_000154465.1_ASM15446v1'],
 's__Faecalibacterium_prausnitzii': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/154/385/GCF_000154385.1_ASM15438v1'],
 's__Bacteroides_uniformis': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/044/361/425/GCF_044361425.1_ASM4436142v1']}

seq_type = 'illumina'
for sample in alt_illumina_mpa.columns:
    DNA_loc = merged_samples[sample]
    for species in alt_illumina_mpa.index:
        if np.isnan(alt_illumina_mpa.loc[species,sample]):
            continue
        ftps = selected_genomes[species]
        for ftp in ftps:
            asm = prepare_species_folder(ftp)
            calc_cov(asm,sample,DNA_loc,seq_type)
'''
            
### Long read data ###
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

selected_genomes = dict.fromkeys(merged_abund.index)
for key in selected_genomes.keys():
    species_name = re.sub('_', ' ', re.sub('s__','',key))
    assembly = refseq[
    (refseq['organism_name'].str.contains(species_name)) &
    (refseq['refseq_category'].isin(['reference genome', 'representative genome']))
    ]
    selected_genomes[key] = assembly['ftp_path'].tolist()

selected_genomes = {'s__Phocaeicola_vulgatus': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/012/825/GCF_000012825.1_ASM1282v1'],
 's__Bacteroides_uniformis': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/044/361/425/GCF_044361425.1_ASM4436142v1'],
 's__Agathobacter_rectalis': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/022/453/685/GCF_022453685.1_ASM2245368v1'],
 's__Alistipes_putredinis': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/154/465/GCF_000154465.1_ASM15446v1'],
 's__Faecalibacterium_prausnitzii': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/154/385/GCF_000154385.1_ASM15438v1']}

for sample in merged_abund.columns:
    DNA_loc = merged_samples[sample]
    if sample in pacbio_samples:
        seq_type = 'pacbio'
    elif sample in nanopore_samples:
        seq_type = 'nanopore'
    else:
        seq_type = 'illumina'
    for species in merged_abund.index:
        if np.isnan(merged_abund.loc[species,sample]):
            continue
        ftps = selected_genomes[species]
        for ftp in ftps:
            asm = prepare_species_folder(ftp)
            calc_cov(asm,sample,DNA_loc,seq_type)

'''
### ARG Dataset Strain Diversity read alignment analysis ###
arg_mpa = mpa_res.loc[:,arg_samples]
arg_mpa = arg_mpa.loc[((arg_mpa > 0.5).sum(axis=1) >= 10),:] # Filter species with atleast 0.5% abundance in atleast 10 samples
arg_mpa = arg_mpa.where(arg_mpa > 0.5, np.nan)

selected_genomes = dict.fromkeys(arg_mpa.index)
for key in selected_genomes.keys():
    species_name = re.sub('_', ' ', re.sub('s__','',key))
    assembly = refseq[
    (refseq['organism_name'].str.contains(species_name)) &
    (refseq['refseq_category'].isin(['reference genome', 'representative genome']))
    ]
    selected_genomes[key] = assembly['ftp_path'].tolist()

selected_genomes = {'s__Phocaeicola_vulgatus': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/012/825/GCF_000012825.1_ASM1282v1'],
 's__Bacteroides_uniformis': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/044/361/425/GCF_044361425.1_ASM4436142v1'],
 's__Agathobacter_rectalis': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/022/453/685/GCF_022453685.1_ASM2245368v1'],
 's__Parabacteroides_distasonis': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/018/279/895/GCF_018279895.1_ASM1827989v1'],
 's__Ruminococcus_bromii': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/014/863/965/GCF_014863965.1_ASM1486396v1'],
 's__Roseburia_faecis': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/015/550/045/GCF_015550045.1_ASM1555004v1'],
 's__Faecalibacterium_prausnitzii': ['https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/154/385/GCF_000154385.1_ASM15438v1']}

seq_type = 'illumina'
for sample in arg_mpa.columns:
    DNA_loc = arg_samples[sample]
    for species in arg_mpa.index:
        if np.isnan(arg_mpa.loc[species,sample]):
            continue
        ftps = selected_genomes[species]
        for ftp in ftps:
            asm = prepare_species_folder(ftp)
            calc_cov(asm,sample,DNA_loc,seq_type)
'''