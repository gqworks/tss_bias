import re, os
import pandas as pd
import pyBigWig
import numpy as np
from pysam import FastaFile
import subprocess

def taxid2ftp(taxid,n=1):
    strain = refseq[(refseq['species_taxid']==int(taxid)) & (refseq['assembly_level']=='Complete Genome')]
    if len(strain) == 0:
        strain = refseq[(refseq['species_taxid']==int(taxid))]
    ftp = strain['ftp_path'][:n].tolist()
    ftp = [re.sub(r'^ftp','https',x) for x in ftp]
    return(ftp)

def asm_2_species(asm):
    name = refseq[refseq['ftp_path'].str.contains(asm)]['organism_name'].item()
    return re.match(r'(^[^ ]+ [^ ]+).*',name).group(1)

def prepare_species_folder(ftp):
    asm = ftp.rsplit('/',1)[1]
    tax_path = f'strains/{asm}'
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
            if not os.path.exists(f'strains/{asm}/{asm}.fna.bwt'): #Index for BWA
                subprocess.call(f'bwa index strains/{asm}/{asm}.fna', shell=True)
            subprocess.call(f'bwa mem -t 64 -x pacbio strains/{asm}/{asm}.fna {DNA_loc}.fastq.gz |samtools view --threads 64 -b - |samtools sort - -o bams/{asm}_{sample}.sorted.bam --threads 64', shell=True)
        elif seq_type == "nanopore":
            if not os.path.exists(f'strains/{asm}/{asm}.fna.bwt'): #Index for BWA
                subprocess.call(f'bwa index strains/{asm}/{asm}.fna', shell=True)
            subprocess.call(f'bwa mem -t 64 -x ont2d strains/{asm}/{asm}.fna {DNA_loc}.fastq.gz |samtools view --threads 64 -b - |samtools sort - -o bams/{asm}_{sample}.sorted.bam --threads 64', shell=True)
        elif seq_type == "illumina":
            subprocess.call(f'bowtie2 -p 64 -x strains/{asm}/{asm}.fna -1 {DNA_loc}_1.fastq.gz -2 {DNA_loc}_2.fastq.gz |samtools view --threads 64 -b - |samtools sort - -o bams/{asm}_{sample}.sorted.bam --threads 64', shell=True)
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
        np.savetxt(f'matrices/{asm}_{sample}.csv',x,fmt='%s',delimiter=',')

if not os.path.exists('strains/'):
        os.mkdir('strains/')
if not os.path.exists('bams/'):
        os.mkdir('bams/')        
if not os.path.exists('matrices/'):
        os.mkdir('matrices/')
if not os.path.exists('beds/'):
        os.mkdir('beds/')

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
Franzosa_mpa = Franzosa_mpa.loc[((Franzosa_mpa > 1).sum(axis=1) >= 5),:] # Filter species with atleast 1% abundance in atleast 5 samples
Franzosa_mpa[Franzosa_mpa < 1] = np.nan
Franzosa_mpa = Franzosa_mpa.drop(39491) #Eubacterium Rectale no reference in database.

seq_type = 'illumina'
for sample in Franzosa_mpa.columns:
    DNA_loc = Franzosa_8[sample]
    for taxid in Franzosa_mpa.index:
        if np.isnan(Franzosa_mpa.loc[taxid,sample]):
            continue
        ftps = taxid2ftp(taxid,3)
        for ftp in ftps:
            asm = prepare_species_folder(ftp)
            calc_cov(asm,sample,DNA_loc,seq_type)


### Alternate Illumina datasets ###
with open('arg_samples') as f:
    arg_samples = f.read().splitlines()
arg_samples = {x.split('\t')[0]:x.split('\t')[1] for x in arg_samples}

with open('hyp_samples') as f:
    hyp_samples = f.read().splitlines()
hyp_samples = {x.split('\t')[0]:x.split('\t')[1] for x in hyp_samples}
merged_samples = {**Franzosa_8, **arg_samples, **hyp_samples}

alt_illumina_mpa = mpa_res.loc[((mpa_res[Franzosa_8.keys()] > 1).sum(axis=1) >= 5) & ((mpa_res[arg_samples.keys()] > 1).sum(axis=1) >= 5) & ((mpa_res[hyp_samples.keys()] > 1).sum(axis=1) >= 5),:]
alt_illumina_mpa[alt_illumina_mpa < 1] = np.nan
alt_illumina_mpa = alt_illumina_mpa.drop(39491) #Eubacterium Rectale no reference in database.

seq_type = 'illumina'
for sample in alt_illumina_mpa.columns:
    DNA_loc = merged_samples[sample]
    for taxid in alt_illumina_mpa.index:
        if np.isnan(alt_illumina_mpa.loc[taxid,sample]):
            continue
        ftps = taxid2ftp(taxid,3)
        for ftp in ftps:
            asm = prepare_species_folder(ftp)
            calc_cov(asm,sample,DNA_loc,seq_type)

### Long read data ###
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
# Filter species with atleast 0.5% abundance in atleast 4 samples in each dataset
merged_abund = merged_abund.loc[((merged_abund[Franzosa_8.keys()] > 0.5).sum(axis=1) >= 4) & ((merged_abund[pacbio_samples.keys()] > 0.5).sum(axis=1) >= 4) & ((merged_abund[nanopore_samples.keys()] > 0.5).sum(axis=1) >= 4),:] 
merged_abund[merged_abund < 0.5] = np.nan
merged_abund = merged_abund.drop(39491) #Eubacterium Rectale no reference in database.

for sample in merged_abund.columns:
    DNA_loc = merged_samples[sample]
    if sample in pacbio_samples:
        seq_type = 'pacbio'
    elif sample in nanopore_samples:
        seq_type = 'nanopore'
    else:
        seq_type = 'illumina'
    for taxid in merged_abund.index:
        if np.isnan(merged_abund.loc[taxid,sample]):
            continue
        ftps = taxid2ftp(taxid,3)
        for ftp in ftps:
            asm = prepare_species_folder(ftp)
            calc_cov(asm,sample,DNA_loc,seq_type)

            
### ARG Dataset Strain Diversity read alignment analysis ###
with open('arg_samples') as f:
    arg_samples = f.read().splitlines()
arg_samples = {x.split('\t')[0]:x.split('\t')[1] for x in arg_samples}

arg_mpa = mpa_res.loc[:,arg_samples]
arg_mpa = arg_mpa.loc[((arg_mpa > 1).sum(axis=1) >= 5),:] # Filter species with atleast 1% abundance in atleast 50% samples
arg_mpa[arg_mpa < 1] = np.nan
arg_mpa = arg_mpa.drop(39491) #Eubacterium Rectale no reference in database.

seq_type = 'illumina'
for sample in arg_mpa.columns:
    DNA_loc = arg_samples[sample]
    for taxid in arg_mpa.index:
        if np.isnan(arg_mpa.loc[taxid,sample]):
            continue
        ftps = taxid2ftp(taxid,3)
        for ftp in ftps:
            asm = prepare_species_folder(ftp)
            calc_cov(asm,sample,DNA_loc,seq_type)
