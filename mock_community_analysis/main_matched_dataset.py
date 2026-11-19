import subprocess
import pandas as pd
import re, os
import numpy as np
import pyBigWig

def calc_cov(asm, sample, DNA_loc):
    ### Align reads ##
    if not os.path.exists(f'bams/{asm}_{sample}.sorted.bam'):
        subprocess.call(f'bowtie2 -p 64 -x strains/{asm}/{asm}.fna -1 {DNA_loc}_1.fastq.gz -2 {DNA_loc}_2.fastq.gz |samtools view --threads 64 -b - |samtools sort - -o bams/{asm}_{sample}.sorted.bam --threads 64', shell=True)
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

def align2nanopore(samp_id, shortread_loc, longread_loc):
    #Index longread ref
    if not os.path.exists(f'{longread_loc}.fasta.1.bt2'):
        subprocess.call(f'bowtie2-build {longread_loc}.fasta {longread_loc}.fasta', shell=True)
    #Align shortread to longread
    if not os.path.exists(f'bams/longread_{samp_id}.sorted.bam'):
        subprocess.call(f'bowtie2 -p 64 -x {longread_loc}.fasta -1 {shortread_loc}_1.fastq.gz -2 {shortread_loc}_2.fastq.gz |samtools view --threads 64 -b - |samtools sort - -o bams/longread_{samp_id}.sorted.bam --threads 64', shell=True)
        subprocess.call(f'samtools index bams/longread_{samp_id}.sorted.bam -@ 64', shell = True)
    #Get read depth of genome
    if not os.path.exists(f'beds/longread_{samp_id}.bw'):
        subprocess.call(f'bamCoverage --bam bams/longread_{samp_id}.sorted.bam -p 64 -o beds/longread_{samp_id}.bw -of bigwig', shell = True)
            
            
###Initialise Data
refseq = pd.read_table('../src/assembly_summary_refseq.txt', header = 1, low_memory=False)
with open('matched_dataset_idlist') as f:
    samples = f.read().splitlines()
samples = [x.split('\t') for x in samples]

#Run metaphlan
for samp in samples:
    if not os.path.exists(f'metaphlan3_res/{samp[0]}.txt'):
        subprocess.call(f'metaphlan primary_seq/{samp[1]}_1.fastq,primary_seq/{samp[1]}_2.fastq --input_type fastq --bowtie2out metaphlan3_res/{samp[0]}.bowtie2.bz2 --nproc 64 -o metaphlan3_res/{samp[0]}.txt', shell=True)

#Read species abundances
mpa_res = pd.read_table('metaphlan3_res/merged_mpa.txt', comment= '#')
mpa_res = mpa_res[mpa_res['clade_name'].str.contains("s__")]
mpa_res.index = [re.match('.*\|([^\|]*)',x).group(1) for x in mpa_res['clade_name']]
mpa_res = mpa_res.drop(columns=['clade_name','NCBI_tax_id'])
mpa_res = mpa_res.loc[(mpa_res > 0.5).sum(axis=1) > 5,:] # Filter species with atleast 0.5% abundance in more than 5 samples
mpa_res = mpa_res.where(mpa_res > 0.5, np.nan)
mpa_res = mpa_res.rename(index={'s__Bacteroides_vulgatus': 's__Phocaeicola_vulgatus', 's__Eubacterium_hallii':'s__Anaerobutyricum_hallii'})

selected_genomes = dict.fromkeys(mpa_res.index)
for key in selected_genomes.keys():
    species_name = re.sub('_', ' ', re.sub('s__','',key))
    assembly = refseq[
    (refseq['organism_name'].str.contains(species_name)) &
    (refseq['refseq_category'].isin(['reference genome', 'representative genome']))
    ]
    selected_genomes[key] = assembly['ftp_path'].tolist()

selected_genomes = {'s__Bacteroides_uniformis': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/044/361/425/GCF_044361425.1_ASM4436142v1',
 's__Phocaeicola_vulgatus': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/012/825/GCF_000012825.1_ASM1282v1',
 's__Alistipes_putredinis': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/154/465/GCF_000154465.1_ASM15446v1',
 's__Anaerobutyricum_hallii': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/018/140/635/GCF_018140635.1_ASM1814063v1',
 's__Anaerostipes_hadrus': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/030/864/025/GCF_030864025.1_ASM3086402v1',
 's__Dorea_longicatena': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/025/150/085/GCF_025150085.1_ASM2515008v1',
 's__Fusicatenibacter_saccharivorans': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/027/667/425/GCF_027667425.1_ASM2766742v1',
 's__Roseburia_faecis': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/015/550/045/GCF_015550045.1_ASM1555004v1',
 's__Faecalibacterium_prausnitzii': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/154/385/GCF_000154385.1_ASM15438v1'}

'''
#Subset nanopore longreads to reduce reference redundancy and computational speed
for samp in samples:
    nanopore_loc = f'primary_seq/{samp[2]}'
    #Output as fasta, as prodigal cannot take gzipped input
    if not os.path.exists(f'{nanopore_loc}_100K.fasta'):
        subprocess.call(f'reformat.sh -qin=33 in={nanopore_loc}.fastq.gz out={nanopore_loc}_100K.fasta samplereadstarget=100000 sampleseed=13', shell = True)
    
    #Call prodigal on long read reference
    if not os.path.exists(f'{nanopore_loc}_100K_prodigal.gff'):
        subprocess.call(f'prodigal -q -p meta -i {nanopore_loc}_100K.fasta -o {nanopore_loc}_100K_prodigal.gff -d {nanopore_loc}_100K_prodigal.fna -f gff', shell= True)

# Generate reference genome (strain) folders
for ftp in selected_genomes.values():
    prepare_species_folder(ftp)
    
#Merge prodigal gene sequences of all strains - Run after creating all strain folders
if not os.path.exists('strains/merged_prodigal.fna'):
    subprocess.call('cat strains/*/*_prodigal.fna > strains/merged_prodigal.fna', shell= True)
'''
    
#Align reads
for samp in samples:
    illumina_loc = f'primary_seq/{samp[1]}'
    nanopore_loc = f'primary_seq/{samp[2]}_100K'
    for species in mpa_res.index:
        if np.isnan(mpa_res.loc[species,samp[0]]):
            continue
        asm = selected_genomes[species].rsplit('/',1)[1]
        calc_cov(asm,samp[0],illumina_loc)
            
    #Align to nanopore reference
    align2nanopore(samp[0],illumina_loc,nanopore_loc)

    #Map genes from long reads to reference genome genes to annotate the long reads species origin
    if not os.path.exists(f'minimap_res/{samp[0]}_minimap2.paf'):
        subprocess.call(f'minimap2 strains/merged_prodigal.fna {nanopore_loc}_prodigal.fna > minimap_res/{samp[0]}_minimap2.paf', shell = True)

'''
#Simulate short reads from long reads
for samp in samples:
    fna_file = f'primary_seq/{samp[2]}_100K.fasta'
    if not os.path.exists(f'simu_seq/{samp[0]}_1.fq'):
        subprocess.call(f'ml ART && art_illumina -na -ss HS25 -i {fna_file} -p -l 150 --fcov 30 -m 200 -s 10 -o simu_seq/{samp[0]}_', shell= True)
    if not os.path.exists(f'bams/{samp[0]}_simu.sorted.bam'):
        subprocess.call(f'bowtie2 --threads 64 -x {fna_file} -1 simu_seq/{samp[0]}_1.fq -2 simu_seq/{samp[0]}_2.fq |samtools view --threads 64 -b - |samtools sort - -o bams/{samp[0]}_simu.sorted.bam --threads 64', shell=True)
        subprocess.call(f'samtools index bams/{samp[0]}_simu.sorted.bam -@ 64',shell=True)
    if not os.path.exists(f'beds/{samp[0]}_simu.bw'):
        subprocess.call(f'bamCoverage --bam bams/{samp[0]}_simu.sorted.bam -p 64 -o beds/{samp[0]}_simu.bw -of bigwig', shell = True)
'''