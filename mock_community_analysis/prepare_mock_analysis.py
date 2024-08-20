import subprocess
import os

### Prepare reference files
# Subset HiFi Long reads
#if not os.path.exists('mock_hifi_ref/hifi_100K_sample.fa'):
#    subprocess.call('reformat.sh -qin=33 in=primary_seq/SRR11606871_subreads.fastq.gz out=mock_hifi_ref/hifi_100K_sample.fa samplereadstarget=100000 sampleseed=13', shell = True)
    
# Indexing
#subprocess.call('bowtie2-build --thread 64 mock_ref/merged_genomes.fna mock_ref/merged_genomes.fna', shell = True)
#subprocess.call('bowtie2-build --thread 64 mock_hifi_ref/hifi_100K_sample.fa mock_hifi_ref/hifi_100K_sample.fa', shell = True)

# Align 2 Reference genomes
if not os.path.exists('bams/SRR8359173_all_ref.sorted.bam'):
    subprocess.call('bowtie2 --threads 64 -x mock_ref/merged_genomes.fna -1 primary_seq/SRR8359173_1.fastq.gz -2 primary_seq/SRR8359173_2.fastq.gz |samtools view --threads 64 -b - |samtools sort - -o bams/SRR8359173_all_ref.sorted.bam --threads 64', shell = True)
    subprocess.call('samtools index bams/SRR8359173_all_ref.sorted.bam -@ 64', shell = True)
    
#Align 2 Long read reference
#if not os.path.exists('bams/SRR8359173_hifi.sorted.bam'):
#    subprocess.call('bowtie2 --threads 64 -a -x mock_hifi_ref/hifi_100K_sample.fa -1 illumina/SRR8359173_1.fastq.gz -2 illumina/SRR8359173_2.fastq.gz|samtools view --threads 64 -b - |samtools sort - -o bams/SRR8359173_hifi.sorted.bam --threads 64', shell = True)
#    subprocess.call('samtools index bams/SRR8359173_hifi.sorted.bam -@ 64', shell = True)

# Get coverage
subprocess.call('bamCoverage --bam bams/SRR8359173_all_ref.sorted.bam -p 64 -o beds/SRR8359173_all_ref.bw -of bigwig', shell = True)
#subprocess.call('bamCoverage --bam bams/SRR8359173_hifi.sorted.bam -p 64 -o beds/SRR8359173_hifi.bw -of bigwig', shell = True)

# Annotate genes w/ prodigal
#subprocess.call('prodigal -i mock_ref/merged_genomes.fna -o mock_ref/merged_genomes_prodigal.gff -f gff -p meta -d mock_ref/merged_genomes_prodigal.fna', shell = True)
#subprocess.call('prodigal -i mock_hifi_ref/hifi_100K_sample.fa -o mock_hifi_ref/hifi_100K_sample_prodigal.gff -f gff -p meta -d mock_hifi_ref/hifi_100K_sample_prodigal.fna', shell = True)

#Map genes from long reads to reference genome genes to annotate the long reads species origin
subprocess.call('minimap2 mock_ref/merged_genomes_prodigal.fna mock_hifi_ref/hifi_100K_sample_prodigal.fna > minimap_res/hifi_all_minimap2.paf', shell = True)
