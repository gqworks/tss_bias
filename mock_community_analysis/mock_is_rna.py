import pysam
import pyBigWig
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

SMALL_BUFFER = 5

def calc_gene_bias(annot,aln_file, read_len = 151, window=50, offset=0):
    inward_reads = []
    outward_reads = []
    for gene in annot:
        inward = 0
        outward = 0
        if gene[6] == '+':
            start = int(gene[3]) + offset - SMALL_BUFFER
            stop = int(gene[3]) + window + offset
            for read in aln_file.fetch(gene[0],start,stop):
                if read.mate_is_unmapped or read.is_unmapped or read.reference_start < start or read.reference_start > stop:
                    continue
                if read.is_reverse:
                    outward += 1
                else:
                    inward += 1
        else:
            stop = int(gene[4]) - read_len - offset + SMALL_BUFFER
            start = int(gene[4]) - read_len - window - offset
            for read in aln_file.fetch(gene[0],start,stop):
                if read.mate_is_unmapped or read.is_unmapped or read.reference_start < start or read.reference_start > stop:
                    continue
                if read.is_reverse:
                    inward += 1
                else:
                    outward += 1
        inward_reads.append(inward)
        outward_reads.append(outward)
    return inward_reads, outward_reads

def calc_gene_bias_single(annot,aln_file, read_len = 151, window=50, offset=0):
    inward_reads = []
    outward_reads = []
    for gene in annot:
        inward = 0
        outward = 0
        if gene[6] == '+':
            start = int(gene[3]) + offset - SMALL_BUFFER
            stop = int(gene[3]) + window + offset
            for read in aln_file.fetch(gene[0],start,stop):
                if read.is_unmapped or read.reference_start < start or read.reference_start > stop:
                    continue
                if read.is_reverse:
                    outward += 1
                else:
                    inward += 1
        else:
            stop = int(gene[4]) - read_len - offset + SMALL_BUFFER
            start = int(gene[4]) - read_len - window - offset
            for read in aln_file.fetch(gene[0],start,stop):
                if read.is_unmapped or read.reference_start < start or read.reference_start > stop:
                    continue
                if read.is_reverse :
                    inward += 1
                else:
                    outward += 1
        inward_reads.append(inward)
        outward_reads.append(outward)
    return inward_reads, outward_reads

def add_p_val(ax,lft,rgt,y,h,p):
    ax.plot([lft, lft, rgt, rgt], [y, y+h, y+h, y], lw=1.5, c='k')
    ax.text((lft + rgt) * .5, y+h, ('n.s.' if p > 0.15 else 'p < %.2g' if p > 0.001 else 'p < %.1g') % max(p+1e-20, 1e-20), ha='center', va='bottom', color='k')

def adj_ratio(a,b):
    return (a+1)/(b+1)


# Load alignments
aln_file = pysam.AlignmentFile(f'bams/SRR8359173_all_ref.sorted.bam', "rb")
aln_file.references
#aln_file = pysam.AlignmentFile("/groups/cgsd/gordonq/TSS_depth/intergenic_hypothesis/mock_community/SRR8359173_all_ref.sorted.bam", "rb")

# Load gene regions
gff_ref = pd.read_table('mock_ref/merged_genomes_prodigal.gff', comment = '#', header=None)
gff_ref['len'] = gff_ref[4]-gff_ref[3]
gff_ref = gff_ref[gff_ref['len'] > 500]
gff_ref = gff_ref[gff_ref[3] > 750] #Remove start gene to prevent fetching before reference start
gff_ref_np = np.array(gff_ref)

#Get read length
for read in aln_file.fetch():
    read_len = len(read.get_forward_sequence())
    break

inward_offset_0, outward_offset_0 = calc_gene_bias(gff_ref_np,aln_file,read_len=read_len,window=50,offset=0)
inward_offset_100, outward_offset_100 = calc_gene_bias(gff_ref_np,aln_file,read_len=read_len,window=50,offset=100)
inward_offset_200, outward_offset_200 = calc_gene_bias(gff_ref_np,aln_file,read_len=read_len,window=50,offset=200)
inward_offset_300, outward_offset_300 = calc_gene_bias(gff_ref_np,aln_file,read_len=read_len,window=50,offset=300)
inward_offset_400, outward_offset_400 = calc_gene_bias(gff_ref_np,aln_file,read_len=read_len,window=50,offset=400)
inward_offset_0_s, outward_offset_0_s = calc_gene_bias_single(gff_ref_np,aln_file,read_len=read_len,window=50,offset=0)
inward_offset_100_s, outward_offset_100_s = calc_gene_bias_single(gff_ref_np,aln_file,read_len=read_len,window=50,offset=100)
inward_offset_200_s, outward_offset_200_s = calc_gene_bias_single(gff_ref_np,aln_file,read_len=read_len,window=50,offset=200)
inward_offset_300_s, outward_offset_300_s = calc_gene_bias_single(gff_ref_np,aln_file,read_len=read_len,window=50,offset=300)
inward_offset_400_s, outward_offset_400_s = calc_gene_bias_single(gff_ref_np,aln_file,read_len=read_len,window=50,offset=400)

###The differences
diff_offest_0 = [adj_ratio(a,b) for a, b in zip(inward_offset_0, outward_offset_0)]
diff_offest_100 = [adj_ratio(a,b) for a, b in zip(inward_offset_100, outward_offset_100)]
diff_offest_200 = [adj_ratio(a,b) for a, b in zip(inward_offset_200, outward_offset_200)]
diff_offest_300 = [adj_ratio(a,b) for a, b in zip(inward_offset_300, outward_offset_300)]
diff_offest_400 = [adj_ratio(a,b) for a, b in zip(inward_offset_400, outward_offset_400)]
diff_offest_0_s = [adj_ratio(a,b) for a, b in zip(inward_offset_0, outward_offset_0_s)]
diff_offest_100_s = [adj_ratio(a,b) for a, b in zip(inward_offset_100_s, outward_offset_100_s)]
diff_offest_200_s = [adj_ratio(a,b) for a, b in zip(inward_offset_200_s, outward_offset_200_s)]
diff_offest_300_s = [adj_ratio(a,b) for a, b in zip(inward_offset_300_s, outward_offset_300_s)]
diff_offest_400_s = [adj_ratio(a,b) for a, b in zip(inward_offset_400_s, outward_offset_400_s)]

dat_diff = pd.DataFrame({
    "diff_offest_0": diff_offest_0, "diff_offest_0_s": diff_offest_0_s,
    "diff_offest_100": diff_offest_100, "diff_offest_100_s": diff_offest_100_s,
    "diff_offest_200": diff_offest_200, "diff_offest_200_s": diff_offest_200_s,
    "diff_offest_300": diff_offest_300, "diff_offest_300_s": diff_offest_300_s,
    "diff_offest_400": diff_offest_400, "diff_offest_400_s": diff_offest_400_s
})

dat_diff_melt = dat_diff.melt()
dat_diff_melt['mapping_type'] = ['singleton' if '_s' in x else 'pair' for x in dat_diff_melt['variable']]
dat_diff_melt['offset'] = [re.match('.*_(\d+).*',x).group(1) for x in dat_diff_melt['variable']]

upper_lim = 1.4
sns.boxplot(data = dat_diff_melt, x = 'offset', y= 'value', hue='mapping_type', showfliers=False)
add_p_val(plt,-0.2,0.2,upper_lim+0.2,0.02,mannwhitneyu(dat_diff_melt.loc[(dat_diff_melt.mapping_type=='pair')&(dat_diff_melt.offset=='0'),'value'],dat_diff_melt.loc[(dat_diff_melt.mapping_type=='singleton')&(dat_diff_melt.offset=='0'),'value']).pvalue)
add_p_val(plt,0.8,1.2,upper_lim+0.2,0.02,mannwhitneyu(dat_diff_melt.loc[(dat_diff_melt.mapping_type=='pair')&(dat_diff_melt.offset=='100'),'value'],dat_diff_melt.loc[(dat_diff_melt.mapping_type=='singleton')&(dat_diff_melt.offset=='100'),'value']).pvalue)
add_p_val(plt,1.8,2.2,upper_lim+0.2,0.02,mannwhitneyu(dat_diff_melt.loc[(dat_diff_melt.mapping_type=='pair')&(dat_diff_melt.offset=='200'),'value'],dat_diff_melt.loc[(dat_diff_melt.mapping_type=='singleton')&(dat_diff_melt.offset=='200'),'value']).pvalue)
add_p_val(plt,2.8,3.2,upper_lim+0.2,0.02,mannwhitneyu(dat_diff_melt.loc[(dat_diff_melt.mapping_type=='pair')&(dat_diff_melt.offset=='300'),'value'],dat_diff_melt.loc[(dat_diff_melt.mapping_type=='singleton')&(dat_diff_melt.offset=='300'),'value']).pvalue)
add_p_val(plt,3.8,4.2,upper_lim+0.2,0.02,mannwhitneyu(dat_diff_melt.loc[(dat_diff_melt.mapping_type=='pair')&(dat_diff_melt.offset=='400'),'value'],dat_diff_melt.loc[(dat_diff_melt.mapping_type=='singleton')&(dat_diff_melt.offset=='400'),'value']).pvalue)
plt.savefig(f"plots/mock_all_SRR8359173_inward_vs_outward_diff.pdf")
plt.clf()