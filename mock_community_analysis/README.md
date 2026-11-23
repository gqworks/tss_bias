# Retired

This folder is not the rerun path. The mock-community analyses live in `tssbias`.

```bash
python -m tssbias download data/manifests/mock.tsv
python -m tssbias mock
python -m tssbias five-prime --cohort mock
python -m tssbias mock --hifi
python -m tssbias download data/manifests/matched_longread.tsv
python -m tssbias matched-longread
```

`mock` is the ATCC MSA-1003 exact-strain control. `five-prime --cohort mock` is the 5′ check. `mock --hifi` writes a 100,000-read subset of `SRR11606871` to `{scratch_root}/mock/hifi/SRR11606871_100k.fa` and runs Prodigal, Bowtie2, and minimap2 on that file only. `matched-longread` is the gut same-sample test.
