# GW170817 EOS inference using a NuclearMatter-Backend EOS sequence (tabular method).
#
# Mirrors the reference flow in
#   LIGO-ScienceMode/O4_era/rift-eos-inference-tabular-method/analysis_generation/
#   tabular_EOS_analysis/Makefile  (target rundir_GW170817_tabular)
# but builds the tabular EOS sequence with our differentiable backend instead of
# consuming a precomputed LCEHL posterior-sample file.
#
# The whole point (per RO): the per-EOS evidence is NOT stored -- it is regenerated
# by running CIP through the hyperpipeline with a fixed EOS sequence, given all.net.
# Here we (1) generate the sequence file, then (2) hand it to the pipeline exactly
# as --internal-tabular-eos-file.  CIP auto-detects our NSSequence format via
# EOSManager.EOSSequenceFromFile (falls back to the legacy Landry file if you prefer).

# --- paths (override on the command line as needed) ------------------------------
NMB        ?= $(abspath ../..)
REF        ?= /Users/rossma/unixhome/Projects/LIGO-ScienceMode/O4_era/rift-eos-inference-tabular-method/analysis_generation/tabular_EOS_analysis
INI        ?= $(REF)/GW170817.ini
COINC      ?= $(REF)/coinc.xml
CACHE      ?= $(REF)/local.cache
PSD_DIR    ?= $(REF)/psds
GRID_SIZE  ?= 20000

# --- EOS sequence model knobs ----------------------------------------------------
# A modest metamodel grid over symmetry-energy slope and high-density stiffness.
GRID       ?= L_sym=40,50,60,70;cs2_tail=0.30,0.40,0.50,0.60
NPTS       ?= 120
SEQ_NSS    ?= $(CURDIR)/nmb_eos_sequence.nss.h5
SEQ_LEGACY ?= $(CURDIR)/nmb_eos_sequence.landry.h5

PY         ?= PYTHONPATH=$(NMB)/src python

# 1. Build the EOS sequence (both NSS-native and legacy Landry formats).
$(SEQ_NSS) $(SEQ_LEGACY): $(NMB)/scripts/build_eos_sequence.py
	$(PY) $(NMB)/scripts/build_eos_sequence.py \
	    --model metamodel --grid "$(GRID)" --n-points $(NPTS) \
	    --out $(SEQ_NSS) --legacy $(SEQ_LEGACY)

sequence: $(SEQ_NSS)   ## just (re)build the EOS sequence file

# 2. Run the GW170817 tabular-EOS pipeline against our NSS sequence.
#    CIP auto-detects the format; pass SEQ=$(SEQ_LEGACY) to use the legacy file.
SEQ ?= $(SEQ_NSS)
rundir_GW170817_nmb: $(SEQ)
	util_RIFT_pseudo_pipe.py --assume-matter \
	    --internal-tabular-eos-file $(SEQ) \
	    --use-ini $(INI) --use-rundir $(CURDIR)/$@ --use-coinc $(COINC) \
	    --fake-data-cache $(CACHE) --force-initial-grid-size $(GRID_SIZE)
	cp $(PSD_DIR)/*psd.xml.gz $@/

# Same, but using the legacy EOSSequenceLandry file (proves drop-in equivalence).
rundir_GW170817_nmb_legacy: $(SEQ_LEGACY)
	util_RIFT_pseudo_pipe.py --assume-matter \
	    --internal-tabular-eos-file $(SEQ_LEGACY) \
	    --use-ini $(INI) --use-rundir $(CURDIR)/$@ --use-coinc $(COINC) \
	    --fake-data-cache $(CACHE) --force-initial-grid-size $(GRID_SIZE)
	cp $(PSD_DIR)/*psd.xml.gz $@/

clean:
	rm -f $(SEQ_NSS) $(SEQ_LEGACY)

.PHONY: sequence rundir_GW170817_nmb rundir_GW170817_nmb_legacy clean
