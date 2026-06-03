"""RIFT integration: EOS-sequence I/O (robust NSSequence + legacy EOSSequenceLandry).

NOTE (logged for later improvement): the accessors here currently lock to a SINGLE
stable branch (the rising branch up to M_max).  Because the low-level sequence
parameter is the central pseudo-enthalpy h_c (monotone), the better long-term framing
is to parameterize the inference *directly* along h_c (a path integral in enthalpy),
which handles multiple / disconnected stable branches (twin stars) natively instead
of reducing to a single Lambda(m_ref) ordering.  The NSSequence format already stores
everything vs h_c, so it is the substrate for that.  See docs/backlog.md.
"""
from .sequence import (write_nss, NSSequence, write_landry_legacy,
                       open_sequence, SequenceRepresentation,
                       NSS_FIELDS, NSS_UNITS)
from .compressed import (write_pca, write_pca_from_nss, PCASequence,
                         fit_sequence_pca, compression_ratio, PCA_FIELDS)

from .evidence import (CIPTabularEvidence, posterior_mass_lnZ,
                       read_eos_indices_xml)

__all__ = ["write_nss", "NSSequence", "write_landry_legacy",
           "open_sequence", "SequenceRepresentation",
           "NSS_FIELDS", "NSS_UNITS",
           "write_pca", "write_pca_from_nss", "PCASequence",
           "fit_sequence_pca", "compression_ratio", "PCA_FIELDS",
           "CIPTabularEvidence", "posterior_mass_lnZ", "read_eos_indices_xml"]
