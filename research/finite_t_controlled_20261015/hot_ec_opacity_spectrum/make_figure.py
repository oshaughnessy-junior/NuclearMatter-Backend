"""Detailed selected free EC spectral coefficients and emitted moment density."""
from pathlib import Path
import json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent;rows=[r for i in[0,1]for r in json.loads((P/f'total_fine_part{i}.json').read_text())['rows']];E=np.array([r['E']for r in rows]);j=np.array([r['spectrum']['j']['value']for r in rows]);chi=np.array([r['spectrum']['chi']['value']for r in rows]);rho=E*E/(2*math.pi**2*197.3269804**3);sel=json.loads((P/'results.json').read_text())['rows'];fig,ax=plt.subplots(1,2,figsize=(9.2,3.6),layout='constrained');ax[0].semilogy(E,j,label='j: EC emissivity per occupation');ax[0].semilogy(E,chi,label='χ: true νe absorption');ax[0].semilogy(E,j+chi,ls='--',label='j+χ: fixed-bath relaxation');ax[0].scatter([r['Enu']for r in sel],[r['rules'][-1]['j']['value']for r in sel],s=17,color='C0');ax[0].set(xlabel='Neutrino energy [MeV]',ylabel='Coefficient [s⁻¹]',title='Separate direct FD occupations');ax[0].legend(fontsize=8)
ax[1].plot(E,rho*j*1e8,label=r'$\rho_Ej$: number / dE');ax[1].plot(E,E*rho*j*1e8,label=r'$E\rho_Ej$: energy / dE');ax[1].set(xlabel='Neutrino energy [MeV]',ylabel='Moment density ×10⁸\n(number: fm⁻³ s⁻¹ MeV⁻¹; energy: fm⁻³ s⁻¹)',title='Empty-radiation emission, one helicity');ax[1].legend(fontsize=8);fig.suptitle('Free p+e⁻ ↔ n+νe, fixed T=2 MeV / nB=10⁻⁷ fm⁻³ / Ye=0.3',fontsize=11);fig.savefig(P/'spectral_coefficients.png',dpi=180);fig.savefig(P/'spectral_coefficients.pdf')
