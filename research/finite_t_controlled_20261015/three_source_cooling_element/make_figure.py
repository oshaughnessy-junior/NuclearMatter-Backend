"""Discrete short element trajectory and reservoir energy accounting."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
P=Path(__file__).parent;x=json.loads((P/'results.json').read_text());tr=x['rows'][-1];states=[tr['initial']]+[r['final_state']for r in tr['records']];t=np.linspace(0,tr['duration']*1e3,len(states));escaped=[0.];Q=0.
for rec in tr['records']:
 dt=tr['duration']/tr['steps'];Q+=dt*(rec['source']['Q']+rec['second_source']['Q'])/2;escaped.append(Q)
fig,ax=plt.subplots(1,3,figsize=(11.7,3.5),layout='constrained');ax[0].plot(t,[s['T']for s in states],'-o',ms=4);ax[0].set(xlabel='Time [ms]',ylabel='Temperature [MeV]',title='Actual coupled T(u, Ye)');ax[0].ticklabel_format(axis='y',useOffset=False,style='plain');ax[1].plot(t,[s['Ye']for s in states],'-o',ms=4,color='C1');ax[1].set(xlabel='Time [ms]',ylabel='Ye',title='Three physical CC sources');ax[1].ticklabel_format(axis='y',useOffset=False,style='plain');e0=states[0]['energy_density'];g0=states[0]['photon']['energy_density'];ax[2].plot(t,np.array(escaped)*1e8,'o-',ms=4,label='Escaped νe+anti-νe');ax[2].plot(t,[(e0-s['energy_density'])*1e8 for s in states],'x--',label='Total reservoir loss');ax[2].plot(t,[(g0-s['photon']['energy_density'])*1e8 for s in states],'-s',ms=3,label='Photon subset loss');ax[2].set(xlabel='Time [ms]',ylabel='Energy density [10⁻⁸ MeV fm⁻³]',title='No thermostat or photon escape');ax[2].legend(fontsize=8);fig.suptitle('Fixed nB=10⁻⁷ fm⁻³, trapped LTE photons and equilibrium e± — Heun 4 discrete steps',fontsize=11);fig.savefig(P/'cooling_element.png',dpi=180);fig.savefig(P/'cooling_element.pdf')
