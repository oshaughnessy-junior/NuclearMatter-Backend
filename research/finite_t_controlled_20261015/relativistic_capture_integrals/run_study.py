"""Reproduce one selected point, avoiding an implicit broad expensive sweep."""
from __future__ import annotations
import argparse,json
import capture as C
from dilute_anchor import anchor
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--state',choices=['warm','dilute'],default='warm')
p.add_argument('--order',type=int,default=8)
p.add_argument('--angle-order',type=int,default=8)
p.add_argument('--inner-order',type=int,default=16)
p.add_argument('--phi-order',type=int,default=4)
p.add_argument('--tail',type=float,default=40.)
p.add_argument('--outer-tail',type=float,default=24.)
p.add_argument('--anchor',action='store_true')
a=p.parse_args()
s=C.EOS.matter_state(2.,1e-4,.29,.01)if a.state=='warm'else C.EOS.matter_state(.2,1e-15,.3,0.)
if a.anchor:
 if a.state!='dilute':p.error('Blocking-free anchor is qualified only for the dilute state')
 r=anchor(s.neutron,150.,C.K.V.MMU,s.proton.mass,radial_order=a.order,target_angle_order=a.angle_order,cm_angle_order=a.inner_order)
else:r=C.integrate(s,150.,lepton_order=a.order,angle_order=a.angle_order,energy_order=a.inner_order,phi_order=a.phi_order,tail=a.tail,outer_tail=a.outer_tail)
print(json.dumps(r,indent=2))
