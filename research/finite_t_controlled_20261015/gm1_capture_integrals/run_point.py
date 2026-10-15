"""Reproduce one selected point with explicitly chosen numerical orders."""
import argparse,json
import capture as C
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--flavor',choices=['e','mu'],required=True)
p.add_argument('--order',type=int,required=True,help='Requires separate convergence evidence; no universally qualified default')
p.add_argument('--angle-order',type=int)
p.add_argument('--inner-order',type=int)
p.add_argument('--inner-tail',type=float,default=60.)
p.add_argument('--outer-tail',type=float,default=32.)
p.add_argument('--coordinate',choices=['cos','q'],default='cos')
a=p.parse_args();matter=C.A.matter_state(8.,.22,.17,.04)
r=C.integrate(matter,40.,a.flavor,lepton_order=a.order,angle_order=a.angle_order or a.order,inner_order=a.inner_order or 2*a.order,inner_tail=a.inner_tail,outer_tail=a.outer_tail,coordinate=a.coordinate)
print(json.dumps(r,indent=2))
