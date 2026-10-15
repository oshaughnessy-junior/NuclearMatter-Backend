from exchange import *
import json
P=Path(__file__).resolve().parent
label=sys.argv[1];method=sys.argv[2];order=int(sys.argv[3]);inner=int(sys.argv[4]);angle=int(sys.argv[5]);cut=float(sys.argv[6])
if label=='hot':rad=Radiation(3.,4.)
elif label=='cold':rad=Radiation(1.5,2.)
elif label=='packet':rad=Radiation(3.,4.,True)
elif label=='LTE':rad=Radiation(2.,4.)
else:raise ValueError('Unknown physical case')
sp=nc.state_density(2.,.002,0.)
result=integrate(sp,rad,method=method,energy_order=order,inner_order=inner,angle_order=angle,cutoff=cut)
name=f'{label}_{method}_{order}_{inner}_{angle}_{cut:g}.json';(P/name).write_text(json.dumps(result,indent=2)+'\n');print(name,'runtime',result['runtime_seconds'],'values',result['values'])
