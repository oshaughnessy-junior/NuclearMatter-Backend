"""Independent Strumia/Vissani Eq6 real-form-factor polynomial, vacuum only."""
import current as C

def amplitude(channel,p,ff):
    v,a,f,gp=ff;g1,g2=-a,-gp
    M=(channel.initial_mass+channel.final_mass)/2
    # Their Delta stays mn-mp positive on physical charge-conjugate channels.
    # Generalized algebraic crossing keeps this signed Delta with B reversal.
    delta=(channel.final_mass-channel.initial_mass)*(1 if channel.antineutrino else -1)
    m2=channel.lepton_mass**2;M2=M*M
    t=C.V.dot(p['k']-p['l'],p['k']-p['l'])
    u=C.V.dot(p['k']-p['pb'],p['k']-p['pb']);su=p['s']-u
    A=(t-m2)*(4*v*v*(4*M2+t+m2)+4*g1*g1*(-4*M2+t+m2)+f*f*(t*t/M2+4*t+4*m2)+4*m2*t*g2*g2/M2+8*v*f*(2*t+m2)+16*m2*g1*g2)
    A-=delta**2*((4*v*v+t*f*f/M2)*(4*M2+t-m2)+4*g1*g1*(4*M2-t+m2)+4*m2*g2*g2*(t-m2)/M2+8*v*f*(2*t-m2)+16*m2*g1*g2)
    A-=32*m2*M*delta*g1*(v+f)
    B=16*t*g1*(v+f)+4*m2*delta*(f*f+v*f+2*g1*g2)/M
    D=4*(v*v+g1*g1)-t*f*f/M2
    return 8*C.V.GF**2*C.V.VUD**2*(A+(-1 if channel.antineutrino else 1)*su*B+su*su*D)/16
