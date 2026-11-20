      subroutine nu_1s0_pbf(T,Tc,mst,kf,q_1s0_pbf)
c     This subroutine uses only the Axial part: good for n & p !
c     From : Neutrino emission due to proton pairing in neutron stars
c            Kaminker, A. D.; Haensel, P.; Yakovlev, D. G.
c            1999A&A...345L..14K
c     Vector part is put to zero according to:
c     "Vector current conservation and neutrino emission from 
c     singlet-paired baryons in neutron stars"
c     Leinson, L.B., & Perez, A.
c     2006PhLB..638..114L
       implicit real*8 (a-h,k-z)
        if (T.le.Tc) then
         pf=kf*197.d0
         vf=pf/mst/940.d0
         a_v=0.d0
         a_a=1.60d0*vf**2*(mst**2+11.d0/42.d0)
         a=a_v+a_a
c        The vector part in "a" has been put to zero !
         tau=T/Tc
         u=dsqrt(1.d0-tau)*(1.456d0-0.157d0/dsqrt(tau)+1.764d0/tau)
         q_1s0_pbf=1.170d21*mst**2*vf*(T/1.d9)**7*3.d0*a*
     1              control_pbf_1S0(u)
        else
         q_1s0_pbf=0.0d0
        end if
       return
      end
c *********************************************************************
c *********************************************************************
      subroutine nu_n3p2_B_pbf(T,Tc,mst,kf,q_n3p2_pbf)
c     This subroutine uses only the Axial part from
c     Neutrino emission due to Cooper pairing of nucleons in cooling neutron stars
c     Yakovlev, D. G.; Kaminker, A. D.; Levenfish, K. P.
c     1999A&A...343..650Y
c     Vector part is put to zero according to:
c     Neutrino emission due to Cooper pairing in neutron stars
c     Leinson, L.B., & Perez, A.
c     2006astro.ph..6653L
c
c     For the j=2 m=0 gap
       implicit real*8 (a-h,k-z)
       parameter (pi = 3.14159265d0,rhonucl=2.8d14)
c nmbackend patch: a_A = c_A g_A^2 with c_A from the pairing file
c (default 0.5: Leinson 2010, as in A. W. Steiner's fork; release: 2)
       common/gapscale/sxp1s0,sxn3p2,ca3p2
        if (T.le.Tc) then
         pf=kf*197.d0
         vf=pf/mst/940.d0
         g_A=1.26d0
         a_v=0.d0
         a_a=ca3p2*g_A**2
         a=a_v+a_a
c        The vector part in "a" has been put to zero !
         tau=T/Tc
         u=dsqrt(1.d0-tau)*(0.7893d0+1.764d0/tau)
         q_n3p2_pbf=1.170d21*mst**2*vf*(T/1.d9)**7*3.d0*a*
     1              control_pbf_3P2_B(u)
        else
         q_n3p2_pbf=0.0d0
        end if
       return
      end
c *********************************************************************
c *********************************************************************
      function control_pbf_1S0(v)
       implicit real*8 (a-h,k-z)
        x=0.602d0*v**2+0.5942d0*v**4+0.288d0*v**6
        y=dsqrt(0.5547d0+dsqrt(0.4453d0**2+0.01130*v**2))
        z=dexp(-dsqrt(4.d0*v**2+2.245d0**2)+2.245d0)
        control_pbf_1S0=x*y*z
       return
      end
c *********************************************************************
c *********************************************************************
      function control_pbf_3P2_B(v)
       implicit real*8 (a-h,k-z)
        x=(1.204d0*v**2+3.733d0*v**4+0.3191*v**6)/(1.d0+0.3511*v**2)
        y=(0.7591+dsqrt(0.2409d0**2+0.3145d0*v**2))**2
        z=dexp(-dsqrt(4.d0*v**2+0.4616d0**2)+0.4616d0)
        control_pbf_3P2_B=x*y*z
       return
      end
c *********************************************************************
c *********************************************************************
