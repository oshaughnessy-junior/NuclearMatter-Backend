c *********************************************************************
c nmbackend: isothermal-star scan.  build_nscool.sh appends this to the
c part of NSCool.f that reads the inputs and builds the star (everything
c before INCLUDE 'NSCool_OPEN.inc.f'), giving NSCool_iso.out.  In place
c of the time loop it sets the redshifted temperature to a uniform
c value Ttil in every zone and sums, with the calls and weights of
c NSCool's own loop:
c   C     = sum cv(i) dV(i)                      [erg/K]
c   Lnu   = sum e^(2phi) qnu(i) dV(i)            [erg/s, at infinity]
c   Lheat = sum e^(2phi) heat(i) dV(i)
c   Lgam  = 4 pi R^2 sigma Teff^4 e^(2phi_s), Teff = fteff(Ttil/e^phi_s)
c so that C dTtil/dt = -(Lnu - Lheat + Lgam) is NSCool's energy
c equation summed over zones.  Reads from stdin, after the master
c file name: ntil, Ttil_min, Ttil_max (log-spaced grid, K), rho_b.
c Tin: for T_b = Ttil at zone imax, the redshifted temperature at the
c first zone denser than rho_b (g/cm3), from NSCool's flux equation
c nlum + fp a2ephin dtemp = 0 with nlum = Lgam held fixed (no neutrino
c losses or heat capacity in the layer).  rho_b <= rhosurf: Tin = Ttil.
c *********************************************************************
      read(5,*) jn, tt_min, tt_max, rho_b
      time=0.d0
      istep=1
      call accretion_rate(time+dtime,dtime,m_dot)
      write(6,'(a)') 'nmbackend isothermal scan'
      write(6,'(a,1p3e17.8e3)') 'star M R e^phi_s =',
     1      emas(imax),rad(imax),ephi(imax)
      write(6,'(a,1p1e17.8e3)') 'eta used by fteff =',eta
      write(6,'(a)') 'columns: Ttil C Lnu Lheat Lgam Teff_inf'//
     1  ' Lnu_durca Lnu_murca_brem Lnu_pbf Lnu_crust C_core Tin'
      do jt=1,jn
       tt=tt_min*(tt_max/tt_min)**(dfloat(jt-1)/dfloat(max(1,jn-1)))
       c_tot=0.d0
       c_core=0.d0
       l_nu=0.d0
       l_h=0.d0
       l_du=0.d0
       l_mb=0.d0
       l_pbf=0.d0
       l_cr=0.d0
       do i=1,imax,2
        t=tt/ephi(i)
        d=rrho(i)
        a=a_cell(i)
        a1=a_ion(i)
        z=z_ion(i)
        call neutrino(i,t,d,a,z,qnu(i),
     1    qeebrem(i),qnpb(i),qplasma(i),qsynch(i),qbubble(i),
     1    qpair(i),qphoto(i),qbrem_nn(i),
     2    qmurca_nucl(i),qbrem_nucl(i),qmurca_hyp(i),qbrem_hyp(i),
     3    qdurca_np(i),qdurca_lap(i),
     3    qdurca_smn(i),qdurca_smla(i),qdurca_sms0(i),
     4    qfast(i),
     5    qdurca_q(i),qmurca_q(i),
     6    qpbf_n1s0(i),qpbf_n3p2(i),qpbf_p1s0(i),qpbf_q(i),
     7    debug,naa)
        call heating(i,time,dtime,t,d,a,a1,z,tcp(i),
     1               m_dot/ephi(i),i_eleconb,
     2               ephi(i),dephi(i),heat(i))
        call specheat(i,t,d,a,z,cv(i),
     1       cv_n(i),cv_p(i),cv_e(i),cv_m(i),
     2       cv_l(i),cv_sm(i),cv_s0(i),cv_sp(i),
     3       cv_q(i),cv_ion(i))
        dv=dvol(i)+dvol(i+1)
        w=e2phi(i)*dv
        c_tot=c_tot+cv(i)*dv
        l_nu=l_nu+qnu(i)*w
        l_h=l_h+heat(i)*w
        l_du=l_du+(qdurca_np(i)+qdurca_lap(i)+qdurca_smn(i)
     1            +qdurca_smla(i)+qdurca_sms0(i)+qdurca_q(i))*w
        l_mb=l_mb+(qmurca_nucl(i)+qbrem_nucl(i)+qmurca_hyp(i)
     1            +qbrem_hyp(i)+qmurca_q(i))*w
        l_pbf=l_pbf+(qpbf_n1s0(i)+qpbf_n3p2(i)+qpbf_p1s0(i)
     1             +qpbf_q(i))*w
        if (i.le.icore) then
         c_core=c_core+cv(i)*dv
        else
         l_cr=l_cr+qnu(i)*w
        end if
       end do
c      Outer boundary as in the time loop: T_b = Ttil at zone imax.
c      NSCool's printed Teff (teffective) uses zone imax-1 for the
c      redshift; reproduce it from nlum(imax-1) = coeff*teff^4.
       teff=fteff(tt/ephi(imax),ifteff,eta,bf_r(imax),istep,
     1            time,ts1,ts2,z_ion(imax),a_ion(imax),rrho(imax),
     2            debug)
       l_g=4.d0*pi*radius**2*5.67d-5*teff**4*e2phi(imax)
       te_inf=(l_g/e2phi(imax-1)/(4.d0*pi*radius**2*5.67d-5))**.25d0
     1        *ephi(imax-1)
c      March the flux equation inward from zone imax to rho_b.
       xl=l_g/lsol
       iz1=imax
       tcur=tt
       fpout=fpzone(iz1,tcur,ephi)
       do while ((iz1.gt.1).and.(rrho(iz1).lt.rho_b))
        iz0=iz1-2
        ie=iz1-1
        xt0=tcur
        do jit=1,20
         fpin=fpzone(iz0,xt0,ephi)
         xt1=tcur+xl*(debar(ie)+debar(ie+1))/
     1       (.5d0*(fpin+fpout)*a2ephin(ie))
         if (abs(xt1-xt0).lt.1.d-10*xt1) goto 100
         xt0=xt1
        end do
 100    tcur=xt1
        fpout=fpzone(iz0,tcur,ephi)
        iz1=iz0
       end do
       write(6,'(a4,1p12e17.8e3)') 'ISO ',tt,c_tot,l_nu,l_h,l_g,te_inf,
     1       l_du,l_mb,l_pbf,l_cr,c_core,tcur
      end do
c     Optional per-zone dump (for the JAX physics tests): a line with
c     the number of temperatures, then one redshifted T per line.
c     ZSC: scalars; ZST: T-independent zone arrays (0 to imax+1);
c     ZON: every term NSCool sums, at odd (temperature) zones.
      jnd=0
      read(5,*,end=200) jnd
 200  continue
      if (jnd.gt.0) then
       write(6,'(a4,4i7,1p6e17.8e3)') 'ZSC ',imax,icore,idrip,isf,
     1   rhocore,rhodrip,gs14,eta,qimp,radius
       do i=0,imax+1
        write(6,'(a4,2i6,1p26e17.8e3)') 'ZST ',i,idurca_np(i),
     1   rad(i),rrho(i),pres(i),bar(i),debar(i),dvol(i),ephi(i),
     2   a2ephin(i),emas(i),yelect(i),ymuon(i),yneutr(i),yprot(i),
     3   kfe(i),kfm(i),kfn(i),kfp(i),mstn(i),mstp(i),tcn(i),tcp(i),
     4   a_cell(i),a_ion(i),z_ion(i),v_ion(i),fhad(i)
       end do
      end if
      do jd=1,jnd
       read(5,*) tt
       do i=1,imax,2
        t=tt/ephi(i)
        d=rrho(i)
        call neutrino(i,t,d,a_cell(i),z_ion(i),qnu(i),
     1    qeebrem(i),qnpb(i),qplasma(i),qsynch(i),qbubble(i),
     1    qpair(i),qphoto(i),qbrem_nn(i),
     2    qmurca_nucl(i),qbrem_nucl(i),qmurca_hyp(i),qbrem_hyp(i),
     3    qdurca_np(i),qdurca_lap(i),
     3    qdurca_smn(i),qdurca_smla(i),qdurca_sms0(i),
     4    qfast(i),
     5    qdurca_q(i),qmurca_q(i),
     6    qpbf_n1s0(i),qpbf_n3p2(i),qpbf_p1s0(i),qpbf_q(i),
     7    debug,naa)
        call specheat(i,t,d,a_cell(i),z_ion(i),cv(i),
     1       cv_n(i),cv_p(i),cv_e(i),cv_m(i),
     2       cv_l(i),cv_sm(i),cv_s0(i),cv_sp(i),
     3       cv_q(i),cv_ion(i))
        call conduct(i,t,d,a_cell(i),a_ion(i),z_ion(i),qimp,
     1       nbfield2(i),sigx,xlam,0.d0,nues,nuel)
        write(6,'(a4,2i6,1p39e17.8e3)') 'ZON ',jd,i,tt,qnu(i),
     1    qeebrem(i),qnpb(i),qplasma(i),qsynch(i),qbubble(i),
     1    qpair(i),qphoto(i),qbrem_nn(i),
     2    qmurca_nucl(i),qbrem_nucl(i),qmurca_hyp(i),qbrem_hyp(i),
     3    qdurca_np(i),qdurca_lap(i),
     3    qdurca_smn(i),qdurca_smla(i),qdurca_sms0(i),
     4    qfast(i),qdurca_q(i),qmurca_q(i),
     6    qpbf_n1s0(i),qpbf_n3p2(i),qpbf_p1s0(i),qpbf_q(i),
     7    cv(i),cv_n(i),cv_p(i),cv_e(i),cv_m(i),
     8    cv_l(i),cv_sm(i),cv_s0(i),cv_sp(i),cv_q(i),cv_ion(i),
     9    xlam,fpzone(i,tt,ephi)
       end do
      end do
      print *,'Done !'
      stop
 9997 continue
      print *,'nmbackend: cannot read the master file'
      stop 1
      end
c *********************************************************************
c fp of NSCool's time loop at zone i, redshifted temperature ttil.
      function fpzone(i,ttil,ephi)
      implicit real*8 (a-h,k-z)
      parameter (c=2.99792d10,lsol=3.826d33)
      INCLUDE 'size.inc.f'
      INCLUDE 'profile_star.inc.f'
      INCLUDE 'profile_comp.inc.f'
      INCLUDE 'mag_field.inc.f'
      dimension ephi(0:isize)
      t=ttil/ephi(i)
      call conduct(i,t,rrho(i),a_cell(i),a_ion(i),z_ion(i),qimp,
     1             nbfield2(i),sig,lam,0.d0,nu_e_s,nu_e_l)
      call opacity(t,rrho(i),a_cell(i),z_ion(i),kap)
      acd=7.56d-15*c/(3.d0*kap*rrho(i))
      fpzone=(lam+4.d0*acd*t**3)*bar(i)/lsol
      end
