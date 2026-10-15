      program cold_mu_driver
      implicit real*8 (a-h,k-z)
      integer iid,ios
      include 'size.inc.f'
      include 'rho_limits.inc.f'
      include 'profile_star.inc.f'
      include 'profile_comp.inc.f'
      include 'pairing.inc.f'
      include 'fermi.inc.f'
      include 'control_nu.inc.f'
      do
       read(*,*,iostat=ios) iid,i,t,kfe(i),kfm(i),kfn(i),kfp(i),
     1      mstn(i),mstp(i),tcn(i),tcp(i),isf
       if (ios.ne.0) exit
       call numurca_nucl(i,t,q)
       write(*,'(i6,1x,es26.17e3)') iid,q
      end do
      end
