      program pbf_review
      implicit real*8 (a-h,k-z)
      integer ident
      common/gapscale/sxp1s0,sxn3p2,ca3p2
      sxp1s0=1.d0
      sxn3p2=1.d0
      ca3p2=.5d0
      do
       read(*,*,iostat=ios) ident,t,tc,mst,kf
       if (ios.ne.0) exit
       call nu_1s0_pbf(t,tc,mst,kf,qa)
       call nu_n3p2_B_pbf(t,tc,mst,kf,qb)
       write(*,'(i4,2(1x,es26.17e3))') ident,qa,qb
      end do
      end
