      program independent_pbf
      implicit real*8 (a-h,k-z)
      common/gapscale/sxp1s0,sxn3p2,ca3p2
      sxp1s0=1.d0
      sxn3p2=1.d0
      do
        read(*,*,iostat=ios) t,tc,mst,kf,ca3p2
        if (ios.ne.0) exit
        call nu_1s0_pbf(t,tc,mst,kf,qa)
        call nu_n3p2_B_pbf(t,tc,mst,kf,qb)
        write(*,'(2(es26.17e3,1x))') qa,qb
      end do
      end
