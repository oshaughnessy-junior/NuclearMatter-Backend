c Scalar driver of the built NSCool core lepton conductivity.
      program probe
      implicit real*8(a-h,k-z)
 10   read(5,*,end=99) t,ke,km,kp,mp,tcp
      call con_core_lep(t,ke,km,kp,mp,tcp,
     1                 0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,
     2                 sigma,le,lm,0.d0,nus,nul)
      write(6,'(1p,e25.16e3)') le+lm
      goto 10
 99   end
