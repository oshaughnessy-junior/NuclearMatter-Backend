      program cold_mu_mirror_driver
      implicit real*8 (a-h,k-z)
      integer iid,ios
      parameter (isize=10000)
      dimension tcn(0:isize),tcp(0:isize)
      dimension mstp(0:isize),mstn(0:isize)
      dimension kfe(0:isize),kfm(0:isize),kfp(0:isize),kfn(0:isize)
      do
       read(*,*,iostat=ios) iid,i,t,kfe(i),kfm(i),kfn(i),kfp(i),
     1      mstn(i),mstp(i),tcn(i),tcp(i),isf
       if (ios.ne.0) exit
       call numurca_nucl(i,t,q,tcn,tcp,isf,mstn,mstp,
     1      kfe,kfm,kfn,kfp)
       write(*,'(i6,1x,es26.17e3)') iid,q
      end do
      end
