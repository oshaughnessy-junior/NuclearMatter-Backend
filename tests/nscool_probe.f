c Test driver: calls NSCool routines that need no zone arrays.  Built by
c tests/test_cooling_physics_*.py against the objects of the NSCool build;
c run in the NSCool tree (the routines read Code/Data_Files).
c Each stdin line: a key and four numbers; prints one number.
      program probe
      implicit real*8 (a-h,k-z)
      character*8 what
 10   read(5,*,end=99) what, a, b, c, d
      q=0.d0
      if (what.eq.'r11') q=func_r_1s0_1s0(a,b)
      if (what.eq.'r13') q=func_r_1s0_3p2B(a,b)
      if (what.eq.'pbf1') call nu_1s0_pbf(a,b,c,d,q)
      if (what.eq.'pbf3') call nu_n3p2_B_pbf(a,b,c,d,q)
      if (what.eq.'cvion') call cvion(a,b,c,d,q)
      if (what.eq.'plasma') call nplasma(a,b,c,d,q)
      if (what.eq.'npb') call npb_new(a,b,q)
      if (what.eq.'eebrem') call neebrem(a,b,q)
      write(6,'(1p,e25.16e3)') q
      goto 10
 99   end
      block data probedata
      implicit real*8 (a-h,k-z)
      common/gapscale/sxp1s0,sxn3p2,ca3p2
      data ca3p2/0.5d0/
      end
