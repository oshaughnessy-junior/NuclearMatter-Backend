// Source-matched fixture. NOT compiled/executed in current environment.
// Compile against FULL pinned O2scl d68532ff31e59dac65844c3d387e1388cb71a770.
// Example (installation flags vary): c++ o2scl_fixture.cpp $(pkg-config --cflags --libs o2scl) -o fixture
#include <o2scl/eos_had_rmf.h>
#include <iostream>
#include <iomanip>
#include <cmath>
int main() {
  constexpr double hc=197.3269804, T=10./hc;
  o2scl::eos_had_rmf eos;
  // Individual masses are a recorded uniform-matter gauge choice: only
  // (g_i/m_i)^2 enters this model with all extra interactions zero.
  eos.ms=550./hc; eos.mw=783./hc; eos.mr=770./hc;
  eos.cs=std::sqrt(11.785); eos.cw=std::sqrt(7.148); eos.cr=std::sqrt(4.410);
  eos.b=.002947; eos.c=-.001070; eos.mnuc=939./hc; eos.zm_mode=false;
  eos.zeta=eos.xi=eos.a1=eos.a2=eos.a3=eos.a4=eos.a5=eos.a6=eos.b1=eos.b2=eos.b3=0.;
  o2scl::fermion n(939./hc,2.),p(939./hc,2.);
  n.inc_rest_mass=p.inc_rest_mass=true;
  n.non_interacting=p.non_interacting=false;
  n.n=.144; p.n=.016;
  // No internal converged fields supplied. O2scl creates its own continuation.
  o2scl::thermo th;
  int status=eos.calc_temp_e(n,p,T,th);
  double sig,ome,rho; eos.get_fields(sig,ome,rho);
  double f1,f2,f3;
  eos.calc_eq_temp_p(n,p,T,sig,ome,rho,f1,f2,f3,th);
  std::cout<<std::setprecision(17);
  std::cout<<"status "<<status<<"\nnn "<<n.n<<"\nnp "<<p.n
    <<"\nPhi_MeV "<<eos.ms*eos.cs*sig*hc
    <<"\nW_MeV "<<eos.mw*eos.cw*ome*hc
    <<"\nR_MeV "<<eos.mr*eos.cr*rho*hc
    <<"\nmD_MeV "<<n.ms*hc<<"\nmu_n_MeV "<<n.mu*hc
    <<"\nmu_p_MeV "<<p.mu*hc<<"\ne_MeV_fm3 "<<th.ed*hc
    <<"\nP_MeV_fm3 "<<th.pr*hc<<"\ns_fm3 "<<th.en
    <<"\nf_MeV_fm3 "<<(th.ed-T*th.en)*hc
    <<"\nfield_residuals "<<f1<<" "<<f2<<" "<<f3<<"\n";
  return status;
}
