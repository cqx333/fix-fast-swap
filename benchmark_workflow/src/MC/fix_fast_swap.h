#ifdef FIX_CLASS
// clang-format off
FixStyle(fast/swap,FixFastSwap);
// clang-format on
#else

#ifndef FIX_FAST_SWAP_H
#define FIX_FAST_SWAP_H

#include "fix.h"

#include <vector>

namespace LAMMPS_NS {

class NeighList;
class NeighRequest;
class RanPark;

class FixFastSwap : public Fix {
 public:
  FixFastSwap(class LAMMPS *, int, char **);
  ~FixFastSwap() override;

  int setmask() override;
  void init() override;
  void init_list(int, NeighList *) override;

  void post_force(int vflag) override;
  void post_force_respa(int vflag, int ilevel, int iloop) override;
  void pre_force(int vflag) override;
  void pre_force_respa(int vflag, int ilevel, int iloop) override;

  int pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc) override;
  void unpack_forward_comm(int n, int first, double *buf) override;

  double compute_vector(int i) override;

 private:
  bigint mc_cycle_;
  std::vector<int> candidate_bins_[8];
  bool mc_parallel_enabled;
  bool mc_parallel_checked;

  NeighRequest *request;
  NeighList *list_full;

  RanPark *random_equal;
  RanPark *random_unequal;

  int nevery;
  int ncycle;
  double beta;
  int seed;

  int nswaptypes;
  int *type_list;
  double *mu;

  int niswap_local;
  int niswap_global;
  int niswap_before;
  int *local_iatom_list;
  int nmax_local_list;

  int unequal_cutoffs;

  bigint n_attempts;
  bigint n_accepts;
  bigint n_attempts_global;
  bigint n_accepts_global;
  bigint last_global_reduce_step;

  void build_local_swap_list();
  void build_parallel_candidate_bins();
  int pick_i_swap_atom();
  int pick_i_swap_atom_serial();
  bool check_parallel_mc_feasible();
  int pick_i_swap_atom_parallel();
  int pick_new_type(int itype, int &jtype);
  double energy_local_atom(int i);
  int attempt_one_flip();
  double energy_full();
};

}    // namespace LAMMPS_NS

#endif
#endif
