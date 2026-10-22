/* ----------------------------------------------------------------------
   Custom Fix: fast/swap
   - Pairwise potentials only (temporary restriction): if a many-body pair style
     is detected, this fix will abort with a clear error message.
   - Localized energy-difference evaluation using a FULL neighbor list.
   - Functionality intended to match the original implementation semantics.
------------------------------------------------------------------------- */

#include "fix_fast_swap.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "lmptype.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair.h"
#include "random_park.h"
#include "update.h"

#include "compute.h"
#include "kspace.h"

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>

#ifndef LAMMPS_FORCE_INLINE
#define LAMMPS_FORCE_INLINE inline
#endif

using namespace LAMMPS_NS;
using namespace FixConst;

namespace {
struct SelfCheckConfig {
  bool enabled = false;
  int period = 0;
  double tolerance = 1.0e-10;
  bigint attempts = 0;
  double max_abs_diff = 0.0;
};

std::unordered_map<FixFastSwap *, SelfCheckConfig> self_check_registry;

SelfCheckConfig &self_check_for(FixFastSwap *fix)
{
  return self_check_registry[fix];
}
}    // namespace

/* ---------------------------------------------------------------------- */

FixFastSwap::FixFastSwap(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), request(nullptr), list_full(nullptr), random_equal(nullptr),
    random_unequal(nullptr), nevery(1), ncycle(0), beta(0.0), seed(12345), nswaptypes(0),
    type_list(nullptr), mu(nullptr), niswap_local(0), niswap_global(0), niswap_before(0),
    local_iatom_list(nullptr), nmax_local_list(0), unequal_cutoffs(0),
    mc_parallel_enabled(true), mc_parallel_checked(false)
{
  // Minimal syntax:
  // fix ID group fast/swap Nevery swap_fraction T seed types N t1 t2 ... tN mu m1 m2 ... mN
  //
  // Example:
  // fix 10 all fast/swap 1 0.2 1.0 12345 types 2 1 2 mu 0.0 0.5
  //
  // Here mu list corresponds to the listed swap types in the same order.
  // mu is stored as mu[atom_type].

  if (narg < 11) error->all(FLERR, "Illegal fix fast/swap command (too few args)");

  nevery = utils::inumeric(FLERR, arg[3], false, lmp);
  if (nevery <= 0) error->all(FLERR, "Fix fast/swap nevery must be > 0");

  /* atom/swap-compatible: fixed number of MC attempts per invocation */
  ncycle = utils::inumeric(FLERR, arg[4], false, lmp);
  if (ncycle <= 0) error->all(FLERR, "Fix fast/swap ncycle must be > 0");

  const double temperature = utils::numeric(FLERR, arg[5], false, lmp);
  if (temperature <= 0.0) error->all(FLERR, "Fix fast/swap requires T > 0");
  beta = 1.0 / (force->boltz * temperature);

  seed = utils::inumeric(FLERR, arg[6], false, lmp);
  if (seed <= 0) error->all(FLERR, "Fix fast/swap requires seed > 0");
  if (seed >= 2147483647)
    error->all(FLERR, "Fix fast/swap requires seed < 2147483647 for RanPark");

  int iarg = 7;

  // allocate mu for all atom types
  mu = new double[atom->ntypes + 1];
  for (int i = 0; i <= atom->ntypes; i++) mu[i] = 0.0;

  // parse "types"
  if (strcmp(arg[iarg], "types") != 0) error->all(FLERR, "Fix fast/swap expects keyword 'types'");
  iarg++;

  nswaptypes = utils::inumeric(FLERR, arg[iarg], false, lmp);
  if (nswaptypes < 2) error->all(FLERR, "Fix fast/swap requires at least 2 swap types");
  iarg++;

  type_list = new int[nswaptypes];
  for (int k = 0; k < nswaptypes; k++, iarg++) {
    if (iarg >= narg) error->all(FLERR, "Fix fast/swap: not enough type indices after 'types'");
    type_list[k] = utils::inumeric(FLERR, arg[iarg], false, lmp);
    if (type_list[k] < 1 || type_list[k] > atom->ntypes)
      error->all(FLERR, "Fix fast/swap: invalid atom type in type list");
  }

  // parse "mu"
  if (iarg >= narg || strcmp(arg[iarg], "mu") != 0)
    error->all(FLERR, "Fix fast/swap expects keyword 'mu' after types list");
  iarg++;

  for (int k = 0; k < nswaptypes; k++, iarg++) {
    if (iarg >= narg) error->all(FLERR, "Fix fast/swap: not enough mu values after 'mu'");
    const double muk = utils::numeric(FLERR, arg[iarg], false, lmp);
    mu[type_list[k]] = muk;
  }

  SelfCheckConfig &self_check = self_check_for(this);
  while (iarg < narg) {
    if (strcmp(arg[iarg], "check") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "Illegal fix fast/swap command: check requires yes/no");
      self_check.enabled = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
      if (self_check.enabled) {
        if (iarg + 1 >= narg)
          error->all(FLERR,
                     "Illegal fix fast/swap command: check yes requires period and tolerance");
        self_check.period = utils::inumeric(FLERR, arg[iarg], false, lmp);
        self_check.tolerance = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
        if (self_check.period <= 0)
          error->all(FLERR, "Fix fast/swap check period must be > 0");
        if (self_check.tolerance < 0.0)
          error->all(FLERR, "Fix fast/swap check tolerance must be >= 0");
        iarg += 2;
      } else {
        self_check.period = 0;
      }
    } else {
      error->all(FLERR, "Illegal fix fast/swap command: unknown keyword after mu");
    }
  }

  // RNGs
  mc_cycle_ = 0;
  random_equal = new RanPark(lmp, seed);
  // Distinct rank-local streams; preserve rank 0 and avoid seed overflow/zero.
  const int local_seed = static_cast<int>(
      (static_cast<long long>(seed) - 1 + static_cast<long long>(comm->me)) % 2147483646LL) + 1;
  random_unequal = new RanPark(lmp, local_seed);
  if (comm->me == 0)
    utils::logmesg(lmp,
                   "Fix fast/swap RNG revision 20261020: rank-local seeds; serial stream unchanged\n");

  // Communication: for this pairwise-only version we only need to forward atom types.
  // NOTE: If you later re-enable many-body support, these sizes must be revisited.
  comm_forward = 1;   // forward: type
  comm_reverse = 0;   // reverse: none (pairwise-only)

  // no thermo output yet
  scalar_flag = 0;
  vector_flag = 0;
  global_freq = 1;

  // allocate local list
  nmax_local_list = 0;
  local_iatom_list = nullptr;

  // --- thermo output setup ---
  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;

  // initialize counters
  n_attempts = 0;
  n_accepts = 0;
  n_attempts_global = 0;
  n_accepts_global = 0;
  last_global_reduce_step = -1;
}

/* ---------------------------------------------------------------------- */

FixFastSwap::~FixFastSwap()
{
  delete random_equal;
  delete random_unequal;
  delete[] type_list;
  delete[] mu;
  memory->destroy(local_iatom_list);
  self_check_registry.erase(this);
}

/* ---------------------------------------------------------------------- */


int FixFastSwap::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= PRE_FORCE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixFastSwap::init()
{
  // -------------------------------------------------------------------
  // Pairwise-only guard
  // -------------------------------------------------------------------
  if (force->pair && force->pair->manybody_flag) {
    error->all(FLERR,
               "Fix fast/swap (this version) does not support many-body pair styles. "
               "Please switch to a pairwise potential (e.g., lj/cut) or use a many-body-capable "
               "version of this fix.");
  }

  if (force->pair == nullptr) error->all(FLERR, "Fix fast/swap requires a pair style");
  if (!force->pair->single_enable)
    error->all(FLERR, "Fix fast/swap requires a pair style with pair->single() support");
  if (force->kspace)
    error->all(FLERR,
               "Fix fast/swap does not yet support KSpace long-range energy terms");
  if (force->pair->tail_flag)
    error->all(FLERR,
               "Fix fast/swap does not yet support pair_modify tail corrections");
  if (atom->molecular != Atom::ATOMIC)
    error->all(FLERR,
               "Fix fast/swap supports only atomic systems without bond, angle, dihedral, "
               "or improper energy terms");

  if (atom->q_flag) {
    double qmin_local = 1.0e300;
    double qmax_local = -1.0e300;
    int found_local = 0;
    for (int i = 0; i < atom->nlocal; i++) {
      int is_swap_type = 0;
      for (int k = 0; k < nswaptypes; k++) {
        if (atom->type[i] == type_list[k]) {
          is_swap_type = 1;
          break;
        }
      }
      if (!is_swap_type) continue;
      qmin_local = std::min(qmin_local, atom->q[i]);
      qmax_local = std::max(qmax_local, atom->q[i]);
      found_local = 1;
    }
    int found_global = 0;
    MPI_Allreduce(&found_local, &found_global, 1, MPI_INT, MPI_SUM, world);
    if (found_global) {
      double qmin_global = 0.0;
      double qmax_global = 0.0;
      MPI_Allreduce(&qmin_local, &qmin_global, 1, MPI_DOUBLE, MPI_MIN, world);
      MPI_Allreduce(&qmax_local, &qmax_global, 1, MPI_DOUBLE, MPI_MAX, world);
      if (qmax_global - qmin_global > 1.0e-12)
        error->all(FLERR,
                   "Fix fast/swap changes atom types only; all swappable charged atoms must "
                   "have identical charge");
    }
  }

  SelfCheckConfig &self_check = self_check_for(this);
  if (self_check.enabled) {
    if (comm->nprocs > 1 && comm->me == 0)
      error->warning(FLERR,
                     "Fix fast/swap check yes currently validates single-rank local energy "
                     "differences only; checks are skipped on multi-rank runs");
    if (comm->me == 0) {
      char msg[256];
      std::snprintf(msg, sizeof(msg),
                    "Fix fast/swap: self-check enabled every %d local attempts with tolerance %.6g\n",
                    self_check.period, self_check.tolerance);
      utils::logmesg(lmp, msg);
    }
  }

  // Detect unequal cutoffs among swap types and request a neighbor-list cutoff
  // that covers every possible old/new trial interaction.  Local energy
  // differences remain exact for finite-range pairwise potentials because
  // energy_local_atom() still applies each type pair's own cutsq test.
  unequal_cutoffs = 0;
  double max_swap_cutsq = 0.0;
  double **cutsq = force->pair->cutsq;
  for (int a = 0; a < nswaptypes; a++) {
    for (int b = 0; b < nswaptypes; b++) {
      for (int k = 1; k <= atom->ntypes; k++) {
        max_swap_cutsq = std::max(max_swap_cutsq, cutsq[type_list[a]][k]);
        max_swap_cutsq = std::max(max_swap_cutsq, cutsq[type_list[b]][k]);
        if (cutsq[type_list[a]][k] != cutsq[type_list[b]][k]) unequal_cutoffs = 1;
      }
    }
  }

  // Request a FULL neighbor list.  If trial type changes can increase the
  // pair cutoff, the list must cover the largest possible swap-type cutoff
  // before evaluating the trial energy difference.
  request = neighbor->add_request(this, NeighConst::REQ_FULL);
  if (unequal_cutoffs && max_swap_cutsq > 0.0) request->set_cutoff(std::sqrt(max_swap_cutsq));

  if (comm->me == 0) {
    utils::logmesg(lmp, "Fix fast/swap: requested FULL neighbor list\n");
    if (unequal_cutoffs)
      utils::logmesg(lmp,
                     "Fix fast/swap: detected unequal cutoffs among swap types -> using the "
                     "maximum trial cutoff in the FULL neighbor list\n");
  }
}

/* ---------------------------------------------------------------------- */

void FixFastSwap::init_list(int /*id*/, NeighList *ptr)
{
  // In LAMMPS 2Aug2023, NeighList has no "half" flag.
  // Since this fix only requests REQ_FULL, the list we receive here
  // must be the FULL neighbor list we asked for.
  list_full = ptr;

  if (comm->me == 0) utils::logmesg(lmp, "Fix fast/swap: neighbor list attached (assumed FULL)\n");
}

/* ---------------------------------------------------------------------- */

void FixFastSwap::post_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  // Only act at outermost level
  if (ilevel == 0) post_force(vflag);
}

void FixFastSwap::pre_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  // Only act at outermost level
  if (ilevel == 0) pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixFastSwap::post_force(int /*vflag*/)
{
  if (!list_full) return;
  if (nevery > 1 && (update->ntimestep % nevery)) return;

  // build eligible atom list and global offsets
  build_local_swap_list();
  if (niswap_global == 0) return;
  // number of trials this invocation
  for (int it = 0; it < ncycle; it++) { attempt_one_flip(); 
  }
}


/* ----------------------------------------------------------------------
   Perform Monte Carlo swap cycles at PRE_FORCE.
   This is aligned with LAMMPS fix atom/swap behavior: trial moves happen
   after neighbor list build (if any) and before force evaluation.
------------------------------------------------------------------------- */

void FixFastSwap::pre_force(int /*vflag*/)
{
  if (!list_full) return;
  if (nevery > 1 && (update->ntimestep % nevery)) return;

  // build eligible atom list and global offsets
  build_local_swap_list();
  if (niswap_global == 0) return;

  // number of trials this invocation
  bigint done = 0;
  while (done < ncycle) {
    const int attempted_global = attempt_one_flip();
    done += (attempted_global > 0) ? attempted_global : 1;
  }
}

/* ----------------------------------------------------------------------
   Build local list of eligible atoms (in group, optional region not implemented here)
------------------------------------------------------------------------- */

void FixFastSwap::build_local_swap_list()
{
  const int nlocal = atom->nlocal;
  const int *mask = atom->mask;
  const int *type = atom->type;

  // ensure capacity
  if (nlocal > nmax_local_list) {
    nmax_local_list = nlocal;
    memory->destroy(local_iatom_list);
    memory->create(local_iatom_list, nmax_local_list, "fast/swap:local_iatom_list");
  }

  // build list: in groupbit AND type is in swap type set
  niswap_local = 0;
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    int ti = type[i];
    int ok = 0;
    for (int k = 0; k < nswaptypes; k++) {
      if (ti == type_list[k]) {
        ok = 1;
        break;
      }
    }
    if (!ok) continue;

    local_iatom_list[niswap_local++] = i;
  }

  // global counts + prefix offsets
  MPI_Allreduce(&niswap_local, &niswap_global, 1, MPI_INT, MPI_SUM, world);

  // prefix sum to get niswap_before (exclusive)
  int prefix;
  MPI_Scan(&niswap_local, &prefix, 1, MPI_INT, MPI_SUM, world);
  niswap_before = prefix - niswap_local;

  if (comm->nprocs > 1 && mc_parallel_enabled) build_parallel_candidate_bins();
}

/* ----------------------------------------------------------------------
   Build cached candidate lists for the 2x2x2 active subregions.
   This is done once per MC invocation rather than once per trial move.
------------------------------------------------------------------------- */

void FixFastSwap::build_parallel_candidate_bins()
{
  for (int sid = 0; sid < 8; ++sid) candidate_bins_[sid].clear();

  if (niswap_local == 0) return;

  const double xmid = 0.5 * (domain->sublo[0] + domain->subhi[0]);
  const double ymid = 0.5 * (domain->sublo[1] + domain->subhi[1]);
  const double zmid = 0.5 * (domain->sublo[2] + domain->subhi[2]);

  double **x = atom->x;
  for (int k = 0; k < niswap_local; ++k) {
    const int ii = local_iatom_list[k];
    const double xi = x[ii][0];
    const double yi = x[ii][1];
    const double zi = x[ii][2];
    if (xi < domain->sublo[0] || xi >= domain->subhi[0] || yi < domain->sublo[1] ||
        yi >= domain->subhi[1] || zi < domain->sublo[2] || zi >= domain->subhi[2])
      continue;
    const int hx = (xi >= xmid) ? 1 : 0;
    const int hy = (yi >= ymid) ? 1 : 0;
    const int hz = (zi >= zmid) ? 1 : 0;
    const int sid = hx | (hy << 1) | (hz << 2);
    candidate_bins_[sid].push_back(ii);
  }
}

/* ----------------------------------------------------------------------
   Pick one global eligible atom uniformly (same on all ranks)
   Return local atom index on owning rank, else -1
------------------------------------------------------------------------- */

int FixFastSwap::pick_i_swap_atom_serial()
{
  int i = -1;
  if (niswap_global == 0) return -1;

  int iwhichglobal = static_cast<int>(niswap_global * random_equal->uniform());
  if (iwhichglobal < 0) iwhichglobal = 0;
  if (iwhichglobal >= niswap_global) iwhichglobal = niswap_global - 1;

  if (iwhichglobal >= niswap_before && iwhichglobal < (niswap_before + niswap_local)) {
    const int iwhichlocal = iwhichglobal - niswap_before;
    i = local_iatom_list[iwhichlocal];
  }
  return i;
}

int FixFastSwap::pick_i_swap_atom()
{
  if (niswap_global == 0) return -1;

  // Permanently serial execution path
  if (comm->nprocs == 1 || !mc_parallel_enabled) { return pick_i_swap_atom_serial(); }

  // Check parallel MC feasibility only once
  if (!mc_parallel_checked) {
    mc_parallel_checked = true;

    if (!check_parallel_mc_feasible()) {
      mc_parallel_enabled = false;

      if (comm->me == 0) {
        error->warning(FLERR,
                       "FastSwap: parallel MC disabled permanently. "
                       "Safe interior too small for current MPI decomposition.");
      }

      return pick_i_swap_atom_serial();
    }
  }

  // True parallel site selection
  return pick_i_swap_atom_parallel();
}

bool FixFastSwap::check_parallel_mc_feasible()
{
  // 1) Obtain a conservative radius rcut + skin:
  //    ghost communication cutoff (may differ by direction, take the maximum)
  const double rcut_skin =
      std::max(comm->cutghost[0], std::max(comm->cutghost[1], comm->cutghost[2]));

  // 2) Determine the required pairwise independence radius: rcut + skin.
  double R_required = rcut_skin;

  // 3) Size of a single sub-subdomain after splitting the local domain into 2×2×2
  const double Lx_sub = 0.5 * (domain->subhi[0] - domain->sublo[0]);
  const double Ly_sub = 0.5 * (domain->subhi[1] - domain->sublo[1]);
  const double Lz_sub = 0.5 * (domain->subhi[2] - domain->sublo[2]);

  // 4) Minimum sub-subdomain length along directions with neighboring processors
  double Lmin = 1.0e100;
  if (comm->procgrid[0] > 1) Lmin = std::min(Lmin, Lx_sub);
  if (comm->procgrid[1] > 1) Lmin = std::min(Lmin, Ly_sub);
  if (comm->procgrid[2] > 1) Lmin = std::min(Lmin, Lz_sub);

  // If there is effectively no parallel dimension
  // (should not happen here since nprocs > 1), act conservatively
  if (Lmin > 1.0e90) return false;

  // 5) Take the global minimum Lmin across all MPI ranks (most conservative)
  double Lmin_global = 0.0;
  MPI_Allreduce(&Lmin, &Lmin_global, 1, MPI_DOUBLE, MPI_MIN, world);

  // 6) Additional necessary condition:
  //    at least one rank must contain swappable atoms,
  //    otherwise parallel selection is meaningless
  int local_has = (niswap_local > 0) ? 1 : 0;
  int global_has = 0;
  MPI_Allreduce(&local_has, &global_has, 1, MPI_INT, MPI_SUM, world);
  if (global_has == 0) return false;

  // 7) Criterion:
  //    the minimum sub-subdomain length must exceed the required interaction radius
  return (Lmin_global > R_required);
}

// fix_fast_swap.cpp

int FixFastSwap::pick_i_swap_atom_parallel()
{
  if (niswap_global == 0) return -1;

  // 1) Synchronously select a sub-subdomain index sid: 0..7
  const int sid = static_cast<int>((mc_cycle_ * 1664525ULL + 1013904223ULL) % 8);

  // 2) Use the subregion candidate cache constructed in build_local_swap_list().
  //    This avoids scanning all local swappable atoms for every trial move.
  const std::vector<int> &candidates = candidate_bins_[sid];
  if (candidates.empty()) return -1;

  // 3) Uniformly sample from the local candidate list
  const int m = static_cast<int>(candidates.size());

  // IMPORTANT: idx must be generated using an RNG
  // that does NOT require cross-rank synchronization
  int idx = static_cast<int>(m * random_unequal->uniform());
  if (idx < 0) idx = 0;
  if (idx >= m) idx = m - 1;

  return candidates[idx];
}

/* ---------------------------------------------------------------------- */

int FixFastSwap::pick_new_type(int itype, int &jtype)
{
  // choose a new type from type_list different from itype
  int jswap = static_cast<int>(nswaptypes * random_unequal->uniform());
  if (jswap < 0) jswap = 0;
  if (jswap >= nswaptypes) jswap = nswaptypes - 1;

  jtype = type_list[jswap];
  while (jtype == itype) {
    jswap = static_cast<int>(nswaptypes * random_unequal->uniform());
    if (jswap < 0) jswap = 0;
    if (jswap >= nswaptypes) jswap = nswaptypes - 1;
    jtype = type_list[jswap];
  }
  return jswap;
}

/* ----------------------------------------------------------------------
   Local energy of atom i due to pair interactions with its neighbor list.
   Uses FULL neighbor list to avoid missing partners when i appears as 2nd index.
------------------------------------------------------------------------- */

double FixFastSwap::energy_local_atom(int iatom)
{
  double **x = atom->x;
  int *type = atom->type;
  tagint *tag = atom->tag;

  const int itype = type[iatom];

  NeighList *list = list_full;    // FULL list
  int jnum = list->numneigh[iatom];
  int *jlist = list->firstneigh[iatom];

  double ei = 0.0;
  double fpair = 0.0;

  const double xi = x[iatom][0];
  const double yi = x[iatom][1];
  const double zi = x[iatom][2];

  for (int jj = 0; jj < jnum; jj++) {
    const int jraw = jlist[jj];
    const double factor_coul = force->special_coul[jraw >> SBBITS & 3];
    const double factor_lj = force->special_lj[jraw >> SBBITS & 3];
    int j = jraw & NEIGHMASK;

    if (tag && tag[iatom] == tag[j]) continue;

    double delx = xi - x[j][0];
    double dely = yi - x[j][1];
    double delz = zi - x[j][2];
    double rsq = delx * delx + dely * dely + delz * delz;

    int jtype = type[j];

    if (rsq < force->pair->cutsq[itype][jtype]) {
      ei += force->pair->single(iatom, j, itype, jtype, rsq, factor_coul, factor_lj, fpair);
    }
  }

  return ei;
}


/* ----------------------------------------------------------------------
   Attempt a single type-flip (swap) trial move.
   - Select an eligible atom i from the local list
   - Propose a new type (from the user-provided type set)
   - Compute dE using a localized energy-difference evaluation
   - Accept/reject with Metropolis criterion including chemical potentials
------------------------------------------------------------------------- */

int FixFastSwap::attempt_one_flip()
{
  // ------------------------------------------------------------
  // Scheme B: multi-rank independent MC proposals + independent accept/reject
  // - Each rank may propose at most one swap per call (i >= 0).
  // - Accept/reject is LOCAL (no global MAX).
  // - Only if at least one rank ACCEPTS do we synchronize atom->type (and rebuild if needed).
  // - mc_cycle_ is advanced exactly once per invocation on ALL ranks.
  //
  // Requirements:
  // 1) pick_i_swap_atom_parallel() uses mc_cycle_ (or deterministic) for sid; local idx uses random_unequal.
  // 2) pick_new_type() must use random_unequal (NOT random_equal).
  // 3) comm_forward must be consistent with pairwise type-only communication.
  // 4) Parallel-feasibility check must guarantee the pairwise independence radius.
  // ------------------------------------------------------------

  mc_cycle_++;    // must be executed on all ranks exactly once per invocation

  if (niswap_global == 0) return 0;

  // Each rank may return a local atom index or -1
  const int i = pick_i_swap_atom();
  const int attempted_local = (i >= 0) ? 1 : 0;

  int itype = 0;
  int jtype = 0;
  double ediff = 0.0;

  // Local bookkeeping + choose new type (LOCAL RNG only)
  if (attempted_local) {
    n_attempts++;
    itype = atom->type[i];
    pick_new_type(itype, jtype);    // MUST use random_unequal / local RNG internally
  }

  int accept_local = 0;

  // ============================================================
  // Pairwise branch (LJ, etc.)
  // ============================================================

  if (attempted_local) {
    SelfCheckConfig &self_check = self_check_for(this);
    bool do_self_check = false;
    double full_before = 0.0;
    if (self_check.enabled && comm->nprocs == 1) {
      self_check.attempts++;
      if ((self_check.attempts % self_check.period) == 0) {
        do_self_check = true;
        full_before = energy_full();
      }
    }

    // Trial toggle locally, compute local energy change
    const double e_before = energy_local_atom(i);
    atom->type[i] = jtype;
    const double e_after = energy_local_atom(i);

    ediff = e_after - e_before;

    if (do_self_check) {
      comm->forward_comm(this);
      const double full_after = energy_full();
      const double abs_diff = std::fabs((full_after - full_before) - ediff);
      self_check.max_abs_diff = std::max(self_check.max_abs_diff, abs_diff);
      if (abs_diff > self_check.tolerance) {
        if (comm->me == 0) {
          char msg[256];
          std::snprintf(msg, sizeof(msg),
                        "Fix fast/swap self-check failed: |dE_full-dE_local| = %.17g, tolerance = %.17g\n",
                        abs_diff, self_check.tolerance);
          utils::logmesg(lmp, msg);
        }
        error->all(FLERR,
                   "Fix fast/swap self-check failed: localized energy difference "
                   "does not match full energy difference");
      }
    }

    const double arg = beta * (-ediff + mu[jtype] - mu[itype]);
    const double pacc = (arg >= 0.0) ? 1.0 : std::exp(arg);

    if (random_unequal->uniform() < pacc) {
      accept_local = 1;
      n_accepts++;
      // keep jtype
    } else {
      // reject: rollback immediately, so rejected trials do NOT leak into sync
      atom->type[i] = itype;
      if (do_self_check) comm->forward_comm(this);
    }
  }

  // One collective is sufficient to determine both progress and whether a
  // ghost-type synchronization is required.
  int mc_local[2] = {attempted_local, accept_local};
  int mc_global[2] = {0, 0};
  MPI_Allreduce(mc_local, mc_global, 2, MPI_INT, MPI_SUM, world);
  const int attempted_sum = mc_global[0];
  const int accept_sum = mc_global[1];

  if (attempted_sum == 0) return 0;

  // If nobody accepted, no global state changed -> no comm needed
  if (accept_sum == 0) return attempted_sum;

  // At least one rank accepted: synchronize ghost atom types.  For unequal
  // cutoffs, rebuild the border/neighbor state after an accepted type change
  // so ghost images and downstream pair lists remain consistent.  The MC
  // acceptance energy itself is still the local maximum-cutoff evaluation above
  // rather than a full-system energy fallback.
  if (unequal_cutoffs) {
    if (domain->triclinic) domain->x2lamda(atom->nlocal);
    comm->exchange();
    comm->borders();
    if (domain->triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
    if (modify->n_pre_neighbor) modify->pre_neighbor();
    neighbor->build(1);
  } else {
    comm->forward_comm(this);    // pairwise: comm_forward should be 1 (type only)
  }

  return attempted_sum;
}





/* ----------------------------------------------------------------------
   Forward communication pack/unpack.
   Pairwise-only version: only atom type is forwarded (comm_forward = 1).
------------------------------------------------------------------------- */

int FixFastSwap::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  int m = 0;
  int *type = atom->type;

  for (int i = 0; i < n; i++) {
    int idx = list[i];

    buf[m++] = static_cast<double>(type[idx]);

  }
  return m;
}



void FixFastSwap::unpack_forward_comm(int n, int first, double *buf)
{
  int m = 0;
  int *type = atom->type;

  for (int i = 0; i < n; i++) {
    int idx = first + i;


    type[idx] = static_cast<int>(buf[m++]);


  }
}


double FixFastSwap::compute_vector(int n)
{
  // Perform the global reduction only once per timestep,
  // upon the first request for output
  if (update->ntimestep != last_global_reduce_step) {
    bigint a_local = n_attempts;
    bigint s_local = n_accepts;

    MPI_Allreduce(&a_local, &n_attempts_global, 1, MPI_LMP_BIGINT, MPI_SUM, world);
    MPI_Allreduce(&s_local, &n_accepts_global, 1, MPI_LMP_BIGINT, MPI_SUM, world);

    last_global_reduce_step = update->ntimestep;
  }

  if (n == 0) return static_cast<double>(n_attempts_global);    // f_fix[1]
  if (n == 1) return static_cast<double>(n_accepts_global);     // f_fix[2]
  return 0.0;
}


double FixFastSwap::energy_full()
{
  double **x = atom->x;
  int *type = atom->type;
  tagint *tag = atom->tag;

  NeighList *list = list_full;
  if (!list) return 0.0;

  double epair_local = 0.0;
  double fpair = 0.0;

  const int inum = list->inum;
  int *ilist = list->ilist;

  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    if (i >= atom->nlocal) continue;

    const int itype = type[i];
    const double xi = x[i][0];
    const double yi = x[i][1];
    const double zi = x[i][2];

    const int jnum = list->numneigh[i];
    int *jlist = list->firstneigh[i];

    for (int jj = 0; jj < jnum; jj++) {
      const int jraw = jlist[jj];
      const double factor_coul = force->special_coul[jraw >> SBBITS & 3];
      const double factor_lj = force->special_lj[jraw >> SBBITS & 3];
      const int j = jraw & NEIGHMASK;

      if (tag && tag[i] == tag[j]) continue;

      const double delx = xi - x[j][0];
      const double dely = yi - x[j][1];
      const double delz = zi - x[j][2];
      const double rsq = delx * delx + dely * dely + delz * delz;

      const int jtype = type[j];
      if (rsq < force->pair->cutsq[itype][jtype])
        epair_local +=
            force->pair->single(i, j, itype, jtype, rsq, factor_coul, factor_lj, fpair);
    }
  }

  double epair_global = 0.0;
  MPI_Allreduce(&epair_local, &epair_global, 1, MPI_DOUBLE, MPI_SUM, world);

  return 0.5 * epair_global;
}





