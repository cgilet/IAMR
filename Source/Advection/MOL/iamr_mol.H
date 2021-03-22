#ifndef IAMR_MOL_H_
#define IAMR_MOL_H_

#include <AMReX_MultiFab.H>
#include <AMReX_BCRec.H>

namespace MOL {

    void ComputeAofs ( amrex::MultiFab& aofs, int aofs_comp, int ncomp,
                       amrex::MultiFab const& state, int state_comp,
                       D_DECL( amrex::MultiFab const& umac,
                               amrex::MultiFab const& vmac,
                               amrex::MultiFab const& wmac),
                       D_DECL( amrex::MultiFab& xedge,
                               amrex::MultiFab& yedge,
                               amrex::MultiFab& zedge),
                       int  edge_comp,
                       bool known_edgestate,
                       D_DECL( amrex::MultiFab& xfluxes,
                               amrex::MultiFab& yfluxes,
                               amrex::MultiFab& zfluxes),
                       int fluxes_comp,
                       amrex::Vector<amrex::BCRec> const& bcs,
		                     amrex::BCRec  const* d_bcrec_ptr,
                       amrex::Geometry const& geom,
                       amrex::Real dt
#ifdef AMREX_USE_EB
                       , std::string redistribution_type
#endif
                       );

    void ComputeSyncAofs ( amrex::MultiFab& aofs, int aofs_comp, int ncomp,
                           amrex::MultiFab const& state, int state_comp,
                           D_DECL( amrex::MultiFab const& umac,
                                   amrex::MultiFab const& vmac,
                                   amrex::MultiFab const& wmac),
                           D_DECL( amrex::MultiFab const& ucorr,
                                   amrex::MultiFab const& vcorr,
                                   amrex::MultiFab const& wcorr),
                           D_DECL( amrex::MultiFab& xedge,
                                   amrex::MultiFab& yedge,
                                   amrex::MultiFab& zedge),
                           int  edge_comp,
                           bool known_edgestate,
                           D_DECL( amrex::MultiFab& xfluxes,
                                   amrex::MultiFab& yfluxes,
                                   amrex::MultiFab& zfluxes),
                           int fluxes_comp,
                           amrex::Vector<amrex::BCRec> const& bcs,
			   amrex::BCRec  const* d_bcrec_ptr,
                           amrex::Geometry const& geom,
                           amrex::Real dt
#ifdef AMREX_USE_EB
                           , std::string redistribution_type
#endif
                          );


    void ComputeEdgeState (amrex::Box const& a_bx,
                           D_DECL( amrex::Array4<amrex::Real> const& xedge,
                                   amrex::Array4<amrex::Real> const& yedge,
                                   amrex::Array4<amrex::Real> const& zedge),
                           amrex::Array4<amrex::Real const> const& q,
                           const int ncomp,
                           D_DECL( amrex::Array4<amrex::Real const> const& umac,
                                   amrex::Array4<amrex::Real const> const& vmac,
                                   amrex::Array4<amrex::Real const> const& wmac),
                           amrex::Box const&       domain,
                           amrex::Vector<amrex::BCRec> const& bcs,
			                 amrex::BCRec  const* d_bcrec_ptr);

    void ExtrapVelToFaces ( const amrex::MultiFab&  vel,
                            D_DECL(amrex::MultiFab& umac,
                                   amrex::MultiFab& vmac,
                                   amrex::MultiFab& wmac ),
                            const amrex::Geometry&  a_geom,
                            const amrex::Vector<amrex::BCRec>& a_bcs);

    void PredictVelOnFaces ( D_DECL( amrex::Box const& ubx,
                                     amrex::Box const& vbx,
                                     amrex::Box const& wbx ),
                             D_DECL( amrex::Array4<amrex::Real> const& u,
                                     amrex::Array4<amrex::Real> const& v,
                                     amrex::Array4<amrex::Real> const& w ),
                             amrex::Array4<amrex::Real const > const& vcc,
                             const amrex::Geometry&  geom,
                             const amrex::BCRec* bcs );

#ifdef AMREX_USE_EB
    void EB_ComputeEdgeState ( amrex::Box const& bx,
                               D_DECL( amrex::Array4<amrex::Real> const& xedge,
                                       amrex::Array4<amrex::Real> const& yedge,
                                       amrex::Array4<amrex::Real> const& zedge),
                               amrex::Array4<amrex::Real const> const& q,
                               const int ncomp,
                               D_DECL( amrex::Array4<amrex::Real const> const& umac,
                                       amrex::Array4<amrex::Real const> const& vmac,
                                       amrex::Array4<amrex::Real const> const& wmac),
                               amrex::Box const&       domain,
                               amrex::Vector<amrex::BCRec> const& bcs,
 			                     amrex::BCRec  const* d_bcrec_ptr,
                               D_DECL( amrex::Array4<amrex::Real const> const& fcx,
                                       amrex::Array4<amrex::Real const> const& fcy,
                                       amrex::Array4<amrex::Real const> const& fcz),
                               amrex::Array4<amrex::Real const> const& ccc,
                               amrex::Array4<amrex::EBCellFlag const> const& flag);

    void EB_PredictVelOnFaces ( amrex::Box const& ccbx,
                                D_DECL( amrex::Box const& ubx,
                                        amrex::Box const& vbx,
                                        amrex::Box const& wbx ),
                                D_DECL( amrex::Array4<amrex::Real> const& u,
                                        amrex::Array4<amrex::Real> const& v,
                                        amrex::Array4<amrex::Real> const& w ),
                                amrex::Array4<amrex::Real const > const& vcc,
                                amrex::Array4<amrex::EBCellFlag const> const& flag,
                                D_DECL( amrex::Array4<amrex::Real const> const& fcx,
                                        amrex::Array4<amrex::Real const> const& fcy,
                                        amrex::Array4<amrex::Real const> const& fcz),
                                amrex::Array4<amrex::Real const> const& ccc,
                                const amrex::Geometry&  geom,
                                const amrex::BCRec* bcs );

#endif

}


#endif
