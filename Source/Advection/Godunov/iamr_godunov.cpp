#include <iamr_godunov.H>
#include <iamr_advection.H>
#include <NS_util.H>

using namespace amrex;


void
Godunov::ComputeAofs ( MultiFab& aofs, const int aofs_comp, const int ncomp,
                       MultiFab const& state, const int state_comp,
                       AMREX_D_DECL( MultiFab const& umac,
                                     MultiFab const& vmac,
                                     MultiFab const& wmac),
                       AMREX_D_DECL( MultiFab& xedge,
                                     MultiFab& yedge,
                                     MultiFab& zedge),
                       const int  edge_comp,
                       const bool known_edgestate,
                       AMREX_D_DECL( MultiFab& xfluxes,
                                     MultiFab& yfluxes,
                                     MultiFab& zfluxes),
                       int fluxes_comp,
                       MultiFab const& fq,
                       const int fq_comp,
                       MultiFab const& divu,
                       BCRec const* d_bc,
                       Geometry const& geom,
                       Gpu::DeviceVector<int>& iconserv,
                       const Real dt,
                       const bool use_ppm,
                       const bool use_forces_in_trans,
                       const bool is_velocity  )
{
    BL_PROFILE("Godunov::ComputeAofs()");

    //FIXME - check on adding tiling here
    for (MFIter mfi(aofs); mfi.isValid(); ++mfi)
    {

        const Box& bx   = mfi.tilebox();

        //
        // Get handlers to Array4
        //
        AMREX_D_TERM( const auto& fx = xfluxes.array(mfi,fluxes_comp);,
                      const auto& fy = yfluxes.array(mfi,fluxes_comp);,
                      const auto& fz = zfluxes.array(mfi,fluxes_comp););

        AMREX_D_TERM( const auto& xed = xedge.array(mfi,edge_comp);,
                      const auto& yed = yedge.array(mfi,edge_comp);,
                      const auto& zed = zedge.array(mfi,edge_comp););

        AMREX_D_TERM( const auto& u = umac.const_array(mfi);,
                      const auto& v = vmac.const_array(mfi);,
                      const auto& w = wmac.const_array(mfi););

        if (not known_edgestate)
        {
            ComputeEdgeState( bx, ncomp,
                              state.array(mfi,state_comp),
                              AMREX_D_DECL( xed, yed, zed ),
                              AMREX_D_DECL( u, v, w ),
                              divu.array(mfi),
                              fq.array(mfi,fq_comp),
                              geom, dt, d_bc,
                              iconserv.data(),
                              use_ppm,
                              use_forces_in_trans,
                              is_velocity );
        }

        Advection::ComputeFluxes( bx, AMREX_D_DECL( fx, fy, fz ),
                       AMREX_D_DECL( u, v, w ),
                       AMREX_D_DECL( xed, yed, zed ),
                       geom, ncomp );

        Advection::ComputeDivergence( bx,
                           aofs.array(mfi,aofs_comp),
                           AMREX_D_DECL( fx, fy, fz ),
                           AMREX_D_DECL( xed, yed, zed ),
                           AMREX_D_DECL( u, v, w ),
                           ncomp, geom, iconserv.data() );

	//
	// NOTE this sync cannot protect temporaries in ComputeEdgeState, ComputeFluxes
	// or ComputeDivergence, since functions have their own scope. As soon as the
	// CPU hits the end of the function, it will call the destructor for all
	// temporaries created in that function.
	//
        Gpu::streamSynchronize();  // otherwise we might be using too much memory
    }

}



void
Godunov::ComputeSyncAofs ( MultiFab& aofs, const int aofs_comp, const int ncomp,
                           MultiFab const& state, const int state_comp,
                           AMREX_D_DECL( MultiFab const& umac,
                                         MultiFab const& vmac,
                                         MultiFab const& wmac),
                           AMREX_D_DECL( MultiFab const& ucorr,
                                         MultiFab const& vcorr,
                                         MultiFab const& wcorr),
                           AMREX_D_DECL( MultiFab& xedge,
                                         MultiFab& yedge,
                                         MultiFab& zedge),
                           const int  edge_comp,
                           const bool known_edgestate,
                           AMREX_D_DECL( MultiFab& xfluxes,
                                         MultiFab& yfluxes,
                                         MultiFab& zfluxes),
                           int fluxes_comp,
                           MultiFab const& fq,
                           const int fq_comp,
                           MultiFab const& divu,
                           BCRec const* d_bc,
                           Geometry const& geom,
                           Gpu::DeviceVector<int>& to_be_removed, // iconserv,
                           const Real dt,
                           const bool use_ppm,
                           const bool use_forces_in_trans,
                           const bool is_velocity  )
{
    BL_PROFILE("Godunov::ComputeAofs()");

    // Sync is always conservative
    std::vector<int> iconserv(ncomp,1);

    //FIXME - check on adding tiling here
    for (MFIter mfi(aofs); mfi.isValid(); ++mfi)
    {

        const Box& bx   = mfi.tilebox();

        //
        // Get handlers to Array4
        //
        AMREX_D_TERM( const auto& fx = xfluxes.array(mfi,fluxes_comp);,
                      const auto& fy = yfluxes.array(mfi,fluxes_comp);,
                      const auto& fz = zfluxes.array(mfi,fluxes_comp););

        AMREX_D_TERM( const auto& xed = xedge.array(mfi,edge_comp);,
                      const auto& yed = yedge.array(mfi,edge_comp);,
                      const auto& zed = zedge.array(mfi,edge_comp););

        AMREX_D_TERM( const auto& uc = ucorr.const_array(mfi);,
                      const auto& vc = vcorr.const_array(mfi);,
                      const auto& wc = wcorr.const_array(mfi););

        if (not known_edgestate)
        {

            AMREX_D_TERM( const auto& u = umac.const_array(mfi);,
                          const auto& v = vmac.const_array(mfi);,
                          const auto& w = wmac.const_array(mfi););

            ComputeEdgeState( bx, ncomp,
                              state.array(mfi,state_comp),
                              AMREX_D_DECL( xed, yed, zed ),
                              AMREX_D_DECL( u, v, w ),
                              divu.array(mfi),
                              fq.array(mfi,fq_comp),
                              geom, dt, d_bc,
                              iconserv.data(),
                              use_ppm,
                              use_forces_in_trans,
                              is_velocity );
        }

        Advection::ComputeFluxes( bx, AMREX_D_DECL( fx, fy, fz ),
                                  AMREX_D_DECL( uc, vc, wc ),
                                  AMREX_D_DECL( xed, yed, zed ),
                                  geom, ncomp );


        Advection::ComputeDivergence( bx, aofs.array(mfi, aofs_comp), D_DECL(fx,fy,fz),
                                      D_DECL( xed, yed, zed ), D_DECL( uc, vc, wc ),
                                      ncomp, geom, iconserv.data());

	// Note this sync is needed since ComputeEdgeState() contains temporaries
	// Not sure it's really needed when known_edgestate==true
        Gpu::streamSynchronize();  // otherwise we might be using too much memory
    }
}
