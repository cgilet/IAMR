#include <iamr_mol.H>
#include <iamr_constants.H>
#include <iamr_advection.H>
#ifdef AMREX_USE_EB
#include <AMReX_MultiCutFab.H>
#include <iamr_redistribution.H>
#endif

using namespace amrex;


void
MOL::ComputeAofs ( MultiFab& aofs, int aofs_comp, int ncomp,
                   MultiFab const& state, int state_comp,
                   D_DECL( MultiFab const& umac,
                           MultiFab const& vmac,
                           MultiFab const& wmac),
                   D_DECL( MultiFab& xedge,
                           MultiFab& yedge,
                           MultiFab& zedge),
                   int  edge_comp,
                   bool known_edgestate,
                   D_DECL( MultiFab& xfluxes,
                           MultiFab& yfluxes,
                           MultiFab& zfluxes),
                   int fluxes_comp,
                   Vector<BCRec> const& bcs,
		          BCRec  const* d_bcrec_ptr,
                   Geometry const&  geom,
                   Real dt
#ifdef AMREX_USE_EB
                   , std::string redistribution_type
#endif
                  )
{
    BL_PROFILE("MOL::ComputeAofs()");

    AMREX_ALWAYS_ASSERT(aofs.nComp()  >= aofs_comp  + ncomp);
    AMREX_ALWAYS_ASSERT(state.nComp() >= state_comp + ncomp);
    D_TERM( AMREX_ALWAYS_ASSERT(xedge.nComp() >= edge_comp  + ncomp);,
            AMREX_ALWAYS_ASSERT(yedge.nComp() >= edge_comp  + ncomp);,
            AMREX_ALWAYS_ASSERT(zedge.nComp() >= edge_comp  + ncomp););
    D_TERM( AMREX_ALWAYS_ASSERT(xfluxes.nComp() >= fluxes_comp  + ncomp);,
            AMREX_ALWAYS_ASSERT(yfluxes.nComp() >= fluxes_comp  + ncomp);,
            AMREX_ALWAYS_ASSERT(zfluxes.nComp() >= fluxes_comp  + ncomp););
    AMREX_ALWAYS_ASSERT(aofs.nGrow() == 0);
    D_TERM( AMREX_ALWAYS_ASSERT(xfluxes.nGrow() == xedge.nGrow());,
            AMREX_ALWAYS_ASSERT(yfluxes.nGrow() == yedge.nGrow());,
            AMREX_ALWAYS_ASSERT(zfluxes.nGrow() == zedge.nGrow()););

#ifdef AMREX_USE_EB
    AMREX_ALWAYS_ASSERT(state.hasEBFabFactory());

    auto const& ebfactory = dynamic_cast<EBFArrayBoxFactory const&>(state.Factory());
#endif

    Box  const& domain = geom.Domain();

    MFItInfo mfi_info;

    if (Gpu::notInLaunchRegion())  mfi_info.EnableTiling().SetDynamic(true);
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(aofs,mfi_info); mfi.isValid(); ++mfi)
    {
        auto const& bx = mfi.tilebox();
	int ng_f = xfluxes.nGrow();
	D_TERM( const Box& xbx = mfi.grownnodaltilebox(0,ng_f);,
		const Box& ybx = mfi.grownnodaltilebox(1,ng_f);,
		const Box& zbx = mfi.grownnodaltilebox(2,ng_f); );

        D_TERM( Array4<Real> fx = xfluxes.array(mfi,fluxes_comp);,
                Array4<Real> fy = yfluxes.array(mfi,fluxes_comp);,
                Array4<Real> fz = zfluxes.array(mfi,fluxes_comp););

	D_TERM( Array4<Real> xed = xedge.array(mfi,edge_comp);,
		Array4<Real> yed = yedge.array(mfi,edge_comp);,
		Array4<Real> zed = zedge.array(mfi,edge_comp););

#ifdef AMREX_USE_EB
        // Initialize covered cells
        auto const& flagfab = ebfactory.getMultiEBCellFlagFab()[mfi];
        auto const& flag    = flagfab.const_array();
	auto const& gtbx = mfi.growntilebox(ng_f);

        if (flagfab.getType(gtbx) == FabType::covered)
        {
            auto const& aofs_arr = aofs.array(mfi, aofs_comp);
            amrex::ParallelFor(bx, ncomp, [aofs_arr] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                aofs_arr( i, j, k, n ) = covered_val;
            });

            amrex::ParallelFor(xbx, ncomp, [fx,xed] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                fx( i, j, k, n ) = 0.0;
		xed( i, j, k, n ) = 0.0;
            });

            amrex::ParallelFor(ybx, ncomp, [fy,yed] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                fy( i, j, k, n ) = 0.0;
		yed( i, j, k, n ) = 0.0;
            });

#if (AMREX_SPACEDIM==3)
            amrex::ParallelFor(zbx, ncomp, [fz,zed] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                fz( i, j, k, n ) = 0.0;
		zed( i, j, k, n ) = 0.0;
            });
#endif

        }
        else
#endif
        {
            D_TERM( Array4<Real const> u = umac.const_array(mfi);,
                    Array4<Real const> v = vmac.const_array(mfi);,
                    Array4<Real const> w = wmac.const_array(mfi););

	    // Grown box on which to compute the edge states and fluxes for regular boxes
	    Box gbx = mfi.growntilebox(ng_f);

	    Box tmpbox = amrex::surroundingNodes(gbx);
	    // Space for fluxes
	    int tmpcomp = ncomp*AMREX_SPACEDIM;

#ifdef AMREX_USE_EB
	    // PeleLM needs valid flux on all ng_f cells. If !known_edgestate, need 2 additional
	    //  cells in state to compute the slopes needed to compute the edge state.
	    int halo = known_edgestate ? 0 : ng_f+2;
            bool regular = flagfab.getType(amrex::grow(bx,halo)) == FabType::regular;
	    if (!regular) {
	      // Grown box on which to compute the edge states and fluxes for EB containing boxes
	      // need at least 2 filled ghost cells all around for redistribution
	      gbx = amrex::grow(bx,ng_f);
	      tmpbox = amrex::surroundingNodes(gbx);
	      // Not sure if we really need 3(incflo) here or 2
	      int ng_diff = 3-ng_f;
	      if ( ng_diff>0 )
		tmpbox.grow(ng_diff);

	      // Add space for the temporaries needed by Redistribute
#if (AMREX_SPACEDIM == 3)
	      tmpcomp += ncomp;
#else
	      tmpcomp += 2*ncomp;
#endif
	    }
#endif
	    FArrayBox tmpfab(tmpbox, tmpcomp);
	    Elixir eli = tmpfab.elixir();

#ifdef AMREX_USE_EB
            if (!regular)
            {
                D_TERM( Array4<Real const> fcx = ebfactory.getFaceCent()[0]->const_array(mfi);,
                        Array4<Real const> fcy = ebfactory.getFaceCent()[1]->const_array(mfi);,
                        Array4<Real const> fcz = ebfactory.getFaceCent()[2]->const_array(mfi););


                D_TERM( auto apx = ebfactory.getAreaFrac()[0]->const_array(mfi);,
                        auto apy = ebfactory.getAreaFrac()[1]->const_array(mfi);,
                        auto apz = ebfactory.getAreaFrac()[2]->const_array(mfi); );

                Array4<Real const> ccc = ebfactory.getCentroid().const_array(mfi);

                auto vfrac = ebfactory.getVolFrac().const_array(mfi);

                // Compute edge state if needed
                if (!known_edgestate)
                {
		    Array4<Real const> const q = state.const_array(mfi,state_comp);

		    EB_ComputeEdgeState( gbx, D_DECL(xed,yed,zed), q, ncomp,
                                         D_DECL(u,v,w), domain, bcs, d_bcrec_ptr,
                                         D_DECL(fcx,fcy,fcz), ccc, flag );
                }

                // Compute fluxes
                Advection::EB_ComputeFluxes(gbx, D_DECL(fx,fy,fz), D_DECL(u,v,w),
                                            D_DECL(xed,yed,zed), D_DECL(apx,apy,apz),
                                            geom, ncomp, flag );

                //
                // Compute divergence and redistribute
                //
		// div at ncomp*3 to make space for the 3 redistribute temporaries
                Array4<Real> divtmp_arr = tmpfab.array(ncomp*3);



                // Compute conservative divergence
                // Redistribute needs 2 ghost cells in div
	        Box g2bx = amrex::grow(bx,2);
                Advection::EB_ComputeDivergence(g2bx, divtmp_arr, D_DECL(fx,fy,fz), vfrac,
                                                ncomp, geom );

                // Redistribute
		Array4<Real> scratch = tmpfab.array(0);
                Redistribution::Apply( bx, ncomp, aofs.array(mfi, aofs_comp), divtmp_arr,
                                       state.const_array(mfi, state_comp), scratch, flag,
                                       AMREX_D_DECL(apx,apy,apz), vfrac,
                                       AMREX_D_DECL(fcx,fcy,fcz), ccc, geom, dt,
                                       redistribution_type );

                // Change sign because for EB redistribution we compute -div
                aofs[mfi].mult(-1., bx, aofs_comp, ncomp);
            }
            else
#endif
            {
                // Compute edge state if needed
                if (!known_edgestate)
                {
                    Array4<Real const> const q = state.const_array(mfi,state_comp);
                    ComputeEdgeState( gbx, D_DECL( xed, yed, zed ), q, ncomp,
                                      D_DECL( u, v, w ), domain, bcs, d_bcrec_ptr);

                }

                // Compute fluxes
                Advection::ComputeFluxes(gbx, D_DECL(fx,fy,fz), D_DECL(u,v,w),
                                         D_DECL(xed,yed,zed), geom, ncomp );

                // Compute divergence
                std::vector<int>  iconserv(ncomp,1); // for now only conservative
                Advection::ComputeDivergence(bx, aofs.array(mfi, aofs_comp), D_DECL(fx,fy,fz),
                                             D_DECL( xed, yed, zed ), D_DECL( u, v, w ),
                                             ncomp, geom, iconserv.data());

            }
        }
    }
}




void
MOL::ComputeSyncAofs ( MultiFab& aofs, int aofs_comp, int ncomp,
                       MultiFab const& state, int state_comp,
                       D_DECL( MultiFab const& umac,
                               MultiFab const& vmac,
                               MultiFab const& wmac),
                       D_DECL( MultiFab const& ucorr,
                               MultiFab const& vcorr,
                               MultiFab const& wcorr),
                       D_DECL( MultiFab& xedge,
                               MultiFab& yedge,
                               MultiFab& zedge),
                       int  edge_comp,
                       bool known_edgestate,
                       D_DECL( MultiFab& xfluxes,
                               MultiFab& yfluxes,
                               MultiFab& zfluxes),
                       int fluxes_comp,
                       Vector<BCRec> const& bcs,
		              BCRec  const* d_bcrec_ptr,
                       Geometry const&  geom,
                       Real dt
#ifdef AMREX_USE_EB
                       , std::string redistribution_type
#endif
                  )

{
    BL_PROFILE("MOL::ComputeSyncAofs()");

    AMREX_ALWAYS_ASSERT(state.nComp() >= state_comp + ncomp);
    AMREX_ALWAYS_ASSERT(aofs.nComp()  >= aofs_comp  + ncomp);
    D_TERM( AMREX_ALWAYS_ASSERT(xedge.nComp() >= edge_comp  + ncomp);,
            AMREX_ALWAYS_ASSERT(yedge.nComp() >= edge_comp  + ncomp);,
            AMREX_ALWAYS_ASSERT(zedge.nComp() >= edge_comp  + ncomp););
    D_TERM( AMREX_ALWAYS_ASSERT(xfluxes.nComp() >= fluxes_comp  + ncomp);,
            AMREX_ALWAYS_ASSERT(yfluxes.nComp() >= fluxes_comp  + ncomp);,
            AMREX_ALWAYS_ASSERT(zfluxes.nComp() >= fluxes_comp  + ncomp););
    D_TERM( AMREX_ALWAYS_ASSERT(xfluxes.nGrow() == xedge.nGrow());,
            AMREX_ALWAYS_ASSERT(yfluxes.nGrow() == yedge.nGrow());,
            AMREX_ALWAYS_ASSERT(zfluxes.nGrow() == zedge.nGrow()););

    // Sync is always conservative
    std::vector<int>  iconserv(ncomp,1); // Sync is always conservative

#ifdef AMREX_USE_EB
    AMREX_ALWAYS_ASSERT(state.hasEBFabFactory());
    // We need at least two ghost nodes for redistribution

    auto const& ebfactory = dynamic_cast<EBFArrayBoxFactory const&>(state.Factory());
#endif

    Box  const& domain = geom.Domain();

    MFItInfo mfi_info;

    if (Gpu::notInLaunchRegion()) mfi_info.EnableTiling().SetDynamic(true);
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(aofs,mfi_info); mfi.isValid(); ++mfi)
    {
        auto const& bx = mfi.tilebox();
	D_TERM( const Box& xbx = mfi.nodaltilebox(0);,
		const Box& ybx = mfi.nodaltilebox(1);,
		const Box& zbx = mfi.nodaltilebox(2); );

        D_TERM( Array4<Real> fx = xfluxes.array(mfi,fluxes_comp);,
                Array4<Real> fy = yfluxes.array(mfi,fluxes_comp);,
                Array4<Real> fz = zfluxes.array(mfi,fluxes_comp););

	D_TERM( Array4<Real> xed = xedge.array(mfi,edge_comp);,
		Array4<Real> yed = yedge.array(mfi,edge_comp);,
		Array4<Real> zed = zedge.array(mfi,edge_comp););

#ifdef AMREX_USE_EB
        // Initialize covered cells
        auto const& flagfab = ebfactory.getMultiEBCellFlagFab()[mfi];
        auto const& flag    = flagfab.const_array();

        if (flagfab.getType(bx) == FabType::covered)
        {
            auto const& aofs_arr = aofs.array(mfi, aofs_comp);
            amrex::ParallelFor(bx, ncomp, [aofs_arr] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                aofs_arr( i, j, k, n ) = covered_val;
            });

            amrex::ParallelFor(xbx, ncomp, [fx,xed] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                fx( i, j, k, n ) = 0.0;
		xed( i, j, k, n ) = 0.0;
            });

            amrex::ParallelFor(ybx, ncomp, [fy,yed] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                fy( i, j, k, n ) = 0.0;
		yed( i, j, k, n ) = 0.0;
            });

#if (AMREX_SPACEDIM==3)
            amrex::ParallelFor(zbx, ncomp, [fz,zed] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                fz( i, j, k, n ) = 0.0;
		zed( i, j, k, n ) = 0.0;
            });
#endif
        }
        else
#endif
        {
            D_TERM( Array4<Real const> uc = ucorr.const_array(mfi);,
                    Array4<Real const> vc = vcorr.const_array(mfi);,
                    Array4<Real const> wc = wcorr.const_array(mfi););

	    Box tmpbox = amrex::surroundingNodes(bx);
	    int tmpcomp = ncomp*(AMREX_SPACEDIM+1);
#ifdef AMREX_USE_EB
	    Box gbx = bx;
	    // Need 2 grow cells in state to compute the slopes needed to compute the edge state.
	    int halo = known_edgestate ? 0 : 2;
            bool regular = flagfab.getType(amrex::grow(bx,halo)) == FabType::regular;
	    if (!regular) {
	      // Grown box on which to compute the fluxes and divergence.
	      gbx.grow(2);
	      tmpbox.grow(3);

	      // Add space for the temporaries needed by Redistribute
#if (AMREX_SPACEDIM == 3)
	      tmpcomp += ncomp;
#else
	      tmpcomp += 2*ncomp;
#endif
	    }
#endif
	    FArrayBox tmpfab(tmpbox, tmpcomp);
	    Elixir eli = tmpfab.elixir();

#ifdef AMREX_USE_EB
            if (!regular)
            {
                D_TERM( Array4<Real const> fcx = ebfactory.getFaceCent()[0]->const_array(mfi);,
                        Array4<Real const> fcy = ebfactory.getFaceCent()[1]->const_array(mfi);,
                        Array4<Real const> fcz = ebfactory.getFaceCent()[2]->const_array(mfi););

                Array4<Real const> ccc = ebfactory.getCentroid().const_array(mfi);

                auto vfrac = ebfactory.getVolFrac().const_array(mfi);

                D_TERM( auto apx = ebfactory.getAreaFrac()[0]->const_array(mfi);,
                        auto apy = ebfactory.getAreaFrac()[1]->const_array(mfi);,
                        auto apz = ebfactory.getAreaFrac()[2]->const_array(mfi); );

                // Compute edge state if needed
                if (!known_edgestate)
                {
		    Array4<Real const> const q = state.const_array(mfi,state_comp);

		    D_TERM( Array4<Real const> u = umac.const_array(mfi);,
			    Array4<Real const> v = vmac.const_array(mfi);,
			    Array4<Real const> w = wmac.const_array(mfi););

                    EB_ComputeEdgeState( gbx, D_DECL(xed,yed,zed), q, ncomp,
                                         D_DECL(u,v,w), domain, bcs, d_bcrec_ptr,
                                         D_DECL(fcx,fcy,fcz), ccc, flag );
                }

                // Compute fluxes
                Advection::EB_ComputeFluxes(gbx, D_DECL(fx,fy,fz), D_DECL(uc,vc,wc),
                                            D_DECL(xed,yed,zed), D_DECL(apx,apy,apz),
                                            geom, ncomp, flag );

                //
                // Compute divergence and redistribute
                //
		// div at ncomp*3 to make space for the 3 redistribute temporaries
                Array4<Real> divtmp_arr = tmpfab.array(ncomp*3);
                Array4<Real> divtmp_redist_arr = tmpfab.array(ncomp*4);

                // Compute conservative divergence
                std::vector<int> iconserv(ncomp,1);
                Box g2bx = amrex::grow(bx,2);
                Advection::EB_ComputeDivergence(g2bx, divtmp_arr, AMREX_D_DECL(fx,fy,fz), vfrac,
                                                ncomp, geom );

                // Redistribute
		Array4<Real> scratch = tmpfab.array(0);
                Redistribution::Apply( bx, ncomp,  divtmp_redist_arr, divtmp_arr,
                                       state.const_array(mfi, state_comp), scratch, flag,
                                       AMREX_D_DECL(apx,apy,apz), vfrac,
                                       AMREX_D_DECL(fcx,fcy,fcz), ccc, geom, dt,
                                       redistribution_type );

                // Subtract contribution to sync aofs -- sign of divergence in aofs is opposite
                // of sign of div as computed by EB_ComputeDivergence, thus it must be subtracted.
                auto const& aofs_arr = aofs.array(mfi, aofs_comp);

                amrex::ParallelFor(bx, ncomp, [aofs_arr, divtmp_redist_arr]
                AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                    aofs_arr( i, j, k, n ) +=  -divtmp_redist_arr( i, j, k, n );
                });

            }
            else
#endif
            {
                // Compute edge state if needed
                if (!known_edgestate)
                {
		    Array4<Real const> const q = state.const_array(mfi,state_comp);

		    D_TERM( Array4<Real const> u = umac.const_array(mfi);,
			    Array4<Real const> v = vmac.const_array(mfi);,
			    Array4<Real const> w = wmac.const_array(mfi););

                    ComputeEdgeState( bx, D_DECL( xed, yed, zed ), q, ncomp,
                                      D_DECL( u, v, w ), domain, bcs, d_bcrec_ptr);

                }

                // Compute fluxes
                Advection::ComputeFluxes(gbx, D_DECL(fx,fy,fz), D_DECL(uc,vc,wc),
                                         D_DECL(xed,yed,zed), geom, ncomp );

                // Compute divergence
                Array4<Real> divtmp_arr = tmpfab.array(ncomp*AMREX_SPACEDIM);
                Advection::ComputeDivergence(bx, divtmp_arr, D_DECL(fx,fy,fz),
                                             D_DECL( xed, yed, zed ), D_DECL( uc, vc, wc ),
                                             ncomp, geom, iconserv.data());

                // Sum contribution to sync aofs
                auto const& aofs_arr = aofs.array(mfi, aofs_comp);

                amrex::ParallelFor(bx, ncomp, [aofs_arr, divtmp_arr]
                AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                    aofs_arr( i, j, k, n ) += divtmp_arr( i, j, k, n );
                });
            }
        }
    }
}
