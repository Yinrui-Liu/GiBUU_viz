C-----------------------------------------------------------------------
C
C   This file provides dummy FORTRAN subroutines for the routines
C   provided by the HepMC3event library.
C
C-----------------------------------------------------------------------

      subroutine hepmc3eventinit(name)
      character(*) name
      write(*,*)'This program has not been linked to HepMC3event. STOP!'
      stop
      end

      subroutine hepmc3eventwrite
      end

      subroutine hepmc3eventclose
      end

      subroutine hepmc3eventaddparticlein(code,px,py,pz,e,status)
      integer code,status
      double precision px,py,pz,e
      end

      subroutine hepmc3eventaddparticleout(code,px,py,pz,e,status)
      integer code,status
      double precision px,py,pz,e
      end

      subroutine hepmc3eventinitevent(wgt)
      double precision wgt
      end

      subroutine hepmc3eventattributeimpact(v)
      double precision v
      end

      subroutine hepmc3eventattributeeventtype(v)
      integer v
      end

      subroutine hepmc3eventsetposition(x,y,z)
      double precision x,y,z
      end
