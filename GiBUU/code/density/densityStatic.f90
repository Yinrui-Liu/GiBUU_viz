!******************************************************************************
!****m* /densityStatic
! NAME
! module densityStatic
!
! PURPOSE
! Collect routines for STATIC density calculations.
!******************************************************************************
module densityStatic

  use nucleusDefinition
  use dichteDefinition
  use Callstack, only: Traceback

  implicit none


  private

  !****************************************************************************
  !****g* densityStatic/useCentroids
  ! SOURCE
  !
  logical, save :: useCentroids = .false.
  ! PURPOSE
  ! If this switch is 'true', then the density of the proton and neutron centers
  ! will be tabulated that is different from the matter density.
  ! NOTES
  ! presently relevant only for densitySwitch_static=2 (Luis routine)
  !****************************************************************************

  public :: staticDensity,staticDensityInit,densityLuis
  public :: ReAdjust


contains

  !****************************************************************************
  !****s* densityStatic/staticDensityInit
  ! NAME
  ! subroutine staticDensityInit(nuc)
  ! PURPOSE
  ! decide, which density parametrisation is used. Then tabulate this and
  ! also set the extreme values for the MC decision.
  ! INPUTS
  ! * type(tNucleus) :: nuc    -- nucleus which is regarded
  ! OUTPUT
  ! * type(tNucleus) :: nuc
  !****************************************************************************
  subroutine staticDensityInit(nuc)

    use output, only: Write_ReadingInput

    type(tNucleus),pointer :: nuc

    NAMELIST /initStaticDensity/ useCentroids

    integer :: ios

    call Write_ReadingInput("initStaticDensity",0)
    rewind(5)
    read(5,nml=initStaticDensity,iostat=ios)
    call Write_ReadingInput("initStaticDensity",0,ios)

    write(*,*)' useCentroids : ', useCentroids

    call Write_ReadingInput("initStaticDensity",1)

    if (nuc%mass <= 2) then
       if (nuc%densityswitch_static > 0) then
          write(*,*)
          write(*,'(A)') '  WARNING: static density not possible with Mass<=2!'
          write(*,'(A)') '  setting densityswitch_static to 0 for this run!!!'
          write(*,*)
          nuc%densityswitch_static = 0
       end if
       return
    end if


    select case (nuc%densitySwitch_static)
    case (0) ! Density=0.
       call TabulateZero(nuc)

    case (1) ! Woods-Saxon distribution, proton=neutron
       call TabulateDensityWoodsSaxon(nuc)

    case (2) ! prescription implemented by L.Alvarez-Russo
             ! corresponds to Oset's papers, e.g. NPA 554, 509 (1993)
       call TabulateDensityLuis(nuc)

    case (3) ! Woods-Saxon distribution,
             ! different parameters for neutrons and protons
       call TabulateDensityLenske(nuc)

    case (4) ! Static Harmonic Oscilator Shell model
       call TabulateDensityHarmOsc(nuc)

    case (5) ! Fermi gas model with no surface term
       call TabulateSphere(nuc)

    case (6) ! based on LDA, implemented by Birger Steinmueller
       call TabulateDensityBirger(nuc)

    case (7) ! based on LDA + Welke potential
       call TabulateDensityBirgerWelke(nuc)

    case (8) ! prescription according Relativistic Thomas-Fermi
             ! Valid only in RMF-mode
       call TabulateDensityExRTF(nuc)

    case default
       write(*,*) 'wrong DensitySwitch_static:', nuc%densitySwitch_static
       call Traceback('STOP')

    end select

    call nuc%SearchMaxVals()
    call nuc%writeAverageDens()

  end subroutine staticDensityInit


  !****************************************************************************
  !****s* densityStatic/TabulateZero
  ! NAME
  ! subroutine TabulateZero(nuc)
  ! PURPOSE
  ! Set the density table to zero
  ! INPUTS
  ! * type(tNucleus) :: nuc
  !
  ! OUTPUT
  ! * type(tNucleus) :: nuc
  !
  !****************************************************************************
  subroutine TabulateZero(nuc)
    use output

    type(tNucleus),pointer :: nuc

    call Write_InitStatus('density tabulation (density=0.)',0)
    write(*,'(A,F8.4)') '    dr   =',nuc%dx
    write(*,'(A,F8.4)') '    r_max=',nuc%dx*float(nuc%maxIndex)

    nuc%densTab = 0.

    call Write_InitStatus('density tabulation (density=0.)',1)

  end subroutine TabulateZero



  !****************************************************************************
  !****s* densityStatic/TabulateSphere
  ! NAME
  ! subroutine TabulateFermiGas(nuc)
  ! PURPOSE
  ! Tabulate a sphere with constant density.
  !
  ! Tabulates the static density to make it available faster for later use
  ! INPUTS
  ! * type(tNucleus) :: nuc
  !
  ! OUTPUT
  ! * type(tNucleus) :: nuc
  !
  !****************************************************************************
  subroutine TabulateSphere(nuc)
    use output

    type(tNucleus),pointer :: nuc

    real :: x
    integer :: i
    real, parameter :: epsilon=0.001

    call Write_InitStatus('density tabulation (Sphere)',0)
    write(*,'(A,F8.4)') '    dr   =',nuc%dx
    write(*,'(A,F8.4)') '    r_max=',nuc%dx*float(nuc%maxIndex)

    if (nuc%radius(0).lt.epsilon) then
       write(*,*) 'SEVERE ERROR: This nucleus is not well defined.'
       write(*,*) 'Radius=',nuc%radius
       write(*,*) 'A=',nuc%mass,'  Z= ',nuc%charge
       write(*,*) 'Not possible to use that nucleus for sphere density.'
       call Traceback('Stop')
    end if

    nuc%densTab = 0.

    do i=0,nuc%MaxIndex
       x = i*nuc%dx
       if (x.le.nuc%radius(0)) then
          nuc%densTab(i,:) = nuc%density
       end if
    end do
    call Write_InitStatus('density tabulation (Sphere)',1)

  end subroutine TabulateSphere


  !****************************************************************************
  !****s* densityStatic/TabulateDensityWoodsSaxon
  ! NAME
  ! subroutine TabulateDensityWoodsSaxon(nuc)
  ! PURPOSE
  ! Tabulate the Woods-Saxon distribution. Tabulates the static density
  ! to make it available faster for later use.
  !
  ! parameters for protons and neutrons are equal
  !
  ! INPUTS
  ! * type(tNucleus) :: nuc
  ! OUTPUT
  ! * type(tNucleus) :: nuc
  !****************************************************************************
  subroutine TabulateDensityWoodsSaxon(nuc)
    use output, only: write_initstatus, paragraph
    use constants, only: pi

    type(tNucleus),pointer :: nuc

    real :: x,h,ratio,s
    integer :: i
    real, parameter :: epsilon=0.001

    call Write_InitStatus('density tabulation (Woods-Saxon)',0)
    write(*,'(A,F8.4)') '    dr   =',nuc%dx
    write(*,'(A,F8.4)') '    r_max=',nuc%dx*float(nuc%maxIndex)

    if (nuc%surface(0)<epsilon .or. nuc%radius(0)<epsilon) then
       write(*,*) 'SEVERE ERROR: This nucleus is not well defined.'
       write(*,*) 'Radius=',nuc%radius(0),'   Surface=',nuc%surface(0)
       write(*,*) 'A=',nuc%mass,'  Z= ',nuc%charge
       write(*,*) 'Not possible to use that nucleus for Woods-Saxon density.'
       call Traceback('Stop')
    end if

    ! check input values overwriting default numbers:
    if ((nuc%radius_input(1) > 0).or.(nuc%surface_input(1) > 0)) then
       write(*,*) "some parameters are overwritten by the input!"

       if (nuc%radius_input(1) > 0)  nuc%radius  = nuc%radius_input(1)
       if (nuc%surface_input(1) > 0) nuc%surface = nuc%surface_input(1)
       nuc%density = -99.9 ! here as dummy
    end if


    ratio = float(nuc%charge)/float(nuc%mass)
    s = 0
    do i=0,nuc%MaxIndex
       x = i*nuc%dx
       h = nuc%density(0) /(1.+exp((x-nuc%radius(0))/nuc%surface(0)))
       s = s + x**2*h
       nuc%densTab(i,:) = h * (/ 1., ratio, 1-ratio /)
    end do

    if (nuc%density(0) <0 ) then
       write(*,*) '...nuc%density will be adjusted'
       s = s * nuc%dx * 4*pi
       h = float(nuc%mass)/s
       nuc%densTab = nuc%densTab * h
       nuc%density = nuc%density * h
    end if

    nuc%density(1:2) = nuc%density(0) * (/ ratio, 1-ratio /)

    write(*,paragraph) ' Parameters: '
    write(*,'(A)')       '              Baryon:  Proton: Neutron:'
    write(*,'(A,3F9.4)') '    rho_max=',nuc%density
    write(*,'(A,3F9.4)') '    radius =',nuc%radius
    write(*,'(A,3F9.4)') '    surface=',nuc%surface

    call Write_InitStatus('density tabulation (Woods-Saxon)',1)

  end subroutine TabulateDensityWoodsSaxon


  !****************************************************************************
  !****s* densityStatic/TabulateDensityHarmOsc
  ! NAME
  ! subroutine TabulateDensityHarmOsc(nuc)
  ! PURPOSE
  ! Tabulate the density distribution according harmonic oscillator shell
  ! modell.
  !
  ! Tabulates the static density to make it available faster for later use
  ! INPUTS
  ! * type(tNucleus) :: nuc
  !
  ! OUTPUT
  ! * type(tNucleus) :: nuc
  !
  ! NOTES
  ! The parameter are taken from the FRITIOF package.
  !****************************************************************************
  subroutine TabulateDensityHarmOsc(nuc)
    use output
    use constants

    type(tNucleus),pointer :: nuc

    real :: x,h,ratio, h1,h2,h3
    integer :: i
    logical :: okay

    integer, parameter:: PossibleA(2:8) = &
         (/   4, -1,     9,   11,    12, -1,    16/)
    real,    parameter:: rCh(2:8) = &
         (/1.74, -1., 2.519, 2.37, 2.446, -1., 2.724/)
    real,    parameter:: d2(2:8) = &
         (rCh**2-0.81**2) / (2.5-4./PossibleA)


    call Write_InitStatus('density tabulation (Harm. Osc.)',0)
    write(*,'(A,F8.4)') '    dr   =',nuc%dx
    write(*,'(A,F8.4)') '    r_max=',nuc%dx*float(nuc%maxIndex)

    okay = .true.
    if ( (nuc%charge<2) .or. (nuc%charge>8) ) then
       okay = .false.
    else
       if (nuc%mass.ne.PossibleA(nuc%charge)) okay = .false.
    end if

    if (.not.okay) then
       write(*,*) 'SEVERE ERROR:'
       write(*,*) 'Radius=',nuc%radius,'   Surface=',nuc%surface
       write(*,*) 'A=',nuc%mass,'  Z= ',nuc%charge
       write(*,*) 'Not possible to use that nucleus for harm. osc. density.'
       call Traceback('Stop')
    end if

    write(*,*) 'Parameters: r_ch = ',rCh(nuc%charge),' d2 = ',d2(nuc%charge)

    ratio = float(nuc%charge)/float(nuc%mass)

    h1 = 4/(pi * d2(nuc%charge))**(3./2.)
    h2 = (nuc%mass-4)/6.

    do i=0,nuc%MaxIndex
       x = i*nuc%dx
       h3 = x**2/d2(nuc%charge)
       h = h1 * (1+h2*h3) * exp(-h3)

       nuc%densTab(i,:) = h * (/ 1., ratio, 1.-ratio /)
    end do
    call Write_InitStatus('density tabulation (Harm. Osc.)',1)

  end subroutine TabulateDensityHarmOsc


  !****************************************************************************
  !****s* densityStatic/TabulateDensityLenske
  ! NAME
  ! subroutine TabulateDensityLenske(nuc)
  ! PURPOSE
  ! Tabulate the density distribution according to Woods-Saxon distribution
  ! but with refined charge radii for proton and neutron according to H. Lenske.
  ! Tabulates the static density to make it available faster for later use.
  ! INPUTS
  ! * type(tNucleus) :: nuc
  ! OUTPUT
  ! * type(tNucleus) :: nuc
  ! NOTES
  ! Everything in fm.
  !****************************************************************************
  subroutine TabulateDensityLenske(nuc)
    use constants, only: pi
    use nucD, only: dfs
    use output, only: write_initstatus, paragraph

    type(tNucleus),pointer :: nuc

    real, dimension(1:3) :: radius, surface, rhoMax
    real, dimension(1:2) :: h,s
    real :: x, ratio=0.
    integer :: i

    call Write_InitStatus('density tabulation (Lenske & Woods-Saxon)',0)
    write(*,'(A,F8.4)') '    dr     =',nuc%dx
    write(*,'(A,F8.4)') '    r_max  =',nuc%dx*float(nuc%maxIndex)

    call DFS(nuc%mass,nuc%charge,radius,surface,rhoMax)

    ! reorder the arrays:
    nuc%radius =  (/ radius(3), radius(1), radius(2) /)
    nuc%surface = (/ surface(3), surface(1), surface(2) /)
    nuc%density = (/ rhoMax(3), rhoMax(1), rhoMax(2) /)

    ratio = float(nuc%charge)/float(nuc%mass)

    if ((any(nuc%radius_input > 0)).or.(any(nuc%surface_input > 0))) then
       write(*,*)
       write(*,*) "some parameters are overwritten by the input!"

       if (nuc%radius_input(1) > 0)  nuc%radius(1)  = nuc%radius_input(1)
       if (nuc%radius_input(2) > 0)  nuc%radius(2)  = nuc%radius_input(2)
       if (nuc%surface_input(1) > 0) nuc%surface(1) = nuc%surface_input(1)
       if (nuc%surface_input(2) > 0) nuc%surface(2) = nuc%surface_input(2)

       nuc%radius(0)  = ratio*nuc%radius(1)  + (1.-ratio)*nuc%radius(2)
       nuc%surface(0) = ratio*nuc%surface(1) + (1.-ratio)*nuc%surface(2)

       nuc%density = -99.9 ! here as dummy
    end if


    ! one could use the approximation of the normalization
    ! 4pi /int dr r^2/(1+exp((r-R)/a)) ~= 4pi/3 R^3(1+pi^2(a/R)^2)
    ! (neglecting terms O( exp(-R/a) ))
    ! cf. Bohr&Mottelson, Nuclear Structure (1998), p.160

    s = 0
    do i=0,nuc%MaxIndex
       x = i*nuc%dx
       h = nuc%density(1:2)/(1+exp((x-nuc%radius(1:2))/nuc%surface(1:2)))
       nuc%densTab(i,1:2) = h
       s = s + x**2*h
    end do

    if (nuc%density(0) <0 ) then
       write(*,*) '...nuc%density will be adjusted'

       s = s * nuc%dx * 4*pi
       h = float(nuc%mass)/s * (/ratio, 1.-ratio/)
       nuc%densTab(:,1) = nuc%densTab(:,1) * h(1)
       nuc%densTab(:,2) = nuc%densTab(:,2) * h(2)
       nuc%density(1:2) = nuc%density(1:2) * h(1:2)
    end if

    nuc%densTab(:,0) = nuc%densTab(:,1)+nuc%densTab(:,2)
    nuc%density(0) = nuc%density(1) + nuc%density(2)

    write(*,*)
    write(*,paragraph) ' Parameters: '
    write(*,'(A)')       '              Baryon:  Proton: Neutron:'
    write(*,'(A,3F9.4)') '    rho_max=',nuc%density
    write(*,'(A,3F9.4)') '    radius =',nuc%radius
    write(*,'(A,3F9.4)') '    surface=',nuc%surface

    call Write_InitStatus('density tabulation (Lenske & Woods-Saxon)',1)

  end subroutine TabulateDensityLenske


  !****************************************************************************
  !****s* densityStatic/TabulateDensityLuis
  ! NAME
  ! subroutine TabulateDensityLuis(nuc)
  ! PURPOSE
  ! Tabulate the density distribution of matter (p and n)
  ! and the density of centers (p and n number densities) following
  ! J.Nieves, E.Oset, C.Garcia-Recio, Nucl.Phys.A 554 (1993) 509
  !
  ! Tabulates the static density to make it available faster for later use
  ! INPUTS
  ! * type(tNucleus) :: nuc
  !
  ! OUTPUT
  ! * type(tNucleus) :: nuc
  !
  ! NOTES
  ! everything in fm
  !****************************************************************************
  subroutine TabulateDensityLuis(nuc)
    use output

    type(tNucleus),pointer :: nuc

    real :: x
    real :: rp,ap,rho0p,rn,an,rho0n
    integer :: i

    call Write_InitStatus('density tabulation (NPA554)',0)
    write(*,'(A,F8.4)') '    dr   =',nuc%dx
    write(*,'(A,F8.4)') '    r_max=',nuc%dx*float(nuc%maxIndex)

    do i=0,nuc%MaxIndex
       x = i*nuc%dx
       call densityLuis(x,nuc%charge,nuc%mass,nuc%densTab(i,1),nuc%densTab(i,2),rp,ap,rho0p,rn,an,rho0n)
    end do
    nuc%densTab(:,0) = nuc%densTab(:,1)+nuc%densTab(:,2)


    write(*,paragraph) ' Parameters: '
    write(*,'(A,3(1x,F8.4))') ' rp, ap, rho0p : ', rp,ap,rho0p
    write(*,'(A,3(1x,F8.4))') ' rn, an, rho0n : ', rn,an,rho0n
    call Write_InitStatus('density tabulation (NPA554)',1)

  end subroutine TabulateDensityLuis

  !****************************************************************************
  !****s* densityStatic/TabulateDensityExRTF
  ! NAME
  ! subroutine TabulateDensityExRTF(nuc)
  ! PURPOSE
  ! Tabulate the density distribution according to Relativistic
  ! Thomas-Fermi model code from Horst Lenske.
  !
  ! Tabulates the static density to make it available faster for later use
  ! INPUTS
  ! * type(tNucleus) :: nuc
  !
  ! OUTPUT
  ! * type(tNucleus) :: nuc
  !
  ! NOTES
  ! everything in fm
  !****************************************************************************
  subroutine TabulateDensityExRTF(nuc)
    use NucExRTF, only: NucExRTF_Main
    use output

    type(tNucleus),pointer :: nuc

    integer :: i
    real, dimension(0:2000, 1:2) :: RTF_dens

    call Write_InitStatus('density tabulation (ExRTF Lenske)',0)
    write(*,'(A,F8.4)') '    dr   =',nuc%dx
    write(*,'(A,F8.4)') '    r_max=',nuc%dx*float(nuc%maxIndex)

    call NucExRTF_Main(nuc%mass,nuc%charge,RTF_dens)

    do i=0,nuc%MaxIndex
       nuc%densTab(i,1) = RTF_dens(i,1)
       nuc%densTab(i,2) = RTF_dens(i,2)
    end do
    nuc%densTab(:,0) = nuc%densTab(:,1)+nuc%densTab(:,2)

    call Write_InitStatus('density tabulation (ExRTF Lenske)',1)

  end subroutine TabulateDensityExRTF


  !****************************************************************************
  !****f* densityStatic/staticDensity
  ! NAME
  ! type(dichte) function staticDensity(r,nucl)
  ! PURPOSE
  ! gives density in the restframe of the nucleus "nucl" at position "r"
  ! INPUTS
  ! * real, dimension(1:3) :: r -- position where density should be calculated
  ! * type(tNucleus),pointer :: nucl    -- nucleus which is regarded
  ! USAGE
  ! (dichte)=staticDensity(...)
  !****************************************************************************
  type(dichte) function staticDensity(r,nucl)

    real, dimension(1:3),intent(in) :: r
    type(tNucleus),pointer       :: nucl

    real :: sqrtR
    integer :: i

    if (nucl%DoInit) then
       call Traceback('nucleus not initialized! stop')
    end if


    select case (nucl%densitySwitch_static)
    case (0)
       staticdensity%proton  = 0.
       staticdensity%neutron = 0.
       staticdensity%baryon  = 0.

    case (1:8) ! Tabulated density distributions

       sqrtR=sqrt(r(1)**2+r(2)**2+r(3)**2)

       if (sqrtR>nucl%MaxIndex*nucl%dx) then
          staticdensity%proton(0)  = 0.
          staticdensity%neutron(0) = 0.
          staticdensity%baryon(0)  = 0.
       else
          i = min(nint(sqrtR/nucl%dx),nucl%maxIndex)
          staticdensity%proton(0)  = nucl%densTab(i,1)
          staticdensity%neutron(0) = nucl%densTab(i,2)
          staticdensity%baryon(0)  = nucl%densTab(i,0)
       end if

       ! set flux to zero (lrf):
       staticdensity%proton(1:3)  = 0.
       staticdensity%neutron(1:3) = 0.
       staticdensity%baryon(1:3)  = 0.

    case default

       write(*,*) 'Error in static density: ', &
            'DensitySwitch_static is not well defined',&
            nucl%densitySwitch_static
       call Traceback('Severe Error : STOP')
    end select

    staticdensity%charge = staticdensity%proton

  end function staticDensity


  !****************************************************************************
  !****s* densityStatic/densityLuis
  ! NAME
  ! subroutine densityLuis(r,z,a,rhop,rhon,rp,ap,rho0p,rn,an,rho0n)
  ! PURPOSE
  ! This routine calculates the proton and neutron densities following
  ! J.Nieves, E.Oset, C.Garcia-Recio, Nucl.Phys.A 554 (1993) 509
  !
  ! returns per default the density of matter, set useCentroids=.true. to
  ! switch to density of centers
  !
  ! INPUTS
  ! * real    :: r  -- radius (fm)
  ! * integer :: z  -- charge of the nucleus
  ! * integer :: a  -- atomic number
  ! RESULT
  ! * real :: rhop,rhon -- Proton and neutron densities (fm^-3) at r
  ! * real :: rp,ap,rho0p,rn,an,rho0n -- parameters of the density distributions
  !****************************************************************************
  subroutine densityLuis(r,z,a,rhop,rhon,rp,ap,rho0p,rn,an,rho0n)
    use constants, only: pi
    use gauss_integration, only: sg20r, rg20r

    real, intent(in)::r
    integer, intent(in)::z
    integer, intent(in)::a
    real, intent(out):: rhop,rhon
    real, intent(out) :: rp,ap,rho0p,rn,an,rho0n

    real::x,rpc,apc,rnc,anc,rmax,rin
    real::resu1,resu2,resu3,resu4
    real,dimension(:),allocatable::absi,orde1,orde2,orde3,orde4
    integer::nin,nins,i
    real, parameter::r2=0.69

    call denspar(z,a,rp,ap,rn,an)

    if (a.le.18) then
       !       For light nuclei, harmonic oscilator densities are used
       !       protons
       rpc=sqrt(rp**2-2./3.*r2)
       x=ap*rp**2/(1.+3./2.*ap)/rpc**2
       apc=2.*x/(2.-3.*x)
       !       neutrons
       rnc=sqrt(rn**2-2./3.*r2)
       x=an*rn**2/(1.+3./2.*an)/rnc**2
       anc=2.*x/(2.-3.*x)

    else
       !       Fermi liquid type
       !       protons
       rpc=rp+5.*r2*rp/(15.*rp**2+7.*pi**2*ap**2)
       apc=sqrt((rp**3+pi**2*ap**2*rpc-rpc**3)/pi**2/rpc)
       !       neutrons
       rnc=rn+5.*r2*rn/(15.*rn**2+7.*pi**2*an**2)
       anc=sqrt((rn**3+pi**2*an**2*rnc-rnc**3)/pi**2/rnc)

    end if

    !       normalization
    nin=20
    rmax=20.
    allocate (absi(20*nin))
    allocate (orde1(20*nin))
    allocate (orde2(20*nin))
    allocate (orde3(20*nin))
    allocate (orde4(20*nin))
    call sg20r(0.,rmax,nin,absi,nins)
    do i=1,nins
       rin=absi(i)
       orde1(i)=4.*pi*rin**2*antz(rin,rp,ap)
       orde2(i)=4.*pi*rin**2*antz(rin,rn,an)
       orde3(i)=4.*pi*rin**2*antz(rin,rpc,apc)
       orde4(i)=4.*pi*rin**2*antz(rin,rnc,anc)
    end do
    call rg20r(0.,rmax,nin,orde1,resu1)
    call rg20r(0.,rmax,nin,orde2,resu2)
    call rg20r(0.,rmax,nin,orde3,resu3)
    call rg20r(0.,rmax,nin,orde4,resu4)


    if (useCentroids) then
       ! use the distributions of nucleon centers:
       rho0p=z/resu3
       rho0n=(a-z)/resu4
       rp=rpc
       ap=apc
       rn=rnc
       an=anc
    else
       ! use matter distributions:
       rho0p=z/resu1
       rho0n=(a-z)/resu2
    end if

    rhop=rho0p*antz(r,rp,ap)
    rhon=rho0n*antz(r,rn,an)

    deallocate(absi,orde1,orde2,orde3,orde4)

  contains

    function antz(rin,rg,ag)
      implicit none
      real, intent(in)::rin,rg,ag
      real:: antz

      if (a.le.18) then
         !       harmonic oscilator
         antz=(1.+ag*(rin/rg)**2)*exp(-(rin/rg)**2)
      else
         !       Fermi liquid
         antz=1./(1.+exp((rin-rg)/ag))
      end if
    end function antz

    subroutine denspar(z,a,rp,ap,rn,an)
      ! this subroutine gives parameters for the harmonic oscillator
      ! form of densities indicated by HO or for the WS form,
      ! taken from Nieves et al, Nucl. Phys. A554 (1993 509, Table I
      implicit none
      integer, intent(in)::z,a
      real, intent(out)::rp,ap,rn,an

      select case (z)
      case (4)  ! Be (9)
         ! proton values from DeJager et al.,
         ! At. Data and Nucl. Data Tables 14, 479 (1974)
         ! neutron values from Koptev et al., Yad. Fiz. 31, 1501 (1980)
         rp=1.78
         ap=0.631
         rn=2.11
         an=1.000
      case (5)   ! B(10)    HO
         rp=1.72
         ap=0.837
         rn=rp
         an=ap
      case (6)   ! C(12)    HO
         rp=1.692
         ap=1.082
         rn=rp
         an=ap
      case (8)
         if (a.eq.16) then
            !             O(16)  HO
            rp=1.833
            ap=1.544
            rn=1.815
            an=1.529
         else
            !             O(18)  HO
            rp=1.881
            ap=1.544
            rn=1.975
            an=2.048
         end if
      case (13) ! Al(27)
         rp=3.05
         ap=0.535
         rn=rp
         an=ap
      case (18)     ! Ar(40)
         rp=3.47
         ap=0.569
         rn=3.64
         an=0.569
      case (20)
         if (a.eq.40) then
            !             Ca(40)
            rp=3.51
            ap=0.563
            rn=3.43
            an=ap
         else
            !             Ca(44)
            rp=3.573
            ap=0.563
            rn=3.714
            an=ap
         end if
      case (26) ! Fe(56)
         rp=3.971
         ap=0.5935
         rn=4.05
         an=ap
      case (29) ! Cu(63)
         rp=4.214
         ap=0.586
         rn=4.31
         an=ap
      case (33) ! As(75)
         rp=4.492
         ap=0.58
         rn=4.64
         an=ap
      case (58) ! Ce(142)
         rp=5.76
         ap=0.535
         rn=5.98
         an=ap
      case (50) ! Sn-isotopes:
         ! Values taken from R. Schmidt et al,
         ! PRC 67, 044308 (2003)
         ! --- p and n center distribution parameters,
         ! should not be further corrected
         select case (A)
         case (112)
            rp=5.416
            ap=0.497
            rn=rp
            an=0.543
         case (116)
            rp=5.399
            ap=0.486
            rn=rp
            an=0.552
         case (120)
            rp=5.356
            ap=0.515
            rn=rp
            an=0.565
         case (124)
            rp=5.530
            ap=0.467
            rn=rp
            an=0.558
         case default
            call Traceback('There is no init for this Sn isotope')
         end select
      case (73) ! Ta(181)
         ! proton values from DeJager et al
         ! neutron values from Koptev et al
         rp=6.38
         ap=0.64
         rn=6.42
         an=0.64
      case (79) ! Au(197)
         rp=6.55
         ap=0.522
         rn=6.79
         an=ap
      case (82)  ! Pb(208)
         rp=6.624
         ap=0.549
         rn=6.89
         an=0.549
      case default
         write(*,*) "For this core the density distribution according to NPA554 is not yet implemented!!"
         write(*,*) "use another density distribution by setting densitySwitch_Static=1 in nl &target!!"
         write(*,*) z

         call Traceback('Stop')
      end select
    end subroutine denspar

  end subroutine densityLuis


  !****************************************************************************
  !****s* densityStatic/TabulateDensityBirger
  ! NAME
  ! subroutine TabulateDensityBirger(nuc)
  ! PURPOSE
  ! Tabulate the density distribution based on a local density approximation
  ! first described by Brueckner et al.
  !
  ! INPUTS
  ! * type(tNucleus) :: nuc
  !
  ! OUTPUT
  ! * type(tNucleus) :: nuc
  !
  !****************************************************************************
  subroutine TabulateDensityBirger(nuc)
    use nucDLDA, only: DFLDA
    use output

    type(tNucleus),pointer :: nuc

    write(*,paragraph) 'Initializing density tabulation, LDA by Birger'
    write(*,'(A,F8.4)') '    dr   =',nuc%dx
    write(*,'(A,F8.4)') '    r_max=',nuc%dx*float(nuc%maxIndex)

    call DFLDA(nuc)
    nuc%densTab(:,0) = nuc%densTab(:,1)+nuc%densTab(:,2)

    write(*,*) 'Chemical Potential', nuc%chemPot

    write(*,paragraph) 'Finished initializing density tabulation'
    write(*,*)


  end subroutine TabulateDensityBirger

  !****************************************************************************
  !****s* densityStatic/TabulateDensityBirgerWelke
  ! NAME
  ! subroutine TabulateDensityBirgerWelke(nuc)
  ! PURPOSE
  ! Tabulate the density distribution based on a local density approximation
  ! first described by Brueckner et al. and a momentum-dependent potential
  !
  ! INPUTS
  ! * type(tNucleus) :: nuc
  !
  ! OUTPUT
  ! * type(tNucleus) :: nuc
  !
  !****************************************************************************
  subroutine TabulateDensityBirgerWelke(nuc)
    use nucDLDA, only: DFLDAWelke
    use output

    type(tNucleus),pointer :: nuc

    write(*,paragraph) 'Initializing density tabulation, LDA and Welke potential by Birger'
    write(*,'(A,F8.4)') '    dr   =',nuc%dx
    write(*,'(A,F8.4)') '    r_max=',nuc%dx*float(nuc%maxIndex)

    call DFLDAWelke(nuc)
    nuc%densTab(:,0) = nuc%densTab(:,1)+nuc%densTab(:,2)

    write(*,*) 'Chemical Potential', nuc%chemPot

    write(*,paragraph) 'Finished initializing density tabulation'
    write(*,*)


  end subroutine TabulateDensityBirgerWelke

  !****************************************************************************
  !****s* densityStatic/ReAdjust
  ! NAME
  ! subroutine ReAdjust(nuc, potP, potN, potC)
  !
  ! PURPOSE
  ! This routine recalculates the density distributions for protons and
  ! neutrons by considering the given potentials as static and fulfill the
  ! condition
  !   sqrt(p_F^2+m_N^2) + U - m_N == E_sep ~ -8MeV
  ! With the Local-Thomas-Fermi, we connect the resulting fermi momentum to
  ! a density,
  !   rho = p_F^3/(3pi^2)
  ! Since the potentials are given as function of r, we calculate rho(r).
  !
  ! Thus, given proton and nucleon baryon potential (for fixed momentum) and
  ! the coulomb potential, the parametrization of the nuclear density is
  ! readjusted.
  !
  ! This routine is called by
  ! baryonPotentialMain/HandPotentialToDensityStatic
  !
  !
  ! INPUTS
  ! * type(tNucleus),pointer :: nuc -- the nucleus to consider
  ! * real, dimension(0:) :: potP, potN -- the proton,neutron potentials
  !   with p=pF. The dimension has to be identical to nuc%densTab(0: ,1:2).
  ! * real, dimension(0:) :: potC -- The Coulomb potential (>0, in GeV)
  !
  ! OUTPUT
  ! * nuc%densTab(0: ,1:2) is changed
  !****************************************************************************
  subroutine ReAdjust(nuc, potP, potN, potC)
    use constants
    use particleDefinition
    use output, only: IntToChar

    logical, parameter :: verbose = .false.

    type(tNucleus),pointer :: nuc
    real, dimension(0:), intent(in) :: potP, potN, potC

    integer :: i
    real :: x, rho, pF
    real :: pFN, pFP
    integer, save :: nCall = 0
    real, dimension(2) :: Sum,fac

    nCall = nCall+1

    if (verbose) then
       write(*,*) 'in Readjust...'
       call printFile("orig")
    end if

    Sum = 0
    do i=0,nuc%MaxIndex
       x = i*nuc%dx

!       pF = 2*mN*(-potP(i)+nuc%ConstBinding)-potP(i)**2+nuc%ConstBinding**2
       pF = (-(potP(i)+potC(i))+nuc%ConstBinding+mN)**2 - mN**2
       if (pF.lt.0.0) then
          nuc%densTab(i,1) = 0.0
       else
          pF = sqrt(pF)
          rho = (pF/hbarc)**3/(3*pi**2)
          nuc%densTab(i,1) = rho
          Sum(1) = Sum(1) + x**2*rho
       end if


!       pF = 2*mN*(-potN(i)+nuc%ConstBinding)-potN(i)**2+nuc%ConstBinding**2
       pF = (-potN(i)+nuc%ConstBinding+mN)**2 - mN**2
       if (pF.lt.0.0) then
          nuc%densTab(i,2) = 0.0
       else
          pF = sqrt(pF)
          rho = (pF/hbarc)**3/(3*pi**2)
          nuc%densTab(i,2) = rho
          Sum(2) = Sum(2) + x**2*rho
       end if
    end do

    if ((Sum(1).eq.0).or.(Sum(2).eq.0)) then
       call Traceback('Failure!')
    end if

    Sum = Sum*4*pi*nuc%dx

    if (verbose) then
       write(*,*) 'P: ',Sum(1),nuc%charge
       write(*,*) 'N: ',Sum(2),nuc%mass-nuc%charge
    end if


    ! Ensure normalization to mass number:

    fac(1)=(nuc%charge)/Sum(1)
    fac(2)=(nuc%mass-nuc%charge)/Sum(2)

    ! method 1: just set the normalization

    if (verbose) then
       write(*,*) 'Scaling rhoN by ',fac(2)
       write(*,*) 'Scaling rhoP by ',fac(1)
    end if

    nuc%densTab(:,2)=nuc%densTab(:,2)*fac(2)
    nuc%densTab(:,1)=nuc%densTab(:,1)*fac(1)
    nuc%densTab(:,0)=nuc%densTab(:,1)+nuc%densTab(:,2)

    nuc%fac = fac

    ! method 2: rescale the radius

!    write(*,*) 'Scaling x by ',facN**(1./3.)
!    nuc%dx = nuc%dx*facN**(1./3.)


    call nuc%searchMaxVals()
    call nuc%writeAverageDens()

    if (verbose) then
       call printFile("new")
    end if

  contains
    subroutine printFile(text)
      character(*), intent(in) :: text

      open(113,file='ReAdjust.'//trim(text)//'.'//IntToChar(nCall)//'.dat', &
           status='unknown')
      rewind(113)
      write(113,'("#",A12,10A13)') 'x','rhoN','rhoP','potN','potP','potC',&
           'pF_N','pF_P','pF_B'
      do i=0,nuc%MaxIndex
         x = i*nuc%dx
         pFN = (3*pi**2*(nuc%densTab(i,2)))**(1./3.)*hbarc
         pFP = (3*pi**2*(nuc%densTab(i,1)))**(1./3.)*hbarc
         pF  = (3*pi**2*(nuc%densTab(i,0))/2)**(1./3.)*hbarc
         write(113,'(10f13.5)') x,nuc%densTab(i,2),nuc%densTab(i,1),&
              PotN(i),PotP(i),PotC(i), pFN, pFP, pF
      end do
      close(113)
    end subroutine printFile

  end subroutine ReAdjust



end module densityStatic
