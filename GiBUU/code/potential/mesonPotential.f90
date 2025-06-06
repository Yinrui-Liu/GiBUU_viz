!******************************************************************************
!****m* /mesonPotentialMain
! NAME
! module mesonPotentialMain
! PURPOSE
! Includes all information about the mesonic potentials.
!******************************************************************************
module mesonPotentialMain
  implicit none
  private

  !****************************************************************************
  !****g* mesonPotentialMain/vectorMesonPot
  ! SOURCE
  !
  integer, save :: vectorMesonPot = 0
  !
  ! PURPOSE
  ! Switch for medium-modification of vector mesons:
  ! * 0 = no modification
  ! * 1 = Brown-Rho-Scaling
  ! * 2 = Brown-Rho-Scaling with momentum dependence
  !   according to Kondtradyuk (see page 162 in Effenberger's thesis).
  !   Currently not available!
  ! NOTES
  ! Can be set in namelist mesonPotential.
  !****************************************************************************

  !****************************************************************************
  !****g* mesonPotentialMain/brownRho
  ! SOURCE
  !
  real, save :: brownRho = 0.16
  !
  ! PURPOSE
  ! Brown-Rho scaling parameter alpha.
  !****************************************************************************

  !****************************************************************************
  !****g* mesonPotentialMain/noPerturbativePotential
  ! SOURCE
  !
  logical,save :: noPerturbativePotential=.false.
  !
  ! PURPOSE
  ! Switch for potential of perturbative particles.
  ! If .true. then perturbative mesons feel no potential.
  ! NOTES
  ! Can be set in namelist mesonPotential.
  !****************************************************************************

  !****************************************************************************
  !****g* mesonPotentialMain/pionPot_Switch
  ! PURPOSE
  ! Switch for pion potential:
  ! * 0 = no potential
  ! * 1 = Oset potential (NPA 554), which is valid up to 70 MeV kinetic energy
  ! * 2 = Kapusta suggestion for pion potential (rather unusual)
  ! * 3 = Delta-Hole potential, which is valid up to 130 MeV kinetic energy
  ! * 4 = Smooth spline transition between switch 1 and 3.
  ! SOURCE
  !
  integer, save :: pionPot_Switch = 0
  ! NOTES
  ! Can be set in namelist mesonPotential.
  !****************************************************************************

  logical,save :: initFlag=.true.

  public :: MesonPotential, getNoPertPot_meson, vecMes_massShift

contains

  !****************************************************************************
  !****f* mesonPotentialMain/getNoPertPot_meson
  ! NAME
  ! logical function getNoPertPot_meson()
  ! PURPOSE
  ! Return the value of 'noPerturbativePotential'.
  !****************************************************************************
  logical function getNoPertPot_meson()
    if (initFlag) call init
    getNoPertPot_meson = noPerturbativePotential
  end function getNoPertPot_meson


  !****************************************************************************
  !****s* mesonPotentialMain/init
  ! NAME
  ! subroutine init
  ! PURPOSE
  ! Reads input out of namelist 'mesonPotential' in jobcard.
  !****************************************************************************
  subroutine init

    use output, only: Write_ReadingInput
    use masterPiDelta, only: getMasterPiDelta

    integer :: ios

    !**************************************************************************
    !****n* mesonPotentialMain/mesonPotential
    ! NAME
    ! NAMELIST /mesonPotential/
    ! PURPOSE
    ! Includes switches:
    ! * pionPot_Switch
    ! * noPerturbativePotential
    ! * vectorMesonPot
    ! * brownRho
    !**************************************************************************
    NAMELIST /mesonPotential/ pionPot_Switch, noPerturbativePotential, &
         vectormesonPot, brownRho

    call Write_ReadingInput('mesonPotential',0)
    rewind(5)
    read(5,nml=mesonPotential,iostat=ios)
    call Write_ReadingInput('mesonPotential',0,ios)
    write(*,*) 'vectormesonPot          = ',vectormesonPot
    write(*,*) 'mass shift parameter    = ',brownRho
    write(*,*) 'pionPot_Switch          = ',pionPot_Switch
    write(*,*) 'noPerturbativePotential =', noPerturbativePotential
    call Write_ReadingInput('mesonPotential',1)

    if (getMasterPiDelta()) then
        pionPot_Switch = 4
        write(*,*) 'getMasterPiDelta: pionpot_Switch set to 4 in mesonPotential'
    end if

    initFlag=.false.

  end subroutine init


  !****************************************************************************
  !****f* mesonPotentialMain/vecMes_massShift
  ! NAME
  ! real function vecMes_massShift(ID, rho)
  ! PURPOSE
  ! Return the density-dependent mass shift of the vector mesons.
  ! INPUTS
  ! * integer :: ID -- the particle ID
  ! * real :: rho -- the density
  !****************************************************************************
  real function vecMes_massShift(ID, rho) result (pot)
    use constants, only: rhoNull
    use particleProperties, only: hadron
    use CallStack, only: TRACEBACK

    integer, intent(in) :: ID
    real, intent(in) :: rho
    real :: mass

    if (initFlag) call init

    mass=hadron(ID)%mass

    select case (vectorMesonPot)
    case (0)
       pot = 0.

    case (1) !=== Brown-Rho-Scaling ===
       pot = -mass*brownRho*rho/rhoNull

    case (2) !=== Brown-Rho-Scaling with momentum dependence ===
       ! according to Kondtradyuk (page 162,effenberger dr.-thesis)
       !( 0.1-pAbs) because potential is assumed to vanish around 0.1 GeV
       ! momentum.
       ! Seems like it is rather 0.1 GeV than 1 GeV.
       ! This is rather crude. Has to be improved!!!
       ! pot=-mass*brownRho*rho/rhoNull*(0.1-pAbs)

       call TRACEBACK("vectorMesonPot=2 has been deleted")

    case default
       write(*,*) "Error: Value of vectorMesonPot is invalid: ", vectorMesonPot
       call TRACEBACK()

    end select

  end function


  !****************************************************************************
  !****f* mesonPotentialMain/MesonPotential
  ! NAME
  ! real function MesonPotential(Part, med)
  ! PURPOSE
  ! Meson potential is defined as 0th component of a vector potential in
  ! the local rest frame (LRF).
  ! INPUTS
  ! * type(particle) :: Part  -- particle boosted to LRF
  ! * type(medium)   :: med       -- medium info in LRF
  ! OUTPUT
  ! * real :: Meson_Potential  -- in GeV
  ! NOTES
  ! In contrast to the old BUU code, there is no kBar-potential implemented.
  ! This potential is described in Effenberger's thesis on pages 207-216.
  !****************************************************************************
  real function MesonPotential(Part, med)

    use IDTable, only: pion, rho, omegaMeson, phi
    use ParticleDefinition
    use mediumDefinition
    use pionPot, only: pionPot_Main
    use constants, only: mPi

    type(particle), intent(in) :: Part
    type(medium),   intent(in) :: med

    real :: pAbs, scapot

    if (initFlag) call init
    MesonPotential = 0.

    if (noPerturbativePotential.and.Part%pert) return

    select case (Part%ID)

    case (rho,omegaMeson,phi) ! === nonstrange vector mesons ===
       pAbs=absMom(Part) ! absolute momentum
       scapot=vecMes_massShift(Part%ID,med%density)
       MesonPotential=sqrt((Part%mass+scapot)**2+pAbs**2)-sqrt(Part%mass**2+pAbs**2)

    case (Pion)
       pAbs=absMom(Part) ! absolute momentum
       MesonPotential=pionPot_Main(mPi,Part%mom(1:3),med%densityProton,med%densityNeutron,Part%charge,pionPot_Switch)

    case default  ! particle with no potential
       return

    end select

    ! Check for consistency:
    if (MesonPotential < (-1.)*SQRT(Part%mass**2+pAbs**2)) then
       write(*,*) 'Error in Meson_Potential!!!'
       write(*,*) 'Total energy for meson less than 0 in LRF!'
       write(*,*) 'ID:',Part%ID
       write(*,*) '4-Momentum:',Part%mom
       write(*,*) 'Mass:',Part%mass
       write(*,*) 'MesonPotential =',MesonPotential
       write(*,*) 'scapot =',scapot
       stop
    end if

  end function MesonPotential


end module mesonPotentialMain
