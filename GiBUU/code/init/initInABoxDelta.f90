!******************************************************************************
!****m* /initInABoxDelta
! NAME
! module initInABoxDelta
! PURPOSE
! Intializes Delta baryons to be used in a box filled with nucleons.
!
! The nucleons have to be initialized by initInABox/initializeInABox
!******************************************************************************
module initInABoxDelta

  implicit none
  private

  !****************************************************************************
  !****g* initInABoxDelta/mass
  ! SOURCE
  real, save :: mass = 1.2
  ! PURPOSE
  ! rest mass of Deltas [GeV]
  !****************************************************************************

  !****************************************************************************
  !****g* initInABoxDelta/charge
  ! SOURCE
  integer, save :: charge = +1
  ! PURPOSE
  ! charge of Deltas
  !****************************************************************************

  !****************************************************************************
  !****g* initInABoxDelta/mom
  ! SOURCE
  real, save :: mom = 0.1
  ! PURPOSE
  ! momentum of Deltas (in z direction) [GeV]
  !****************************************************************************

  !****************************************************************************
  !****g* initInABoxDelta/nDelta
  ! SOURCE
  integer, save :: nDelta = 50
  ! PURPOSE
  ! number of Deltas per ensemble
  !****************************************************************************

  public :: initializeInABoxDelta

  logical, save :: initFlag=.true.

contains

  !****************************************************************************
  !****is* initInABoxDelta/initInput
  ! NAME
  ! subroutine initInput
  ! PURPOSE
  ! Reads input out of jobcard. Namelist 'InABoxDelta'.
  !****************************************************************************
  subroutine initInput

    use output, only: Write_ReadingInput
    use callstack, only: traceBack

    integer :: ios

    !**************************************************************************
    !****n* initInABoxDelta/InABoxDelta
    ! NAME
    ! NAMELIST InABoxDelta
    ! PURPOSE
    ! Includes the input parameters:
    ! * mass
    ! * charge
    ! * mom
    ! * nDelta
    !**************************************************************************
    NAMELIST /InABoxDelta/ mass, charge, mom, nDelta

    call Write_ReadingInput('InABoxDelta',0)
    rewind(5)
    read(5,nml=InABoxDelta,iostat=ios)
    call Write_ReadingInput('InABoxDelta',0,ios)

    write(*,*) ' Mass   = ', mass
    write(*,*) ' Charge = ', charge
    write(*,*) ' Mom    = ', mom
    write(*,*) ' nDelta = ', nDelta

    call Write_ReadingInput('InABoxDelta',1)

    initFlag = .false.
  end subroutine initInput



  !****************************************************************************
  !****s* initInABoxDelta/initializeInABoxDelta
  ! NAME
  ! subroutine initializeInABoxDelta(part,flag)
  !
  ! PURPOSE
  ! Initialize Deltas
  !
  ! selection:
  ! * position is choosen as (-1..1,-1..1,0)
  ! * momentum is choosen as (0.,0.,mom)
  !
  ! INPUTS
  ! * type(particle), dimension(:,:) :: part -- particle vector to use
  ! * logical :: flag -- perturbative flag
  !
  ! OUTPUT
  ! * type(particle), dimension(:,:) :: part -- particle vector to use
  !
  ! NOTES
  ! Overwrites the given particle vector
  !****************************************************************************
  subroutine initializeInABoxDelta(part,flag)

    use particleDefinition
    use random, only: rnFlat
    use idTable, only: delta

    integer :: i,j
    type(particle), dimension(:,:) :: part
    logical, intent(in) :: flag

    if (initFlag) call initInput

    do i=lbound(part,dim=1),ubound(part,dim=1)
       call setToDefault(part(i,:))
       do j=1,nDelta
          part(i,j)%ID       = delta
          part(i,j)%charge   = charge
          part(i,j)%mass     = mass
          part(i,j)%mom(1:3) = (/ 0., 0., mom /)
          part(i,j)%mom(0)   = sqrt(mass**2 + mom**2)
          part(i,j)%pos(1:3) = (/ rnFlat(1.,1.), rnFlat(-1.,1.), 0. /)
          part(i,j)%pert     = flag
       end do
    end do
  end subroutine initializeInABoxDelta


end module initInABoxDelta
