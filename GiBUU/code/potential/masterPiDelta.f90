!*******************************************************************************
!****m* /masterPiDelta
! NAME
! module masterPiDelta
!
! PURPOSE
! This module provides a control parameter for all pion-Delta
! in-medium properties
!
! NOTES
! The switch master_piDelta_inmed is read in in the module inputGeneral.
! It allows to turn on or off all in-medium properties of the pion-Delta system,
! the pion potential based on the Nieves-Oset theory, the
! 2N absorption (consistent with the pion potential) and the
! inmedium broadening of the Delta resonance, following Salecedo et al.
!
! This module is used to store this parameter. It is implemented as independent
! module to prevent circular dependencies.
!******************************************************************************
module masterPiDelta

  implicit none
  private

  logical, save :: switchIsSet = .false.
  logical, save :: master_piDelta_inmed = .false.

  public :: setMasterPiDelta
  public :: getMasterPiDelta

contains
  !****************************************************************************
  !****s* masterPiDelta/setMasterPiDelta
  ! NAME
  ! subroutine setMasterPiDelta(v)
  ! PURPOSE
  ! set pionDelta_inmed to the value v
  !****************************************************************************
  subroutine setMasterPiDelta(v)
    logical, intent(in) :: v
    master_piDelta_inmed = v
    switchIsSet = .true.
  end subroutine setMasterPiDelta

  !****************************************************************************
  !****f* masterPiDelta/getMasterPiDelta
  ! NAME
  ! logical function getMasterPiDelta()
  ! PURPOSE
  ! get the value of master_piDelta_inmed
  !****************************************************************************
  logical function getMasterPiDelta()
    use callstack, only: traceback
    if (.not.switchIsSet) then
       call traceback("Switch master_piDelta_inmed not set! Stop")
    end if
    getMasterPiDelta = master_piDelta_inmed
  end function getMasterPiDelta

end module masterPiDelta
