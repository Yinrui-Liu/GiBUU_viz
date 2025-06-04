!******************************************************************************
!****m* /InABoxDeltaAnalysis
! NAME
! module InABoxDeltaAnalysis
! PURPOSE
! Modules includes routines which are used to evaluate the decay width (gamma)
! of the Delta
!******************************************************************************
module InABoxDeltaAnalysis
  implicit none
  private

  public :: InABoxDeltaAnalysis_count
  public :: InABoxDeltaAnalysis_eval

contains

  !****************************************************************************
  !****s* InABoxDeltaAnalysis/InABoxDeltaAnalysis_count
  ! NAME
  ! subroutine InABoxDeltaAnalysis_count(part,time)
  ! INPUTS
  ! * type(particle), dimension(:,:),intent(in) :: part
  ! * real,intent(in) :: time
  ! PURPOSE
  ! Counts number of Deltas in the particleVector and prints it to file
  ! "numDeltas.dat"
  !
  ! The first number is the total number of Deltas, the second one the number
  ! of Deltas, which have not yet undergone any collision.
  !****************************************************************************
  subroutine InABoxDeltaAnalysis_count(part,time)

    use particleDefinition
    use random
    use idTable, only: delta

    type(particle), dimension(:,:), intent(in) :: part
    real, intent(in) :: time

    integer :: i,j
    integer :: numDeltas(1:2)

    numDeltas=0
    do i=lbound(part,dim=1),ubound(part,dim=1)
       do j=lbound(part,dim=2),ubound(part,dim=2)
          if (part(i,j)%ID.eq.delta) then
             numDeltas(1) = numDeltas(1)+1
             if (part(i,j)%event(1).eq.0) then
                numDeltas(2) = numDeltas(2)+1
             end if
          end if
       end do
    end do
    open(120,file="numDeltas.dat",position='append')
    write(120,*) time, numDeltas
    close(120)

  end subroutine InABoxDeltaAnalysis_count


  !****************************************************************************
  !****s* InABoxDeltaAnalysis/InABoxDeltaAnalysis_eval
  ! NAME
  ! subroutine InABoxDeltaAnalysis_eval()
  ! INPUTS
  !
  ! PURPOSE
  ! perform analysis at the end of the run
  !****************************************************************************
  subroutine InABoxDeltaAnalysis_eval()
    use particleDefinition
    use twoBodyStatistics, only: rate

    type(particle), dimension(1:2) :: dummy

    call rate(dummy,dummy,-99.9,.true.)

  end subroutine InABoxDeltaAnalysis_eval



end module InABoxDeltaAnalysis
