!******************************************************************************
!****m* /createPPSmear
! NAME
! module createPPSmear
! PURPOSE
! This module creates a random generator for smearing the momenta according
! a 2D array
!
! The file 'Mom_Smearing_Data_PP.txt' was created by opening the file
! mailed by Karina Scharmann in root,
! > root Whole__Jul23_MomentumSmearing_.root
! and then piping the content by calling
! $ .> abc2.dat
! $ Mom_plot2->Print("all")
! $.>
! into a text file. This was then opened in emacs and transformed by
! query'n'replace into the final file.
!
! The original file has the dimension [0:1501,0:1001], representing
! the ranges 0...1.5GeV and 0...2 for the ratio pReco/pIn
!
! The indices 1501 and 1001 seem to represent 'Randverteilungen', i.e.
! the integrated bins in the resp. direction.
!
! This will be reduced to the 300x200 size used by Jan-Hendrik by just
! adding adjoint bins, so that the reading of the file is the same.
!
!******************************************************************************
module createPPSmear

  implicit none

  private

  real(4), dimension(0:200,0:300) ::  Arr !    indizes: ratio, pIn

  public :: fillArr
  public :: integrateArr
  public :: printLine
  public :: bisectLine
  public :: dumpArr


contains

  subroutine setBinContent(i1,i2,pIn,ratio,val)
    real, intent(in) :: pIn,ratio,val
    integer, intent(in) :: i1,i2

    integer :: ix,iy
    real(4) :: val4

    if (i1 < 0) return
    if (i2 < 0) return
    if (i1 >= 1500) return
    if (i2 >= 1000) return

!    ix = int(pIn/5.0)+1
!    iy = int(ratio/0.01)+1

    ix = int(i1/5.0)+1
    iy = int(i2/5.0)+1


    val4 = val*1.0

    if ((ix>300).or.(iy>200)) then
       write(*,*) "Error: ",i1,i2,pIn,ratio,ix,iy
       stop
    end if

    !    write(*,*) pIn,ratio,ix,iy,val4

    Arr(iy,ix) = Arr(iy,ix) + val4

  end subroutine setBinContent

  subroutine fillArr
    integer :: ios

    integer:: i1,i2
    real :: pIn,ratio, val

    open(10,file="Mom_Smearing_Data_PP.txt",status="OLD")

    Arr = 0.
    do
       read(10,*,IOSTAT=ios) i1, i2, val, pIn, ratio
       if (ios==-1) exit
       call setBinContent(i1, i2, pIn,ratio,val)
    end do

    close(10)

  end subroutine fillArr

  subroutine integrateArr
    integer :: iLine,i
    real(4) :: s

    do iLine=1,300
       s = 0.
       do i=0,200
          s = s + Arr(i,iLine)
          Arr(i,iLine) = s
       end do
       Arr(:,iLine) = Arr(:,iLine) * (1./s)
    end do
  end subroutine integrateArr

  subroutine printLine(iLine)
    integer, intent(in) :: iLine

    integer :: i

    do i=0,200
       write(106,*) iLine*5.0,i*0.01,Arr(i,iLine)
    end do
    write(106,*)
    write(106,*)

  end subroutine printLine

  integer function bisectLine(iLine, val8)
    integer, intent(in) :: iLine
    real, intent(in) :: val8

    integer :: i1,iM,i2
    real(4) :: val

    val=val8
    i1 = 0
    i2 = 200

    if (val < Arr(i1,iLine)) then
       write(*,*) 'Error 1'
       stop
    end if
    if (val > Arr(i2,iLine)) then
       write(*,*) 'Error 2'
       stop
    end if

    do while(i2-i1 > 1)
       iM = (i1+i2)/2
       if (val > Arr(iM,iLine)) then
          i1 = iM
       else
          i2 = iM
       end if
    end do

    bisectLine = i1

  end function bisectLine

  subroutine dumpArr

    open(10,file="Mom_Smearing_Data_PP.bin",status="UNKNOWN",form="UNFORMATTED")
    write(10) Arr
    close(10)

  end subroutine dumpArr


end module createPPSmear


program usePPSmear

  use createPPSmear, only: fillArr, integrateArr, printLine, bisectLine, &
       dumpArr

  implicit none

  call fillArr

  ! for the plotting:
  call printLine(100)
  call printLine(200)
  call printLine(300)


  call integrateArr
!  call printLine(200)

  call dumpArr


  ! some test examples:
  write(*,*) bisectLine(200, 0.1)
  write(*,*) bisectLine(200, 0.2)
  write(*,*) bisectLine(200, 0.5)
  write(*,*) bisectLine(200, 0.7)
  write(*,*) bisectLine(200, 0.9)





  ! How to generate a random number:
!!$  r = ran()
!!$  randomIndex = bisectLine(iLine, r)


end program usePPSmear
